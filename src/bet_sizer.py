# bet_sizer_ce_kelly.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from collections import Counter

from ev_engine import BlackjackEVEngine
from game_mechanics.rules import BlackjackRuleset, RULES, RANKS


@dataclass
class BetRecommendation:
    bet_size: float
    kelly_fraction_raw: float  # k = EV / Var
    kelly_fraction_ce: float  # CE-Kelly with risk_tolerance
    kelly_fraction_applied: float  # after fraction-Kelly scaling & caps
    edge: float
    variance: float
    stdev: float
    max_exposure_units: int
    explanation: str


class BlackjackBetSizerCEKelly:
    def __init__(
        self,
        rules: Optional[BlackjackRuleset] = None,
        *,
        risk_tolerance: float = 1.0,  # ρ in CE-Kelly (0 = pure Kelly; larger = more conservative, 1.0 is considered aggressive)
        fraction_kelly: float = 0.5,  # extra safety multiplier (½-Kelly default)
        risk_cap_per_hand: float = 0.01,  # worst-case bankroll risk cap per hand (e.g., 1%)
        table_min: float = 10.0,
        table_max: float = 1000.0,
        bet_increment: float = 5.0,  # bet must be rounded down to this increment
        max_splits: int = 3,  # up to 4 hands total
        assume_split_independence: bool = True,
        shoe_max_counts: Optional[Dict[str, int]] = None,
        use_comprehensive_cache: bool = True,
    ):
        self.rules = rules or RULES
        self.use_comprehensive_cache = use_comprehensive_cache

        # Try to load split cache first, then comprehensive cache, then standard engine
        if self.use_comprehensive_cache:
            try:
                from split_cache import HandTypeSplitCache

                cache_loader = HandTypeSplitCache(self.rules)
                self.engine = cache_loader.load_cache()
                print("Using hand-type split pre-computed cache (ultra-fast)")
            except (FileNotFoundError, ValueError, ImportError) as e:
                try:
                    from comprehensive_cache import Comprehensive8DeckCache

                    cache_loader = Comprehensive8DeckCache(self.rules)
                    self.engine = cache_loader.load_cache()
                    print("Using comprehensive pre-computed cache")
                except (FileNotFoundError, ValueError, ImportError) as e2:
                    print(
                        f"No pre-computed cache available ({e}, {e2}), using standard engine"
                    )
                    self.engine = BlackjackEVEngine(
                        self.rules, shoe_max_counts=shoe_max_counts
                    )
        else:
            self.engine = BlackjackEVEngine(self.rules, shoe_max_counts=shoe_max_counts)
        self.risk_tolerance = max(0.0, float(risk_tolerance))
        self.fraction_kelly = float(max(0.0, min(1.0, fraction_kelly)))
        self.risk_cap_per_hand = float(max(0.0, risk_cap_per_hand))
        self.table_min = float(table_min)
        self.table_max = float(table_max)
        self.bet_increment = float(max(0.01, bet_increment))
        self.max_splits = int(max_splits)
        self.assume_split_independence = assume_split_independence

    def recommend_bet(
        self, bankroll: float, remaining_counts: Counter[str]
    ) -> BetRecommendation:
        if bankroll <= 0:
            return BetRecommendation(
                bet_size=self.table_min,
                kelly_fraction_raw=0.0,
                kelly_fraction_ce=0.0,
                kelly_fraction_applied=0.0,
                edge=0.0,
                variance=0.0,
                stdev=0.0,
                max_exposure_units=self._max_exposure_units(),
                explanation="Bankroll ≤ 0; table minimum.",
            )

        ev, var = predeal_moments(
            self.engine, remaining_counts, self.assume_split_independence
        )
        stdev = var**0.5
        k_raw = 0.0 if var <= 0.0 or ev <= 0.0 else ev / var

        # If EV is negative, bet minimum
        if ev <= 0:
            bet = self.table_min
            k_ce = 0.0
            k_applied = 0.0
            bet_unclamped = 0.0
            exposure_cap = float("inf")
        else:
            # Certainty-Equivalent Kelly
            k_ce = k_raw / (1.0 + self.risk_tolerance * k_raw)

            # Apply extra fractional Kelly (½-Kelly default)
            k_applied = self.fraction_kelly * k_ce

            # Worst-case exposure cap (splits × doubling)
            max_units = self._max_exposure_units()
            exposure_cap = bankroll * self.risk_cap_per_hand / max(1, max_units)

            bet_unclamped = bankroll * k_applied
            bet = min(bet_unclamped, exposure_cap, self.table_max)

            # Round down to bet increment
            if self.bet_increment > 0:
                bet = (bet // self.bet_increment) * self.bet_increment

            bet = max(bet, self.table_min)

        explanation = self._explain(
            ev, var, k_raw, k_ce, k_applied, bet_unclamped, exposure_cap, bet
        )

        return BetRecommendation(
            bet_size=bet,
            kelly_fraction_raw=k_raw,
            kelly_fraction_ce=k_ce,
            kelly_fraction_applied=k_applied,
            edge=ev,
            variance=var,
            stdev=stdev,
            max_exposure_units=max_units,
            explanation=explanation,
        )

    # ---------- helpers ----------

    def _max_exposure_units(self) -> int:
        max_hands = min(2**self.max_splits, 4)
        doubling_factor = 2 if self.rules.das else 1
        return max_hands * doubling_factor

    def _explain(
        self, ev, var, k_raw, k_ce, k_applied, bet_unclamped, exposure_cap, bet
    ) -> str:
        parts = []
        if ev <= 0:
            parts.append(f"No edge (EV {ev:.2%}).")
        else:
            parts.append(f"Edge {ev:.2%}, Var {var:.3f}, SD {var**0.5:.3f}.")
            parts.append(
                f"Kelly k={k_raw:.2%}, CE-Kelly (ρ={self.risk_tolerance:.2f})={k_ce:.2%}, applied={k_applied:.2%}."
            )
        if bet_unclamped > exposure_cap + 1e-9:
            parts.append("Capped by worst-case exposure limit.")
        if bet >= self.table_max - 1e-9:
            parts.append("Capped at table max.")
        if bet <= self.table_min + 1e-9:
            parts.append("At table min.")
        return " ".join(parts)


# ---- exact predeal moments (reuse from previous message) ----


def predeal_moments(
    engine: BlackjackEVEngine,
    shoe_counts: Counter[str],
    assume_split_independence: bool = True,
) -> Tuple[float, float]:
    """
    Fast predeal moments calculation using comprehensive pre-computed cache.
    Falls back to original calculation if cache miss occurs.
    """
    N = sum(shoe_counts.values())
    if N < 3:
        return 0.0, 0.0

    ev_sum = 0.0
    m2_sum = 0.0
    cache_hits = 0
    cache_misses = 0

    for u in RANKS:
        cu = shoe_counts.get(u, 0)
        if cu <= 0:
            continue
        p_u = cu / N
        deck_u = shoe_counts.copy()
        deck_u[u] -= 1
        Nu = N - 1
        if Nu < 2:
            continue
        for p1 in RANKS:
            c1 = deck_u.get(p1, 0)
            if c1 <= 0:
                continue
            p_p1 = c1 / Nu
            deck_u1 = deck_u.copy()
            deck_u1[p1] -= 1
            Nu1 = Nu - 1
            if Nu1 < 1:
                continue
            for p2 in RANKS:
                c2 = deck_u1.get(p2, 0)
                if c2 <= 0:
                    continue
                p_p2 = c2 / Nu1
                from game_mechanics import HandState

                hand = HandState.from_cards([p1, p2])
                deck_after = deck_u1.copy()
                deck_after[p2] -= 1

                # Use the engine's existing cache - it should be pre-warmed
                ev, var, _sd, _act = engine.evaluate_with_volatility(
                    hand=hand,
                    dealer_up=u,
                    remaining_counts=deck_after,
                    can_double=True,
                    can_cashout=engine.rules.can_cashout,
                    can_split=True,
                    split_level=0,
                    assume_split_independence=assume_split_independence,
                )

                w = p_u * p_p1 * p_p2
                ev_sum += w * ev
                m2_sum += w * (var + ev * ev)

    predeal_ev = ev_sum
    predeal_var = max(0.0, m2_sum - predeal_ev * predeal_ev)
    return predeal_ev, predeal_var
