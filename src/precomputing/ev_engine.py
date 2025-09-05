from __future__ import annotations
import sys
from pathlib import Path

# Add parent directory to path so we can import from sibling modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from collections import Counter, defaultdict

# Import your existing mechanics
# - HandState must provide: total, is_soft, card_count, can_split, is_terminal,
#   is_blackjack, is_busted, cards (iterable of rank strings), and a factory `from_cards(list_of_ranks)`
# - ActionType is an Enum with HIT, STAND, DOUBLE, SPLIT, SURRENDER, INSURANCE
# - CardTransitionCalculator provides:
#     get_hit_transitions(hand: HandState, remaining: Counter[str]) -> Dict[str, HandState]
#     get_split_transitions(hand: HandState, remaining: Counter[str]) -> Dict[Tuple[str,str], Tuple[HandState, HandState]]
from game_mechanics import HandState, ActionType, CardTransitionCalculator
from game_mechanics.rules import (
    BlackjackRuleset,
    RULES,
    RANKS,
    TEN_BUCKET,
    CASHOUT_EV as SURRENDER_EV,
    BLACKJACK_PAYOUT,
)


@dataclass(frozen=True)
class PlayerKey:
    # Minimal, canonical hand descriptor for caching subproblems
    is_pair: bool
    pair_rank: Optional[str]
    total: int
    soft: bool
    num_cards: int
    split_level: int
    # Flags that affect action set legality for THIS node
    can_double: bool
    can_surrender: bool
    can_split: bool

    @staticmethod
    def from_hand(
        hand: HandState,
        split_level: int,
        can_double: bool,
        can_surrender: bool,
        can_split: bool,
    ) -> "PlayerKey":
        is_pair = getattr(hand, "can_split", False)
        pair_rank = None
        if is_pair and getattr(hand, "card_count", 0) == 2:
            # infer pair rank from first card (your HandState likely keeps the cards)
            cards = list(hand.cards)
            pair_rank = cards[0]
        return PlayerKey(
            is_pair=is_pair,
            pair_rank=pair_rank,
            total=hand.total,
            soft=hand.is_soft,
            num_cards=hand.card_count,
            split_level=split_level,
            can_double=can_double,
            can_surrender=can_surrender,
            can_split=can_split,
        )


# ---------------------------
# Engine with global caches
# ---------------------------


class BlackjackEVEngine:
    """
    Exact without-replacement EV engine for a single-situation query:
      evaluate(hand, dealer_up, remaining_counts, flags...) -> (ev, best_action)

    - Global memoization by mixed-radix deck key + player key + upcard + rules
    - Separate dealer-final distribution cache (major speedup)
    - No prop drilling: engine owns caches and uses a reversible "draw/undo" stack

    Also supports (E[X], Var[X]) under the optimal policy via evaluate_with_volatility(...).
    """

    def __init__(
        self,
        rules: BlackjackRuleset | None = None,
        shoe_max_counts: Optional[Dict[str, int]] = None,
    ):
        self.rules = rules or RULES
        self.calc = CardTransitionCalculator()

        # Caches
        # Player moments: (player_key, dealer_up, rules_mask, deck_key) -> (m1, m2, best_action)
        self.ev_cache: Dict[
            Tuple[PlayerKey, str, int, int], Tuple[float, float, ActionType]
        ] = {}
        # Dealer distribution: (dealer_up, rules_mask, deck_key) -> dict{final_total or "BUST": prob}
        # Used for settlement in both EV and moments.
        self.dealer_cache: Dict[Tuple[str, int, int], Dict[int, float]] = {}

        # Live deck state (set on each evaluate() call)
        self.deck: List[int] = [0] * len(RANKS)  # counts per RANKS index
        self.N: int = 0
        self.deck_key: int = 0

        # Mixed-radix strides (depend on MAX counts per rank in the shoe)
        # If not provided, we’ll infer max as “current count + seen so far” dynamically per evaluate call.
        self.shoe_max_counts = (
            shoe_max_counts  # Optional: pass full-shoe caps for a stable keyspace
        )
        self.strides: List[int] = [1] * len(RANKS)

        # Draw/undo stack
        self._stack: List[Tuple[int, int, int]] = []  # (rank_idx, prev_count, prev_N)

    # ------------ Public API ------------

    def evaluate(
        self,
        hand: HandState,
        dealer_up: str,
        remaining_counts: Counter[str],
        *,
        can_double: bool = True,
        cashout_allowed: bool = True,
        can_split: bool = True,
        split_level: int = 0,
    ) -> Tuple[float, ActionType]:
        """
        Compute the *optimal* EV for the current situation.
        """
        self._load_deck(remaining_counts)
        player_key = PlayerKey.from_hand(
            hand, split_level, can_double, cashout_allowed, can_split
        )
        key = (player_key, dealer_up, self._rules_mask(), self.deck_key)

        if key in self.ev_cache:
            m1, _m2, action = self.ev_cache[key]
            return m1, action

        m1, m2, action = self._compute_moments(
            hand, player_key, dealer_up, assume_split_independence=True
        )
        self.ev_cache[key] = (m1, m2, action)
        return m1, action

    def evaluate_with_volatility(
        self,
        hand: HandState,
        dealer_up: str,
        remaining_counts: Counter[str],
        *,
        can_double: bool = True,
        cashout_allowed: bool = True,
        can_split: bool = True,
        split_level: int = 0,
        assume_split_independence: bool = True,
    ) -> Tuple[float, float, float, ActionType]:
        """
        Returns (ev, variance, stdev, best_action) under the optimal-EV policy.
        If assume_split_independence is True, second moments for split hands
        use E[(X1+X2)^2] = E[X1^2] + E[X2^2] + 2 E[X1]E[X2].
        """
        self._load_deck(remaining_counts)
        pk = PlayerKey.from_hand(
            hand, split_level, can_double, cashout_allowed, can_split
        )
        key = (pk, dealer_up, self._rules_mask(), self.deck_key)
        if key in self.ev_cache:
            m1, m2, act = self.ev_cache[key]
            var = max(0.0, m2 - m1 * m1)
            return m1, var, var**0.5, act
        m1, m2, act = self._compute_moments(
            hand, pk, dealer_up, assume_split_independence
        )
        self.ev_cache[key] = (m1, m2, act)
        var = max(0.0, m2 - m1 * m1)
        return m1, var, var**0.5, act

    # ------------ Core EV recursion ------------

    def _compute_ev(
        self, hand: HandState, pk: PlayerKey, dealer_up: str
    ) -> Tuple[float, ActionType]:
        # Terminal (player already busted or stood)
        if hand.is_busted:
            return -1.0, ActionType.STAND  # forced outcome
        if hand.is_terminal and not hand.is_blackjack:
            # A terminal non-bj state means "stand and settle now"
            return self._ev_stand(hand, dealer_up), ActionType.STAND

        # If natural blackjack present
        if hand.is_blackjack:
            # Stand and settle now (dealer can still push with blackjack if peek rules apply)
            return self._ev_stand(hand, dealer_up), ActionType.STAND

        # Evaluate legal actions
        evs: Dict[ActionType, float] = {}

        # Stand
        evs[ActionType.STAND] = self._ev_stand(hand, dealer_up)

        # Surrender Any Time (can be used at any point)
        if self.rules.cashout_allowed:
            evs[ActionType.SURRENDER_ANY_TIME] = SURRENDER_EV

        # Insurance (consider only if dealer shows Ace and rule allows; returns net EV including side bet)
        if self.rules.insurance_allowed and dealer_up == "A" and pk.num_cards == 2:
            evs[ActionType.INSURANCE] = self._ev_insurance(hand, dealer_up)

        # Double (allowed only on first decision, or if your rules differ then relax this)
        if pk.can_double and pk.num_cards == 2:
            evs[ActionType.DOUBLE] = self._ev_double(hand, dealer_up, pk)

        # Hit
        evs[ActionType.HIT] = self._ev_hit(hand, dealer_up, pk)

        # Split
        if pk.can_split and pk.is_pair and pk.num_cards == 2:
            evs[ActionType.SPLIT] = self._ev_split(hand, dealer_up, pk)

        # Pick best
        best_action = max(evs, key=lambda a: evs[a])
        return evs[best_action], best_action

    # ------------ Action EVs ------------

    def _ev_stand(self, hand: HandState, dealer_up: str) -> float:
        if hand.is_busted:
            return -1.0
        # Blackjack payout
        if hand.is_blackjack and self.rules.blackjack_payout == 1.5:
            # If dealer can also have BJ (Ace or Ten upcard), use dealer dist to capture pushes
            if dealer_up in ("A", TEN_BUCKET):
                dist = self._dealer_dist(dealer_up)
                ev = 0.0
                for total, p in dist.items():
                    if total == 21:  # dealer blackjack → push
                        ev += p * 0.0
                    elif total > 21:
                        ev += p * BLACKJACK_PAYOUT
                    else:
                        ev += p * BLACKJACK_PAYOUT
                return ev
            else:
                return BLACKJACK_PAYOUT

        # General settlement vs dealer distribution
        player = hand.total
        dist = self._dealer_dist(dealer_up)
        ev = 0.0
        for total, p in dist.items():
            if total > 21:
                ev += p * 1.0
            elif player > total:
                ev += p * 1.0
            elif player == total:
                ev += p * 0.0
            else:
                ev += p * (-1.0)
        return ev

    def _ev_hit(self, hand: HandState, dealer_up: str, pk: PlayerKey) -> float:
        total_cards = self.N
        if total_cards == 0:
            # No cards to draw; must stand as-is
            return self._ev_stand(hand, dealer_up)

        hit_transitions = self.calc.get_hit_transitions(hand, self._deck_counter())
        ev = 0.0
        for rank, next_hand in hit_transitions.items():
            idx = self._rank_index(rank)
            count = self.deck[idx]
            if count <= 0:
                continue
            p = count / total_cards

            # draw/undo
            self._draw(idx)
            child_pk = PlayerKey.from_hand(
                next_hand,
                pk.split_level,
                can_double=False,
                can_surrender=False,
                can_split=False,
            )
            key = (child_pk, dealer_up, self._rules_mask(), self.deck_key)
            if key in self.ev_cache:
                child_ev, _ = self.ev_cache[key]
            else:
                child_ev, _ = self._compute_ev(next_hand, child_pk, dealer_up)
                self.ev_cache[key] = (
                    child_ev,
                    ActionType.STAND,
                )  # action cached but not used here
            self._undo()

            ev += p * child_ev
        return ev

    def _ev_double(self, hand: HandState, dealer_up: str, pk: PlayerKey) -> float:
        total_cards = self.N
        if total_cards == 0:
            # If no cards to draw, doubling is equivalent to standing with doubled stake (unusual), disallow or treat as stand
            return self._ev_stand(hand, dealer_up)

        hit_transitions = self.calc.get_hit_transitions(hand, self._deck_counter())
        ev = 0.0
        for rank, next_hand in hit_transitions.items():
            idx = self._rank_index(rank)
            count = self.deck[idx]
            if count <= 0:
                continue
            p = count / total_cards

            self._draw(idx)
            # After double, hand is terminal and stands (2x stake)
            stand_ev = self._ev_stand(next_hand, dealer_up)
            self._undo()

            ev += p * (2.0 * stand_ev)
        return ev

    def _ev_split(self, hand: HandState, dealer_up: str, pk: PlayerKey) -> float:
        # Exact sequential evaluation: EV = E_{cards for Hand1} [ EV1 + EV2 | post-branch deck ]
        split_trans = self.calc.get_split_transitions(hand, self._deck_counter())
        total_cards = self.N
        if total_cards < 2:
            return 0.0

        ev = 0.0
        for (c1, c2), (h1, h2) in split_trans.items():
            i1, i2 = self._rank_index(c1), self._rank_index(c2)
            n1 = self.deck[i1]
            n2 = self.deck[i2]
            if n1 <= 0:
                continue
            if c1 == c2 and n1 < 2:
                continue
            if c1 != c2 and n2 <= 0:
                continue

            # Probability of the ordered pair (c1 then c2)
            if c1 == c2:
                p = (n1 / total_cards) * ((n1 - 1) / (total_cards - 1))
            else:
                p = (n1 / total_cards) * (n2 / (total_cards - 1))

            # Remove both cards and evaluate sequentially
            self._draw(i1)
            self._draw(i2)

            # Hand 1 EV
            pk1 = PlayerKey.from_hand(
                h1,
                pk.split_level + 1,
                can_double=self.rules.das,
                can_surrender=False,
                can_split=(
                    False
                    if self.rules.max_splits >= (pk.split_level + 1)
                    else h1.can_split
                ),
            )
            ev1, _ = self._compute_ev(h1, pk1, dealer_up)

            # Hand 2 uses the POST-hand-1 deck; evaluate afresh
            pk2 = PlayerKey.from_hand(
                h2,
                pk.split_level + 1,
                can_double=self.rules.das,
                can_surrender=False,
                can_split=(
                    False
                    if self.rules.max_splits >= (pk.split_level + 1)
                    else h2.can_split
                ),
            )
            ev2, _ = self._compute_ev(h2, pk2, dealer_up)

            self._undo()
            self._undo()

            ev += p * (ev1 + ev2)

        return ev

    def _ev_insurance(self, hand: HandState, dealer_up: str) -> float:
        # Only meaningful when dealer_up == 'A'
        if dealer_up != "A":
            return self._ev_stand(hand, dealer_up)

        ten_idx = self._rank_index(TEN_BUCKET)
        if self.N == 0:
            return -0.5  # lose the side bet, nothing else changes

        p_bj = self.deck[ten_idx] / self.N
        p_no = 1.0 - p_bj

        # If dealer has blackjack:
        # - Insurance pays 2:1 on 0.5 → +1 net on the side
        # - Main hand: loses -1 unless player also has BJ → push
        if hand.is_blackjack:
            v_bj = +1.0  # insurance +1, main pushes 0
        else:
            v_bj = 0.0  # insurance +1 and main -1 → net 0

        # If dealer does NOT have blackjack:
        # - Lose the 0.5 side bet, continue normal game knowing hole-card ≠ Ten.
        #   That “knowledge” is equivalent to conditioning on hole ≠ Ten, which
        #   in practice means we should *not* remove any card from the deck here.
        #   We simply evaluate stand/hit/etc. as normal; the side bet is sunk.
        v_no = -0.5 + self._ev_stand(hand, dealer_up)
        return p_bj * v_bj + p_no * v_no

    # ------------ Dealer distribution (exact, memoized) ------------

    def _dealer_dist(self, dealer_up: str) -> Dict[int, float]:
        k = (dealer_up, self._rules_mask(), self.deck_key)
        if k in self.dealer_cache:
            return self.dealer_cache[k]

        # Build dealer recursion with memo keyed by (total, soft, deck_key).
        memo: Dict[Tuple[int, bool, int], Dict[int, float]] = {}

        # Start dealer hand with upcard already on table (upcard not in deck)
        # We assume the remaining deck passed in has ALREADY excluded the upcard.
        start = HandState.from_cards([dealer_up])
        dist = self._dealer_recursive(start, memo)
        self.dealer_cache[k] = dist
        return dist

    def _dealer_recursive(
        self, hand: HandState, memo: Dict[Tuple[int, bool, int], Dict[int, float]]
    ) -> Dict[int, float]:
        # Dealer rule: stand on totals ≥ 17 (soft behavior depends on rules.s17)
        total = hand.total
        soft = hand.is_soft

        if total > 21:
            return {22: 1.0}  # use 22 as "BUST" sentinel
        if total > 17 or (total == 17 and (self.rules.s17 or not soft)):
            return {total: 1.0}
        # H17 on soft-17: must hit when total == 17 and soft == True (if s17==False)

        key = (total, soft, self.deck_key)
        if key in memo:
            return memo[key]

        outcomes: Dict[int, float] = defaultdict(float)
        total_cards = self.N
        if total_cards == 0:
            outcomes[total] += 1.0
            memo[key] = outcomes
            return outcomes

        # Try drawing each possible rank
        counter = self._deck_counter()
        for rank, cnt in counter.items():
            if cnt <= 0:
                continue
            p = cnt / total_cards
            idx = self._rank_index(rank)

            self._draw(idx)
            next_hand = HandState.from_cards(list(hand.cards) + [rank])
            sub = self._dealer_recursive(next_hand, memo)
            self._undo()

            for t, sp in sub.items():
                outcomes[t] += p * sp

        memo[key] = outcomes
        return outcomes

    # ------------ Deck bookkeeping / keys ------------

    def _load_deck(self, counts: Counter[str]) -> None:
        # Optionally (re)compute strides using provided shoe_max_counts, else infer
        if self.shoe_max_counts:
            max_counts = [self.shoe_max_counts.get(r, 0) for r in RANKS]
        else:
            # Infer a safe max as the current counts (works for single-situation cache;
            # for multi-hand sessions, pass true shoe caps to keep keys stable)
            max_counts = [counts.get(r, 0) for r in RANKS]

        # Build strides
        strides = [1] * len(RANKS)
        prod = 1
        for i, r in enumerate(RANKS):
            strides[i] = prod
            prod *= max_counts[i] + 1
        self.strides = strides

        # Load deck and compute deck_key
        self.deck = [counts.get(r, 0) for r in RANKS]
        self.N = sum(self.deck)
        self.deck_key = 0
        for i, n in enumerate(self.deck):
            self.deck_key += n * self.strides[i]

        # Reset move stack (fresh situation)
        self._stack.clear()

    def _draw(self, idx: int) -> None:
        prev_n = self.deck[idx]
        assert prev_n > 0, "Attempted to draw from empty rank count"
        self.deck[idx] = prev_n - 1
        prev_N = self.N
        self.N = prev_N - 1
        # update deck_key incrementally
        self.deck_key -= self.strides[idx]
        self._stack.append((idx, prev_n, prev_N))

    def _undo(self) -> None:
        idx, prev_n, prev_N = self._stack.pop()
        # restore
        self.deck[idx] = prev_n
        self.N = prev_N
        self.deck_key += self.strides[idx]

    def _deck_counter(self) -> Counter[str]:
        return Counter(
            {RANKS[i]: self.deck[i] for i in range(len(RANKS)) if self.deck[i] > 0}
        )

    def _rank_index(self, rank: str) -> int:
        try:
            return RANKS.index(rank)
        except ValueError:
            if rank in ("J", "Q", "K"):  # if your other code ever passes faces
                return RANKS.index(TEN_BUCKET)
            raise

    def _rules_mask(self) -> int:
        # Pack a mask so rule settings can be memoized too
        # bit0: s17, bit1: das, bit2: can_hit_after_split_aces, bit3: surrender_any_time, bit4: bj_3to2, bit5: insurance, bits 6-12: max_splits (up to 127)
        r = self.rules
        return (
            (1 if r.s17 else 0)
            | ((1 if r.das else 0) << 1)
            | ((1 if r.must_stand_after_split_aces else 0) << 2)
            | ((1 if r.cashout_allowed else 0) << 3)
            | ((1 if r.blackjack_payout == 1.5 else 0) << 4)
            | ((1 if r.insurance_allowed else 0) << 5)
            | ((int(r.max_splits) & 0x7F) << 6)
        )

    def _compute_moments(
        self,
        hand: HandState,
        pk: PlayerKey,
        dealer_up: str,
        assume_split_independence: bool,
    ) -> Tuple[float, float, ActionType]:
        # Terminal handling
        if hand.is_busted:
            return -1.0, 1.0, ActionType.STAND
        if hand.is_terminal and not hand.is_blackjack:
            m1, m2 = self._stand_moments(hand, dealer_up)
            return m1, m2, ActionType.STAND
        if hand.is_blackjack:
            m1, m2 = self._stand_moments(hand, dealer_up)
            return m1, m2, ActionType.STAND

        # Candidate actions -> (m1, m2)
        cand: Dict[ActionType, Tuple[float, float]] = {}

        # Stand
        cand[ActionType.STAND] = self._stand_moments(hand, dealer_up)

        # Surrender any time
        if self.rules.cashout_allowed:
            # Constant payoff SURRENDER_EV
            m1 = SURRENDER_EV
            m2 = SURRENDER_EV * SURRENDER_EV
            cand[ActionType.SURRENDER_ANY_TIME] = (m1, m2)

        # Insurance
        if self.rules.insurance_allowed and dealer_up == "A" and pk.num_cards == 2:
            m1, m2 = self._insurance_moments(hand, dealer_up)
            cand[ActionType.INSURANCE] = (m1, m2)

        # Double
        if pk.can_double and pk.num_cards == 2:
            cand[ActionType.DOUBLE] = self._double_moments(hand, dealer_up, pk)

        # Hit
        cand[ActionType.HIT] = self._hit_moments(hand, dealer_up, pk)

        # Split
        if pk.can_split and pk.is_pair and pk.num_cards == 2:
            cand[ActionType.SPLIT] = self._split_moments(
                hand, dealer_up, pk, assume_split_independence
            )

        # Choose best by maximizing expected value (first moment)
        best_action = max(cand.keys(), key=lambda a: cand[a][0])
        m1, m2 = cand[best_action]
        return m1, m2, best_action

    def _stand_moments(self, hand: HandState, dealer_up: str) -> Tuple[float, float]:
        # Return (E[X], E[X^2]) when standing
        if hand.is_busted:
            return -1.0, 1.0
        # Blackjack payout
        if hand.is_blackjack and self.rules.blackjack_payout == 1.5:
            if dealer_up in ("A", TEN_BUCKET):
                dist = self._dealer_dist(dealer_up)
                m1 = 0.0
                m2 = 0.0
                for total, p in dist.items():
                    if total == 21:
                        payoff = 0.0
                    elif total > 21:
                        payoff = BLACKJACK_PAYOUT
                    else:
                        payoff = BLACKJACK_PAYOUT
                    m1 += p * payoff
                    m2 += p * (payoff * payoff)
                return m1, m2
            else:
                payoff = BLACKJACK_PAYOUT
                return payoff, payoff * payoff

        player = hand.total
        dist = self._dealer_dist(dealer_up)
        m1 = 0.0
        m2 = 0.0
        for total, p in dist.items():
            if total > 21:
                payoff = 1.0
            elif player > total:
                payoff = 1.0
            elif player == total:
                payoff = 0.0
            else:
                payoff = -1.0
            m1 += p * payoff
            m2 += p * (payoff * payoff)
        return m1, m2

    def _hit_moments(
        self, hand: HandState, dealer_up: str, pk: PlayerKey
    ) -> Tuple[float, float]:
        total_cards = self.N
        if total_cards == 0:
            return self._stand_moments(hand, dealer_up)
        trans = self.calc.get_hit_transitions(hand, self._deck_counter())
        m1 = 0.0
        m2 = 0.0
        for rank, next_hand in trans.items():
            idx = self._rank_index(rank)
            cnt = self.deck[idx]
            if cnt <= 0:
                continue
            p = cnt / total_cards
            self._draw(idx)
            child_pk = PlayerKey.from_hand(
                next_hand, pk.split_level, False, False, False
            )
            key = (child_pk, dealer_up, self._rules_mask(), self.deck_key)
            if key in self.ev_cache:
                cm1, cm2, _ = self.ev_cache[key]
            else:
                cm1, cm2, _ = self._compute_moments(
                    next_hand, child_pk, dealer_up, assume_split_independence=True
                )
                self.ev_cache[key] = (cm1, cm2, ActionType.STAND)
            self._undo()
            m1 += p * cm1
            m2 += p * cm2
        return m1, m2

    def _double_moments(
        self, hand: HandState, dealer_up: str, pk: PlayerKey
    ) -> Tuple[float, float]:
        total_cards = self.N
        if total_cards == 0:
            return self._stand_moments(hand, dealer_up)
        trans = self.calc.get_hit_transitions(hand, self._deck_counter())
        m1 = 0.0
        m2 = 0.0
        for rank, next_hand in trans.items():
            idx = self._rank_index(rank)
            cnt = self.deck[idx]
            if cnt <= 0:
                continue
            p = cnt / total_cards
            self._draw(idx)
            sm1, sm2 = self._stand_moments(next_hand, dealer_up)
            self._undo()
            # Doubling scales outcomes by 2 → moments scale by 2 and 4 respectively
            m1 += p * (2.0 * sm1)
            m2 += p * (4.0 * sm2)
        return m1, m2

    def _split_moments(
        self,
        hand: HandState,
        dealer_up: str,
        pk: PlayerKey,
        assume_split_independence: bool,
    ) -> Tuple[float, float]:
        split_trans = self.calc.get_split_transitions(hand, self._deck_counter())
        total_cards = self.N
        if total_cards < 2:
            return 0.0, 0.0
        m1_total = 0.0
        m2_total = 0.0
        for (c1, c2), (h1, h2) in split_trans.items():
            i1, i2 = self._rank_index(c1), self._rank_index(c2)
            n1 = self.deck[i1]
            n2 = self.deck[i2]
            if n1 <= 0:
                continue
            if c1 == c2 and n1 < 2:
                continue
            if c1 != c2 and n2 <= 0:
                continue

            if c1 == c2:
                p = (n1 / total_cards) * ((n1 - 1) / (total_cards - 1))
            else:
                p = (n1 / total_cards) * (n2 / (total_cards - 1))

            self._draw(i1)
            self._draw(i2)
            # Hand 1
            pk1 = PlayerKey.from_hand(
                h1,
                pk.split_level + 1,
                self.rules.das,
                False,
                (
                    False
                    if self.rules.max_splits >= (pk.split_level + 1)
                    else h1.can_split
                ),
            )
            m1_1, m2_1, _ = self._compute_moments(
                h1, pk1, dealer_up, assume_split_independence
            )
            # Hand 2 (using same deck; independence is handled below at the moment level)
            pk2 = PlayerKey.from_hand(
                h2,
                pk.split_level + 1,
                self.rules.das,
                False,
                (
                    False
                    if self.rules.max_splits >= (pk.split_level + 1)
                    else h2.can_split
                ),
            )
            m1_2, m2_2, _ = self._compute_moments(
                h2, pk2, dealer_up, assume_split_independence
            )
            self._undo()
            self._undo()

            # First moment always adds
            m1_pair = m1_1 + m1_2

            if assume_split_independence:
                # E[(X1+X2)^2] = E[X1^2] + E[X2^2] + 2 E[X1]E[X2]
                m2_pair = m2_1 + m2_2 + 2.0 * (m1_1 * m1_2)
            else:
                # Fallback approximation (no cross term knowledge): treat as independent anyway
                m2_pair = m2_1 + m2_2 + 2.0 * (m1_1 * m1_2)

            m1_total += p * m1_pair
            m2_total += p * m2_pair

        return m1_total, m2_total

    def _insurance_moments(
        self, hand: HandState, dealer_up: str
    ) -> Tuple[float, float]:
        if dealer_up != "A":
            return self._stand_moments(hand, dealer_up)
        ten_idx = self._rank_index(TEN_BUCKET)
        if self.N == 0:
            return -0.5, 0.25
        p_bj = self.deck[ten_idx] / self.N
        p_no = 1.0 - p_bj
        v_bj = 1.0 if hand.is_blackjack else 0.0
        v_no = -0.5 + self._stand_moments(hand, dealer_up)[0]
        m1 = p_bj * v_bj + p_no * v_no
        m2 = p_bj * (v_bj * v_bj) + p_no * (v_no * v_no)
        return m1, m2
