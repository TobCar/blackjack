from __future__ import annotations
import sys
from pathlib import Path

# Add parent directory to path so we can import from sibling modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple
from collections import Counter

from game_mechanics import HandState
from ev_engine import BlackjackEVEngine
from game_mechanics.rules import BlackjackRuleset

# ---------- Helpers ----------

RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10")


def shoe_counts(num_decks: int = 1) -> Counter:
    """
    Build a fresh shoe with num_decks (exact 52*num_decks), bucketing 10/J/Q/K as '10'.
    """
    c = Counter()
    for r in RANKS:
        if r == "10":
            # 10, J, Q, K -> 16 per deck
            c[r] = 16 * num_decks
        elif r == "A":
            c[r] = 4 * num_decks
        else:
            c[r] = 4 * num_decks
    return c


def remove_visible(counts: Counter, cards: Iterable[str]) -> Counter:
    """
    Remove visible cards from the shoe counts. Accepts "J/Q/K" and maps to '10'.
    """
    out = counts.copy()
    for card in cards:
        rank = card if card in RANKS else "10"
        if out[rank] <= 0:
            raise ValueError(
                f"Attempting to remove unavailable card: {card} (as {rank})"
            )
        out[rank] -= 1
    return out


# ---------- Scenario definition ----------


@dataclass
class Scenario:
    player_cards: List[str]  # e.g., ["8","8"] or ["A","7"]
    dealer_up: str  # e.g., "9" or "A"
    num_decks: int = 1
    # Action eligibility flags for the current node
    can_double: bool = True
    cashout_allowed: bool = True
    can_split: bool = True
    # Optional rules
    rules: BlackjackRuleset | None = None
    # If you want to override the remaining deck entirely, provide this:
    remaining_counts_override: Counter | None = None


# ---------- Runner ----------


def run_scenarios(scenarios: List[Scenario]) -> List[Dict[str, Any]]:
    """
    Evaluate EV for each scenario with exact without-replacement math.
    Returns a list of dicts with inputs + (ev, best_action).
    """
    results: List[Dict[str, Any]] = []

    # Keep an engine per rule-set to retain hot caches across scenarios with the same rules
    engine_cache: Dict[Tuple, BlackjackEVEngine] = {}

    for sc in scenarios:
        rules = sc.rules or RULES
        rules_key = (
            rules.s17,
            rules.das,
            rules.max_splits,
            rules.must_stand_after_split_aces,
            rules.cashout_allowed,
            rules.blackjack_payout,
            rules.insurance_allowed,
        )
        engine = engine_cache.get(rules_key)
        if engine is None:
            # Provide full-shoe maxima so deck_key stays stable across scenarios in this session
            max_counts = shoe_counts(sc.num_decks)
            engine = BlackjackEVEngine(rules=rules, shoe_max_counts=dict(max_counts))
            engine_cache[rules_key] = engine

        # Build remaining deck: start from a fresh shoe unless overridden
        if sc.remaining_counts_override is not None:
            remaining = sc.remaining_counts_override
        else:
            base = shoe_counts(sc.num_decks)
            # Remove visible cards: player's two (or more) cards and dealer upcard
            visible = list(sc.player_cards) + [sc.dealer_up]
            remaining = remove_visible(base, visible)

        # Player hand
        hand = HandState.from_cards(sc.player_cards)

        # Evaluate
        ev, action = engine.evaluate(
            hand=hand,
            dealer_up=sc.dealer_up if sc.dealer_up in RANKS else "10",
            remaining_counts=remaining,
            can_double=sc.can_double,
            cashout_allowed=sc.cashout_allowed,
            can_split=sc.can_split,
        )

        results.append(
            {
                "player": sc.player_cards,
                "dealer_up": sc.dealer_up,
                "num_decks": sc.num_decks,
                "can_double": sc.can_double,
                "cashout_allowed": sc.cashout_allowed,
                "can_split": sc.can_split,
                "rules": rules,
                "ev": ev,
                "best_action": action.value,
            }
        )

    return results


if __name__ == "__main__":
    # A small battery of scenarios you can change freely.
    scenarios = [
        # Classic pair of 8s vs 9 — often split in S17 DAS
        Scenario(
            player_cards=["8", "8"],
            dealer_up="9",
            num_decks=6,
            can_double=True,
            cashout_allowed=True,
            can_split=True,
            rules=BlackjackRuleset(s17=True, das=True, cashout_allowed=True),
        ),
        # Soft 18 vs 9
        Scenario(
            player_cards=["A", "7"],
            dealer_up="9",
            num_decks=6,
            rules=BlackjackRuleset(s17=True, das=True),
        ),
        # Hard 16 vs 10, surrender allowed
        Scenario(
            player_cards=["10", "6"],
            dealer_up="10",
            num_decks=6,
            cashout_allowed=True,
            rules=BlackjackRuleset(s17=True, das=True, cashout_allowed=True),
        ),
        # Blackjack vs dealer Ace (insurance allowed) — tests push on dealer BJ
        Scenario(
            player_cards=["A", "10"],
            dealer_up="A",
            num_decks=6,
            rules=BlackjackRuleset(s17=True, insurance_allowed=True),
        ),
        # Double test: hard 11 vs 6
        Scenario(
            player_cards=["6", "5"],
            dealer_up="6",
            num_decks=6,
            rules=BlackjackRuleset(s17=True, das=True),
        ),
    ]

    results = run_scenarios(scenarios)
    # Pretty print
    print("\nEV results (exact, without replacement):")
    for r in results:
        print(
            f"Player {r['player']} vs Dealer {r['dealer_up']} "
            f"(decks={r['num_decks']}): EV={r['ev']:.4f}, action={r['best_action']}"
        )
