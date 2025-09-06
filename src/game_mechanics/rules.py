from typing import Tuple, Optional, Protocol
from dataclasses import dataclass


RANKS: Tuple[str, ...] = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10")
TEN_BUCKET = "10"
CASHOUT_EV = -0.6
BLACKJACK_PAYOUT = 1.5


class PeekPolicy(Protocol):
    def __call__(self, upcard: str) -> bool: ...


class PayoutPolicy(Protocol):
    def __call__(
        self,
        player_total: int,
        dealer_total: int,
        bet_multiplier: float = 1.0,
    ) -> float: ...


@dataclass(frozen=True)
class BlackjackRuleset:
    # Identification
    name: str = "playnow-online-casino"

    # Core structural rules
    s17: bool = True
    das: bool = True
    must_stand_after_split_aces: bool = True
    blackjack_payout: float = 1.5
    max_splits: int = 1  # 1 => up to 2 hands total
    can_cashout: bool = False  # Only offered in online casinos

    # Functional policies (callables)
    should_dealer_peek: PeekPolicy = None  # type: ignore[assignment]
    calculate_non_player_blackjack_payout: PayoutPolicy = None  # type: ignore[assignment]


# Default implementations for online-casino variant
def dealer_only_peeks_aces(upcard: str) -> bool:
    """Dealer peeks for blackjack only when showing Ace."""
    return upcard == "A"


def standard_payout_without_player_blackjack(
    player_total: int,
    dealer_total: int,
    bet_multiplier: float = 1.0,
) -> float:
    """
    Calculate payout for a blackjack hand.

    Key rule: Dealer blackjack always pushes against player blackjack
    (regardless of whether dealer peeked or drew to blackjack)

    Args:
        player_total: Player's hand total
        player_is_blackjack: True if player has natural blackjack
        dealer_total: Dealer's hand total (or 22 for bust)
        dealer_is_blackjack: True if dealer has natural blackjack
        bet_multiplier: Multiplier for bet size (1.0=normal, 2.0=doubled)

    Returns:
        Payout as multiple of original bet (positive=win, negative=loss, 0=push)
    """
    # Player busted
    if player_total > 21:
        return -bet_multiplier

    # Dealer busted, player did not
    if dealer_total > 21:
        return bet_multiplier

    # Neither busted: compare totals
    if player_total > dealer_total:
        return bet_multiplier
    elif player_total == dealer_total:
        return 0.0
    else:
        return -bet_multiplier


RULES = BlackjackRuleset(
    s17=True,
    das=True,
    must_stand_after_split_aces=True,  # no further hits after split aces
    blackjack_payout=BLACKJACK_PAYOUT,
    can_cashout=True,
    max_splits=1,
    should_dealer_peek=dealer_only_peeks_aces,
    calculate_non_player_blackjack_payout=standard_payout_without_player_blackjack,
)
