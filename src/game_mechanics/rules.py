from typing import Tuple, Optional, Protocol
from dataclasses import dataclass


RANKS: Tuple[str, ...] = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10")
TEN_BUCKET = "10"
CASHOUT_EV = -0.6
BLACKJACK_PAYOUT = 1.5


@dataclass(frozen=True)
class DealerBJOutcome:
    end_hand: bool
    player_bj_pushes: bool
    lose_additional_bets: bool  # doubles/splits also lose when True


class PeekPolicy(Protocol):
    def __call__(
        self, upcard: str, insurance_offered: bool, insurance_taken: bool
    ) -> bool: ...


class SettlementPolicy(Protocol):
    def __call__(
        self, player_has_blackjack: bool, has_additional_bets: bool
    ) -> DealerBJOutcome: ...


class SurrenderPolicy(Protocol):
    def __call__(self, n_cards: int) -> Optional[float]: ...


class InsuranceOfferPolicy(Protocol):
    def __call__(self, upcard: str) -> bool: ...


@dataclass(frozen=True)
class BlackjackRuleset:
    # Identification
    name: str = "playnow-online-casino"
    
    # Core structural rules
    s17: bool = True
    das: bool = True
    must_stand_after_split_aces: bool = True
    blackjack_payout: float = 1.5
    insurance_allowed: bool = True
    max_splits: int = 1  # 1 => up to 2 hands total
    cashout_allowed: bool = False  # Only offered in online casinos

    # Functional policies (callables)
    offer_insurance: InsuranceOfferPolicy = None  # type: ignore[assignment]
    should_dealer_peek: PeekPolicy = None  # type: ignore[assignment]
    settle_peeked_dealer_blackjack: SettlementPolicy = None  # type: ignore[assignment]
    settle_drawn_dealer_blackjack: SettlementPolicy = None  # type: ignore[assignment]


# Default implementations for online-casino variant
def standard_insurance_offer_on_ace(upcard: str) -> bool:
    return upcard == "A"


def hybrid_peek_only_when_insured(
    upcard: str, insurance_offered: bool, insurance_taken: bool
) -> bool:
    return upcard == "A" and insurance_offered and insurance_taken


def settlement_peeked_dealer_bj(
    player_has_blackjack: bool, has_additional_bets: bool
) -> DealerBJOutcome:
    return DealerBJOutcome(
        end_hand=True, player_bj_pushes=True, lose_additional_bets=False
    )


def settlement_drawn_dealer_bj(
    player_has_blackjack: bool, has_additional_bets: bool
) -> DealerBJOutcome:
    return DealerBJOutcome(
        end_hand=True, player_bj_pushes=False, lose_additional_bets=True
    )


RULES = BlackjackRuleset(
    s17=True,
    das=True,
    must_stand_after_split_aces=True,  # no further hits after split aces
    blackjack_payout=BLACKJACK_PAYOUT,
    insurance_allowed=True,
    cashout_allowed=True,
    max_splits=1,
    should_dealer_peek=hybrid_peek_only_when_insured,
    settle_peeked_dealer_blackjack=settlement_peeked_dealer_bj,
    settle_drawn_dealer_blackjack=settlement_drawn_dealer_bj,
    offer_insurance=standard_insurance_offer_on_ace,
)
