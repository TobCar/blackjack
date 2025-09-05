from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class HandState:
    """Immutable representation of a blackjack hand state"""

    cards: Tuple[str, ...]
    total: int
    is_soft: bool
    card_count: int
    forced_terminal: bool = False

    @property
    def is_busted(self) -> bool:
        return self.total > 21

    @property
    def is_blackjack(self) -> bool:
        return self.total == 21 and self.card_count == 2

    @property
    def is_terminal(self) -> bool:
        """Terminal states: 21+ (always stand on 21, can't act on 22+), or forced by rules."""
        return self.forced_terminal or self.total >= 21

    @property
    def can_split(self) -> bool:
        """Check if this hand can be split (pair of same rank)"""
        return self.card_count == 2 and self.cards[0] == self.cards[1]

    @property
    def split_card(self) -> str:
        """Get the card value for splitting (only valid if can_split is True)"""
        if not self.can_split:
            raise ValueError("Cannot get split card from non-splittable hand")
        return self.cards[0]

    @classmethod
    def from_cards(cls, cards: List[str], force_terminal: bool = False) -> "HandState":
        """Create HandState from list of card strings"""
        total = 0
        aces = 0

        for card in cards:
            if card == "A":
                aces += 1
                total += 11
            elif card in ("J", "Q", "K"):
                total += 10
            else:
                total += int(card)

        # Adjust for aces (make them low if needed)
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1

        is_soft = aces > 0 and total <= 21
        return cls(
            cards=tuple(cards),
            total=total,
            is_soft=is_soft,
            card_count=len(cards),
            forced_terminal=force_terminal,
        )

    def __str__(self) -> str:
        soft_indicator = "Soft " if self.is_soft else "Hard "
        if self.is_busted:
            return f"Bust {self.total}"
        return f"{soft_indicator}{self.total}"

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)
