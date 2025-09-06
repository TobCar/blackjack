"""
Hand classification system for blackjack precomputation.

Provides canonical hand descriptors like H12, H16, S18, P_8, etc.
and utilities to generate all relevant hand classes for precomputation.
"""

from typing import List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from game_mechanics.hand_state import HandState
from game_mechanics.rules import RANKS


class HandType(Enum):
    """Types of hands for classification."""
    HARD = "H"      # Hard total (no usable ace)
    SOFT = "S"      # Soft total (usable ace) 
    PAIR = "P"      # Splittable pair


@dataclass(frozen=True)
class HandClass:
    """Canonical hand classification."""
    type: HandType
    value: int  # Total for H/S, rank value for P
    rank: Optional[str] = None  # Only for pairs
    
    def __str__(self) -> str:
        if self.type == HandType.PAIR:
            return f"P_{self.rank}"
        else:
            return f"{self.type.value}{self.value}"
    
    def __repr__(self) -> str:
        return str(self)


class HandClassifier:
    """Classifies hands into canonical hand classes."""
    
    def __init__(self):
        # Pre-compute all relevant hand classes
        self._all_classes = self._generate_all_classes()
    
    def classify_hand(self, hand: HandState) -> HandClass:
        """Classify a hand into its canonical hand class."""
        if hand.can_split:
            # Pair
            pair_rank = hand.cards[0]
            # Convert face cards to "10" for consistency
            if pair_rank in ("J", "Q", "K"):
                pair_rank = "10"
            return HandClass(HandType.PAIR, self._rank_to_value(pair_rank), pair_rank)
        
        elif hand.is_soft:
            # Soft total
            return HandClass(HandType.SOFT, hand.total)
        
        else:
            # Hard total
            return HandClass(HandType.HARD, hand.total)
    
    def _rank_to_value(self, rank: str) -> int:
        """Convert rank to numeric value."""
        if rank == "A":
            return 1
        elif rank in ("J", "Q", "K", "10"):
            return 10
        else:
            return int(rank)
    
    def _generate_all_classes(self) -> Set[HandClass]:
        """Generate all relevant hand classes for precomputation."""
        classes = set()
        
        # Hard totals: 5-21 (can't have hard 2-4 with 2 cards)
        for total in range(5, 22):
            classes.add(HandClass(HandType.HARD, total))
        
        # Soft totals: 13-21 (A,2 = soft 13, A,10 = soft 21)
        # Note: soft 22+ is impossible (would convert to hard)
        for total in range(13, 22):
            classes.add(HandClass(HandType.SOFT, total))
        
        # Pairs: all ranks
        for rank in RANKS:
            # Normalize face cards to "10"
            normalized_rank = "10" if rank in ("J", "Q", "K") else rank
            if normalized_rank not in [p.rank for p in classes if p.type == HandType.PAIR]:
                classes.add(HandClass(HandType.PAIR, self._rank_to_value(normalized_rank), normalized_rank))
        
        return classes
    
    def get_all_classes(self) -> List[HandClass]:
        """Get all hand classes, sorted for deterministic ordering."""
        return sorted(self._all_classes, key=lambda hc: (hc.type.value, hc.value, hc.rank or ""))
    
    def get_classes_by_type(self, hand_type: HandType) -> List[HandClass]:
        """Get all hand classes of a specific type."""
        return sorted(
            [hc for hc in self._all_classes if hc.type == hand_type],
            key=lambda hc: (hc.value, hc.rank or "")
        )
    
    def parse_hand_class(self, class_str: str) -> HandClass:
        """Parse hand class from string like 'H16', 'S18', 'P_8'."""
        if class_str.startswith("P_"):
            # Pair
            rank = class_str[2:]
            return HandClass(HandType.PAIR, self._rank_to_value(rank), rank)
        
        elif class_str.startswith("H"):
            # Hard total
            total = int(class_str[1:])
            return HandClass(HandType.HARD, total)
        
        elif class_str.startswith("S"):
            # Soft total  
            total = int(class_str[1:])
            return HandClass(HandType.SOFT, total)
        
        else:
            raise ValueError(f"Invalid hand class string: {class_str}")
    
    def create_example_hand(self, hand_class: HandClass) -> HandState:
        """Create an example HandState for a given hand class."""
        if hand_class.type == HandType.PAIR:
            # Create pair of the specified rank
            return HandState.from_cards([hand_class.rank, hand_class.rank])
        
        elif hand_class.type == HandType.SOFT:
            # Soft total: use A + other card to reach total
            needed = hand_class.value - 11  # Ace counts as 11 in soft
            if needed < 2 or needed > 10:
                raise ValueError(f"Invalid soft total: {hand_class.value}")
            
            other_card = "10" if needed == 10 else str(needed)
            return HandState.from_cards(["A", other_card])
        
        else:  # HARD
            # Hard total: avoid aces, create minimal example
            total = hand_class.value
            
            if total < 4:
                raise ValueError(f"Invalid hard total: {total}")
            
            if total <= 10:
                # Use 2 cards that sum to total
                first = min(total // 2, 9)
                second = total - first
                return HandState.from_cards([str(first), str(second)])
            
            elif total <= 20:
                # Use 10 + remainder
                remainder = total - 10
                return HandState.from_cards(["10", str(remainder)])
            
            else:  # total == 21
                # Hard 21: use 10 + J (which becomes 10+10=20 in hand state)
                # Actually need to create a proper hard 21
                return HandState.from_cards(["7", "6", "8"])  # Hard 21
    
    def is_valid_class(self, hand_class: HandClass) -> bool:
        """Check if a hand class is valid/possible."""
        return hand_class in self._all_classes


def filter_hand_classes(
    all_classes: List[HandClass], 
    only_hands: Optional[List[str]] = None
) -> List[HandClass]:
    """Filter hand classes based on command line specification."""
    if only_hands is None:
        return all_classes
    
    classifier = HandClassifier()
    filtered = []
    
    for hand_str in only_hands:
        try:
            hand_class = classifier.parse_hand_class(hand_str)
            if hand_class in all_classes:
                filtered.append(hand_class)
            else:
                print(f"Warning: Hand class {hand_str} not found in generated classes")
        except ValueError as e:
            print(f"Warning: Invalid hand class {hand_str}: {e}")
    
    return filtered