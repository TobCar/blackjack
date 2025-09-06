"""
Legality mask generation for blackjack actions based on rules and game state.

Generates bitmasks where:
- bit 0 (1): STAND
- bit 1 (2): HIT  
- bit 2 (4): DOUBLE
- bit 3 (8): SPLIT
"""

from typing import Dict, Set
from game_mechanics.action_type import ActionType
from game_mechanics.hand_state import HandState
from game_mechanics.rules import BlackjackRuleset
from precomputing.hand_classification import HandClass, HandType

# Bitmask constants for actions
ACTION_BITS = {
    ActionType.STAND: 1,
    ActionType.HIT: 2,
    ActionType.DOUBLE: 4,
    ActionType.SPLIT: 8,
}


class LegalityMaskGenerator:
    """Generates legality masks for hand classes under specific rules."""
    
    def __init__(self, rules: BlackjackRuleset):
        self.rules = rules
    
    def generate_mask(
        self, 
        hand_class: HandClass, 
        dealer_upcard: str,
        *,
        is_first_decision: bool = True,
        split_level: int = 0,
        post_split_aces: bool = False
    ) -> int:
        """
        Generate legality bitmask for a hand class.
        
        Args:
            hand_class: Canonical hand classification
            dealer_upcard: Dealer's upcard
            is_first_decision: Whether this is the first decision for the hand
            split_level: How many splits deep (0 = original hand)  
            post_split_aces: Whether this hand came from splitting aces
            
        Returns:
            Bitmask where each bit represents an action's legality
        """
        mask = 0
        
        # Create example hand for rule checking
        hand = self._create_example_hand(hand_class)
        
        # STAND - always legal unless busted (but busted hands shouldn't reach here)
        if not hand.is_busted:
            mask |= ACTION_BITS[ActionType.STAND]
        
        # HIT - legal unless terminal or post-split aces with must_stand rule
        if not hand.is_terminal:
            if post_split_aces and self.rules.must_stand_after_split_aces:
                # Can't hit after splitting aces
                pass
            else:
                mask |= ACTION_BITS[ActionType.HIT]
        
        # DOUBLE - legal if first decision and not terminal
        if (is_first_decision and 
            not hand.is_terminal and 
            not (post_split_aces and self.rules.must_stand_after_split_aces)):
            
            # Additional check: some rules only allow double after split if DAS is enabled
            if split_level > 0 and not self.rules.das:
                # Can't double after split
                pass
            else:
                mask |= ACTION_BITS[ActionType.DOUBLE]
        
        # SPLIT - legal only for pairs, first decision, within split limits
        if (hand_class.type == HandType.PAIR and 
            is_first_decision and
            split_level < self.rules.max_splits):
            
            mask |= ACTION_BITS[ActionType.SPLIT]
        
        return mask
    
    def _create_example_hand(self, hand_class: HandClass) -> HandState:
        """Create example hand for rule checking."""
        if hand_class.type == HandType.PAIR:
            # Create pair
            return HandState.from_cards([hand_class.rank, hand_class.rank])
        
        elif hand_class.type == HandType.SOFT:
            # Soft total: A + other card
            needed = hand_class.value - 11
            if needed <= 0 or needed > 10:
                raise ValueError(f"Invalid soft total: {hand_class.value}")
            
            other_card = "10" if needed == 10 else str(needed)  
            return HandState.from_cards(["A", other_card])
        
        else:  # HARD
            # Hard total: avoid creating impossible combinations
            total = hand_class.value
            
            if total < 4:
                raise ValueError(f"Invalid hard total: {total}")
            
            # Simple strategy: use smallest cards possible
            if total <= 10:
                # Use 2 cards that sum to total
                first = min(total // 2, 10)
                second = total - first
                return HandState.from_cards([str(first), str(second)])
            
            else:
                # Use 10 + remainder (or face card)
                remainder = total - 10
                if remainder <= 10:
                    return HandState.from_cards(["10", str(remainder)])
                else:
                    # Need 3+ cards, use 10+6+remainder
                    return HandState.from_cards(["10", "6", str(remainder-6)])
    
    def generate_all_masks(
        self, 
        hand_classes: list[HandClass], 
        dealer_upcards: list[str]
    ) -> Dict[tuple[HandClass, str], int]:
        """Generate masks for all hand class + dealer upcard combinations."""
        masks = {}
        
        for hand_class in hand_classes:
            for upcard in dealer_upcards:
                # Generate mask for standard first decision
                mask = self.generate_mask(hand_class, upcard)
                masks[(hand_class, upcard)] = mask
                
                # For pairs, also generate post-split scenarios if relevant
                if hand_class.type == HandType.PAIR:
                    # Post-split, non-aces
                    if hand_class.rank != "A":
                        post_split_mask = self.generate_mask(
                            hand_class, upcard,
                            is_first_decision=True,
                            split_level=1,
                            post_split_aces=False
                        )
                        masks[(hand_class, upcard, "post_split")] = post_split_mask
                    
                    # Post-split aces (if applicable)
                    else:
                        post_split_aces_mask = self.generate_mask(
                            hand_class, upcard,
                            is_first_decision=True, 
                            split_level=1,
                            post_split_aces=True
                        )
                        masks[(hand_class, upcard, "post_split_aces")] = post_split_aces_mask
        
        return masks
    
    def mask_to_actions(self, mask: int) -> Set[ActionType]:
        """Convert bitmask to set of legal actions."""
        actions = set()
        for action, bit in ACTION_BITS.items():
            if mask & bit:
                actions.add(action)
        return actions
    
    def actions_to_mask(self, actions: Set[ActionType]) -> int:
        """Convert set of actions to bitmask."""
        mask = 0
        for action in actions:
            if action in ACTION_BITS:
                mask |= ACTION_BITS[action]
        return mask


def validate_legality_masks(
    masks: Dict[tuple, int], 
    rules: BlackjackRuleset
) -> bool:
    """Validate that generated masks respect rule constraints."""
    violations = []
    
    for key, mask in masks.items():
        hand_class = key[0]
        upcard = key[1]
        context = key[2] if len(key) > 2 else "standard"
        
        # Check DAS constraints
        if context == "post_split" and (mask & ACTION_BITS[ActionType.DOUBLE]):
            if not rules.das:
                violations.append(f"Double after split allowed but DAS=False: {key}")
        
        # Check split ace constraints  
        if context == "post_split_aces":
            if rules.must_stand_after_split_aces:
                if mask & ACTION_BITS[ActionType.HIT]:
                    violations.append(f"Hit after split aces allowed but must_stand_after_split_aces=True: {key}")
                if mask & ACTION_BITS[ActionType.DOUBLE]:
                    violations.append(f"Double after split aces allowed but must_stand_after_split_aces=True: {key}")
        
        # Check that STAND is always available for non-busted hands
        if hand_class.type != HandType.HARD or hand_class.value <= 21:
            if not (mask & ACTION_BITS[ActionType.STAND]):
                violations.append(f"STAND not available for non-busted hand: {key}")
    
    if violations:
        print("Legality mask violations:")
        for v in violations:
            print(f"  - {v}")
        return False
    
    return True