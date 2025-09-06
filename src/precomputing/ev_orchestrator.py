"""
EV computation orchestration for precomputation.

This module orchestrates calls to ev_engine.py to compute per-action EVs
for all (hand_class, dealer_upcard, bucket) combinations.
"""

from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
import numpy as np
from dataclasses import dataclass

from game_mechanics.action_type import ActionType
from game_mechanics.hand_state import HandState
from game_mechanics.rules import BlackjackRuleset, RULES, RANKS
from precomputing.ev_engine import BlackjackEVEngine
from precomputing.hand_classification import HandClass, HandClassifier, HandType
from precomputing.composition_bucketing import CompositionBucketing
from precomputing.legality import LegalityMaskGenerator, ACTION_BITS


@dataclass
class EVResult:
    """Result of EV computation for a single situation."""
    hand_class: HandClass
    dealer_upcard: str
    bucket_id: int
    legal_mask: int
    ev_stand: Optional[float] = None
    ev_hit: Optional[float] = None
    ev_double: Optional[float] = None
    ev_split: Optional[float] = None
    
    def get_action_ev(self, action: ActionType) -> Optional[float]:
        """Get EV for specific action."""
        if action == ActionType.STAND:
            return self.ev_stand
        elif action == ActionType.HIT:
            return self.ev_hit
        elif action == ActionType.DOUBLE:
            return self.ev_double
        elif action == ActionType.SPLIT:
            return self.ev_split
        else:
            return None
    
    def set_action_ev(self, action: ActionType, ev: float) -> None:
        """Set EV for specific action."""
        if action == ActionType.STAND:
            self.ev_stand = ev
        elif action == ActionType.HIT:
            self.ev_hit = ev
        elif action == ActionType.DOUBLE:
            self.ev_double = ev
        elif action == ActionType.SPLIT:
            self.ev_split = ev


class EVOrchestrator:
    """Orchestrates EV computations using ev_engine.py."""
    
    def __init__(
        self, 
        rules: BlackjackRuleset = None,
        bucketing: CompositionBucketing = None
    ):
        self.rules = rules or RULES
        self.bucketing = bucketing or CompositionBucketing()
        self.classifier = HandClassifier()
        self.legality_gen = LegalityMaskGenerator(self.rules)
        
        # Initialize EV engine
        self.engine = BlackjackEVEngine(self.rules)
        
        # Cache for dealer PMF computations
        self._dealer_pmf_cache: Dict[Tuple[str, int], Dict[int, float]] = {}
    
    def compute_all_evs(
        self,
        hand_classes: List[HandClass],
        dealer_upcards: List[str],
        bucket_ids: List[int],
        *,
        progress_callback: Optional[callable] = None
    ) -> List[EVResult]:
        """
        Compute EVs for all combinations of hand classes, upcards, and buckets.
        
        Args:
            hand_classes: List of hand classes to compute
            dealer_upcards: List of dealer upcards  
            bucket_ids: List of bucket IDs to use
            progress_callback: Optional callback for progress reporting
            
        Returns:
            List of EVResult objects with computed EVs
        """
        results = []
        total_combinations = len(hand_classes) * len(dealer_upcards) * len(bucket_ids)
        processed = 0
        
        for hand_class in hand_classes:
            for dealer_upcard in dealer_upcards:
                for bucket_id in bucket_ids:
                    result = self.compute_situation_ev(
                        hand_class, dealer_upcard, bucket_id
                    )
                    results.append(result)
                    
                    processed += 1
                    if progress_callback:
                        progress_callback(processed, total_combinations)
        
        return results
    
    def compute_situation_ev(
        self, 
        hand_class: HandClass, 
        dealer_upcard: str, 
        bucket_id: int
    ) -> EVResult:
        """Compute EV for a single situation."""
        # Get bucket composition
        bucket_counts = self.bucketing.bucket_id_to_counter(bucket_id)
        
        # Remove dealer upcard from composition (already dealt)
        remaining_counts = bucket_counts.copy()
        if dealer_upcard in remaining_counts:
            remaining_counts[dealer_upcard] -= 1
            if remaining_counts[dealer_upcard] <= 0:
                del remaining_counts[dealer_upcard]
        
        # Create example hand for this class
        hand = self._create_hand_for_class(hand_class)
        
        # Remove player cards from remaining deck
        for card in hand.cards:
            if card in remaining_counts:
                remaining_counts[card] -= 1
                if remaining_counts[card] <= 0:
                    del remaining_counts[card]
        
        # Generate legality mask
        legal_mask = self.legality_gen.generate_mask(hand_class, dealer_upcard)
        
        # Initialize result
        result = EVResult(
            hand_class=hand_class,
            dealer_upcard=dealer_upcard, 
            bucket_id=bucket_id,
            legal_mask=legal_mask
        )
        
        # Compute EV for each legal action
        legal_actions = self.legality_gen.mask_to_actions(legal_mask)
        
        for action in legal_actions:
            try:
                ev = self._compute_action_ev(
                    hand, action, dealer_upcard, remaining_counts
                )
                result.set_action_ev(action, ev)
            except Exception as e:
                print(f"Warning: Failed to compute {action} EV for {hand_class} vs {dealer_upcard}, bucket {bucket_id}: {e}")
                # Set to very negative value to indicate unavailable
                result.set_action_ev(action, -999.0)
        
        return result
    
    def _compute_action_ev(
        self,
        hand: HandState,
        action: ActionType,
        dealer_upcard: str,
        remaining_counts: Counter[str]
    ) -> float:
        """Compute EV for specific action using ev_engine."""
        if action == ActionType.STAND:
            return self._compute_stand_ev(hand, dealer_upcard, remaining_counts)
        
        elif action == ActionType.HIT:
            # Use engine's hit evaluation
            ev, _ = self.engine.evaluate(
                hand, dealer_upcard, remaining_counts,
                can_double=False,  # After hit, can't double
                can_split=False    # After hit, can't split
            )
            return ev
        
        elif action == ActionType.DOUBLE:
            return self._compute_double_ev(hand, dealer_upcard, remaining_counts)
        
        elif action == ActionType.SPLIT:
            return self._compute_split_ev(hand, dealer_upcard, remaining_counts)
        
        
        else:
            raise ValueError(f"Unsupported action for EV computation: {action}")
    
    
    def _compute_stand_ev(
        self, 
        hand: HandState, 
        dealer_upcard: str,
        remaining_counts: Counter[str]
    ) -> float:
        """Compute stand EV by evaluating against dealer distribution."""
        # Handle player blackjack specially
        if hand.is_blackjack:
            return self._compute_blackjack_stand_ev(hand, dealer_upcard, remaining_counts)
        
        # Get dealer distribution
        dealer_dist = self._get_dealer_distribution(dealer_upcard, remaining_counts)
        
        # Compute expected payout
        ev = 0.0
        for dealer_total, prob in dealer_dist.items():
            payout = self.rules.calculate_non_player_blackjack_payout(
                player_total=hand.total,
                dealer_total=dealer_total,
                bet_multiplier=1.0
            )
            ev += prob * payout
        
        return ev
    
    def _compute_blackjack_stand_ev(
        self,
        hand: HandState,
        dealer_upcard: str, 
        remaining_counts: Counter[str]
    ) -> float:
        """Compute stand EV for player blackjack with peek/settlement rules."""
        # Check if dealer could have blackjack
        if dealer_upcard not in ("A", "10"):
            return self.rules.blackjack_payout
        
        # Calculate dealer blackjack probability
        total_remaining = sum(remaining_counts.values())
        if total_remaining == 0:
            return self.rules.blackjack_payout  # Can't determine, assume no dealer BJ
        
        if dealer_upcard == "A":
            # Dealer needs 10-value
            ten_count = remaining_counts.get("10", 0)
            dealer_bj_prob = ten_count / total_remaining
        else:  # dealer_upcard == "10"
            # Dealer needs Ace
            ace_count = remaining_counts.get("A", 0)
            dealer_bj_prob = ace_count / total_remaining
        
        # EV = P(dealer BJ) * push + P(no dealer BJ) * blackjack_payout
        return dealer_bj_prob * 0.0 + (1 - dealer_bj_prob) * self.rules.blackjack_payout
    
    def _compute_double_ev(
        self,
        hand: HandState,
        dealer_upcard: str,
        remaining_counts: Counter[str]
    ) -> float:
        """Compute double EV: hit once then stand with 2x bet."""
        total_remaining = sum(remaining_counts.values())
        if total_remaining == 0:
            # No cards to hit, just stand with 2x
            return 2.0 * self._compute_stand_ev(hand, dealer_upcard, remaining_counts)
        
        ev = 0.0
        
        # Try each possible hit card
        for rank, count in remaining_counts.items():
            if count <= 0:
                continue
            
            prob = count / total_remaining
            
            # Create hand after hit
            new_cards = list(hand.cards) + [rank]
            new_hand = HandState.from_cards(new_cards)
            
            # Remove this card from remaining
            hit_remaining = remaining_counts.copy()
            hit_remaining[rank] -= 1
            if hit_remaining[rank] <= 0:
                del hit_remaining[rank]
            
            # Stand with 2x bet
            stand_ev = self._compute_stand_ev(new_hand, dealer_upcard, hit_remaining)
            double_ev = 2.0 * stand_ev
            
            ev += prob * double_ev
        
        return ev
    
    def _compute_split_ev(
        self,
        hand: HandState,
        dealer_upcard: str, 
        remaining_counts: Counter[str]
    ) -> float:
        """Compute split EV by evaluating both hands sequentially."""
        if not hand.can_split:
            raise ValueError("Cannot split non-pair hand")
        
        pair_rank = hand.cards[0]
        total_remaining = sum(remaining_counts.values())
        
        if total_remaining < 2:
            return 0.0  # Not enough cards to split
        
        ev = 0.0
        
        # Try all possible combinations of cards for the two split hands
        for card1 in remaining_counts:
            count1 = remaining_counts[card1]
            if count1 <= 0:
                continue
            
            prob1 = count1 / total_remaining
            
            # Remove card1
            after_card1 = remaining_counts.copy()
            after_card1[card1] -= 1
            if after_card1[card1] <= 0:
                del after_card1[card1]
            
            remaining_after_card1 = sum(after_card1.values())
            if remaining_after_card1 <= 0:
                continue
            
            for card2 in after_card1:
                count2 = after_card1[card2]
                if count2 <= 0:
                    continue
                
                prob2 = count2 / remaining_after_card1
                
                # Create both split hands
                hand1 = HandState.from_cards([pair_rank, card1])
                hand2 = HandState.from_cards([pair_rank, card2])
                
                # Remove card2 from remaining
                after_both_cards = after_card1.copy()
                after_both_cards[card2] -= 1
                if after_both_cards[card2] <= 0:
                    del after_both_cards[card2]
                
                # Evaluate both hands optimally
                # Hand 1: after card1 is dealt to first hand
                ev1, _ = self.engine.evaluate(
                    hand1, dealer_upcard, after_card1,
                    can_double=self.rules.das and pair_rank != "A",  # DAS and not split aces
                    can_split=False,  # Already at split limit for this example
                    split_level=1
                )
                
                # Hand 2: after both cards are dealt
                ev2, _ = self.engine.evaluate(
                    hand2, dealer_upcard, after_both_cards,
                    can_double=self.rules.das and pair_rank != "A",
                    can_split=False,
                    split_level=1
                )
                
                combined_ev = ev1 + ev2
                ev += prob1 * prob2 * combined_ev
        
        return ev
    
    def _get_dealer_distribution(
        self, 
        dealer_upcard: str, 
        remaining_counts: Counter[str]
    ) -> Dict[int, float]:
        """Get dealer final total distribution, with caching."""
        # Create cache key from remaining counts
        counts_tuple = tuple(sorted((rank, count) for rank, count in remaining_counts.items()))
        cache_key = (dealer_upcard, hash(counts_tuple))
        
        if cache_key in self._dealer_pmf_cache:
            return self._dealer_pmf_cache[cache_key]
        
        # Use engine's dealer distribution computation
        dist = self.engine._dealer_dist(dealer_upcard)
        
        # Cache result
        self._dealer_pmf_cache[cache_key] = dist
        return dist
    
    def _create_hand_for_class(self, hand_class: HandClass) -> HandState:
        """Create minimal example hand for hand class."""
        return self.classifier.create_example_hand(hand_class)