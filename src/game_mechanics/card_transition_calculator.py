from typing import Dict, Counter as CounterType, Tuple, Set
from collections import Counter
from .action_type import ActionType
from .hand_state import HandState


class CardTransitionCalculator:
    """Calculate all possible hand transitions when drawing cards"""

    def __init__(self):
        self.all_cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    def get_hit_transitions(
        self, hand_state: HandState, remaining_cards: CounterType
    ) -> Dict[str, HandState]:
        """
        Get all possible hand states after hitting (drawing one card)

        Returns: {card: resulting_hand_state}

        Example: Hard 20 -> {'A': Hard 21, '2': Bust 22, '3': Bust 23, ...}
        """
        if hand_state.is_terminal:
            return {}  # Can't hit on 21+

        transitions = {}

        for card in self.all_cards:
            if remaining_cards.get(card, 0) > 0:
                # Calculate what happens if we draw this card
                new_total = hand_state.total
                new_is_soft = hand_state.is_soft
                new_aces = 0

                if card == "A":
                    new_total += 11
                    new_aces = 1
                    new_is_soft = True
                else:
                    new_total += int(card)

                # Handle ace adjustment
                if new_total > 21 and (new_is_soft or new_aces > 0):
                    new_total -= 10
                    new_is_soft = False

                new_cards = hand_state.cards + (card,)
                new_state = HandState(
                    cards=new_cards,
                    total=new_total,
                    is_soft=new_is_soft,
                    card_count=hand_state.card_count + 1,
                )

                transitions[card] = new_state

        return transitions

    def get_split_transitions(
        self,
        hand_state: HandState,
        remaining_cards: CounterType,
        one_card_after_split_aces: bool = True,
    ) -> Dict[Tuple[str, str], Tuple[HandState, HandState]]:
        """
        Get ALL possible outcomes when splitting a pair

        Returns: {(card1, card2): (hand1_after_card1, hand2_after_card2)}

        This returns ORDERED pairs (card1 to Hand 1 first, then card2 to Hand 2).

        If one_card_after_split_aces is True, post-split hands starting with an Ace are forced terminal (one card only).
        """
        if not hand_state.can_split:
            return {}  # Can only split pairs

        # Need at least 2 cards remaining total for split (ordered draws)
        total_remaining = sum(remaining_cards.values())
        if total_remaining < 2:
            return {}

        transitions = {}

        # Enumerate all ordered combinations of cards (card1 first, then card2)
        for i, card1 in enumerate(self.all_cards):
            if remaining_cards.get(card1, 0) <= 0:
                continue

            # After drawing card1 to first split hand, update remaining cards
            remaining_after_card1 = remaining_cards.copy()
            remaining_after_card1[card1] -= 1

            for j, card2 in enumerate(self.all_cards):
                if remaining_after_card1.get(card2, 0) <= 0:
                    continue

                # Calculate resulting states for each split hand
                pair_card = hand_state.split_card
                hand1_result = HandState.from_cards([pair_card, card1])
                hand2_result = HandState.from_cards([pair_card, card2])

                if one_card_after_split_aces and pair_card == "A":
                    hand1_result = HandState.from_cards(
                        [pair_card, card1], force_terminal=True
                    )
                    hand2_result = HandState.from_cards(
                        [pair_card, card2], force_terminal=True
                    )

                transitions[(card1, card2)] = (hand1_result, hand2_result)

        return transitions

    def get_available_actions(
        self,
        hand_state: HandState,
        is_initial_hand: bool = True,
        cashout_allowed: bool = True,
    ) -> Set[ActionType]:
        """Get all available actions for a given hand state"""

        if hand_state.is_terminal:
            return {ActionType.STAND}  # Must stand on 21+

        actions = {ActionType.HIT, ActionType.STAND}

        if is_initial_hand and hand_state.card_count == 2:
            actions.add(ActionType.DOUBLE)

            if cashout_allowed:
                actions.add(ActionType.SURRENDER_ANY_TIME)

            if hand_state.can_split:
                actions.add(ActionType.SPLIT)

        return actions


def demonstrate_transitions():
    """Demonstrate the card transition calculator"""

    calc = CardTransitionCalculator()

    # Example: Hitting soft 15
    example_hand = HandState.from_cards(["A", "4"])
    print(f"Starting with: {example_hand}")

    # Simulate a shoe with some cards remaining
    remaining = Counter(
        {
            "A": 4,
            "2": 4,
            "3": 4,
            "4": 4,
            "5": 4,
            "6": 4,
            "7": 4,
            "8": 4,
            "9": 4,
            "10": 16,
        }
    )

    hit_transitions = calc.get_hit_transitions(example_hand, remaining)

    print(f"\nHit transitions from {example_hand}:")
    for card, resulting_state in hit_transitions.items():
        print(f"  Draw {card} -> {resulting_state}")

    # Example: A pair for splitting
    pair_8s = HandState.from_cards(["8", "8"])
    print(f"\nStarting with: {pair_8s}")

    if pair_8s.can_split:
        split_transitions = calc.get_split_transitions(pair_8s, remaining)
        print(
            f"\nSplit transitions from {pair_8s} ({len(split_transitions)} total combinations):"
        )
        # Show first few examples
        for i, ((card1, card2), (hand1, hand2)) in enumerate(split_transitions.items()):
            print(f"  Hand1 draws {card1} -> {hand1}, Hand2 draws {card2} -> {hand2}")

    # Show available actions
    actions = calc.get_available_actions(example_hand)
    print(f"\nAvailable actions for {example_hand}: {[a.value for a in actions]}")

    # Show standing on 21
    drew_to_21 = HandState.from_cards(["A", "4", "6"])
    actions_after_21 = calc.get_available_actions(drew_to_21)
    hit_transitions = calc.get_hit_transitions(drew_to_21, remaining)
    print(
        f"\nAvailable actions for {drew_to_21}: {[a.value for a in actions_after_21]}"
    )
    print(f"Hit transitions for {drew_to_21}: {[a.value for a in hit_transitions]}")

    # Show a busted state
    bust_hand = HandState.from_cards(["10", "8", "5"])
    print(f"\nBusted hand: {bust_hand}")
    bust_actions = calc.get_available_actions(bust_hand)
    hit_transitions = calc.get_hit_transitions(bust_hand, remaining)
    print(f"Available actions: {[a.value for a in bust_actions]}")
    print(f"Available hit transitions: {[a.value for a in hit_transitions]}")


if __name__ == "__main__":
    demonstrate_transitions()
