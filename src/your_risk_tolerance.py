"""
Risk Tolerance Calculator - Discover your personal risk aversion parameter (γ)
by answering practical questions about your betting preferences.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class RiskQuestion:
    """A question to determine risk preference."""

    question_id: str
    question_text: str
    scenario: Dict
    answer: Optional[float] = None


class RiskToleranceCalculator:
    """
    Calculate your personal risk aversion parameter (γ) by answering
    questions about betting preferences.
    """

    def __init__(self):
        self.questions = []
        self.calculated_gammas = []

    def create_preference_questions(
        self, bankroll: float = 10000
    ) -> List[RiskQuestion]:
        """
        Generate questions to determine risk tolerance.
        Each question implies a specific gamma when answered.
        """
        questions = []

        # Question 1: Maximum acceptable bet at different edges
        questions.append(
            RiskQuestion(
                question_id="max_bet_2pct_edge",
                question_text=f"""
            You have a ${bankroll:,.0f} bankroll and found a game with:
            - 2% player edge (very good)
            - Standard blackjack variance (~1.3)
            
            What's the MAXIMUM you'd feel comfortable betting per hand?
            """,
                scenario={"edge": 0.02, "variance": 1.3, "bankroll": bankroll},
            )
        )

        # Question 2: Drawdown tolerance
        questions.append(
            RiskQuestion(
                question_id="drawdown_tolerance",
                question_text=f"""
            With a ${bankroll:,.0f} bankroll, what's the maximum drawdown 
            you could experience without losing sleep or quitting?
            
            Enter as percentage (e.g., 30 for 30% drawdown):
            """,
                scenario={"bankroll": bankroll},
            )
        )

        # Question 3: Choice between two games
        questions.append(
            RiskQuestion(
                question_id="variance_preference",
                question_text=f"""
            You must choose between two games:
            
            Game A: 1% edge, low variance (0.5), suggested bet $200
            Game B: 1.5% edge, high variance (2.0), suggested bet $150
            
            What percentage of your ${bankroll:,.0f} bankroll would you bet in Game B?
            (Enter percentage, e.g., 1.5 for 1.5%)
            """,
                scenario={"edge_b": 0.015, "variance_b": 2.0, "bankroll": bankroll},
            )
        )

        # Question 4: Session risk tolerance
        questions.append(
            RiskQuestion(
                question_id="session_risk",
                question_text=f"""
            For a 3-hour session (300 hands), what's the maximum amount
            you'd be willing to lose before walking away?
            
            Enter as percentage of ${bankroll:,.0f} bankroll:
            """,
                scenario={"hands": 300, "bankroll": bankroll},
            )
        )

        # Question 5: Certainty equivalent
        questions.append(
            RiskQuestion(
                question_id="certainty_equivalent",
                question_text=f"""
            Consider this choice:
            
            Option A: Guaranteed $100 profit
            Option B: 60% chance of $300 profit, 40% chance of $100 loss
            
            What guaranteed amount would make you indifferent between 
            the two options? (Enter dollar amount)
            """,
                scenario={"option_b_ev": 60},  # EV = 0.6*300 - 0.4*100 = 140
            )
        )

        # Question 6: Bet sizing comfort
        questions.append(
            RiskQuestion(
                question_id="bet_volatility",
                question_text=f"""
            You're playing with a 1% edge. Over 100 hands, which result
            would make you more comfortable with your bet sizing?
            
            Enter 1 or 2:
            1. Average result: +$100, worst session: -$200, best: +$400
            2. Average result: +$150, worst session: -$500, best: +$800
            
            (Enter the percentage of bankroll you'd bet for your preferred option)
            """,
                scenario={"edge": 0.01, "hands": 100, "bankroll": bankroll},
            )
        )

        return questions

    def calculate_gamma_from_answer(self, question: RiskQuestion) -> float:
        """
        Calculate implied risk aversion from a single answer.
        """
        if question.answer is None:
            return None

        if question.question_id == "max_bet_2pct_edge":
            # From CE Kelly: f* = μ/σ² - γ/2
            # Rearranged: γ = 2(μ/σ² - f*)
            edge = question.scenario["edge"]
            variance = question.scenario["variance"]
            kelly_fraction = question.answer / question.scenario["bankroll"]

            standard_kelly = edge / variance
            gamma = 2 * (standard_kelly - kelly_fraction)
            return max(0, gamma)  # Can't be negative

        elif question.question_id == "drawdown_tolerance":
            # Higher drawdown tolerance = lower risk aversion
            # Approximate mapping based on typical drawdown probabilities
            max_drawdown_pct = question.answer / 100

            # Empirical mapping (derived from simulations)
            if max_drawdown_pct >= 0.5:
                gamma = 0.5  # Very risk tolerant
            elif max_drawdown_pct >= 0.3:
                gamma = 1.5
            elif max_drawdown_pct >= 0.2:
                gamma = 2.5
            elif max_drawdown_pct >= 0.1:
                gamma = 3.5
            else:
                gamma = 4.5  # Very risk averse
            return gamma

        elif question.question_id == "variance_preference":
            # Direct calculation from chosen bet size
            edge = question.scenario["edge_b"]
            variance = question.scenario["variance_b"]
            kelly_fraction = question.answer / 100

            standard_kelly = edge / variance
            gamma = 2 * (standard_kelly - kelly_fraction)
            return max(0, gamma)

        elif question.question_id == "session_risk":
            # Session risk tolerance implies gamma through variance acceptance
            max_loss_pct = question.answer / 100
            hands = question.scenario["hands"]

            # Approximate: for 2-sigma loss to equal max_loss_pct
            # gamma ≈ 4 / (max_loss_pct * sqrt(hands))
            gamma = 4 / (max_loss_pct * np.sqrt(hands))
            return min(5, max(0.5, gamma))  # Reasonable bounds

        elif question.question_id == "certainty_equivalent":
            # Classic utility theory calculation
            ce = question.answer  # Certainty equivalent
            ev = question.scenario["option_b_ev"]

            # For CARA utility: CE = EV - (γ/2) * Variance
            # Variance of option B
            var_b = 0.6 * (300 - ev) ** 2 + 0.4 * (-100 - ev) ** 2

            gamma = 2 * (ev - ce) / var_b if var_b > 0 else 2.0
            return max(0, min(5, gamma))

        elif question.question_id == "bet_volatility":
            # Preference for lower volatility implies higher gamma
            bet_pct = question.answer / 100
            edge = question.scenario["edge"]

            # If they chose smaller bets, higher gamma
            standard_kelly = edge / 1.3  # Assuming standard BJ variance
            gamma = 2 * (standard_kelly - bet_pct)
            return max(0, gamma)

        return 2.0  # Default if calculation fails

    def calculate_overall_gamma(self, questions: List[RiskQuestion]) -> Dict:
        """
        Calculate overall gamma from all answers using different methods.
        """
        gammas = []
        weights = []

        # Weight different questions by reliability
        question_weights = {
            "max_bet_2pct_edge": 1.5,  # Most direct
            "drawdown_tolerance": 1.0,
            "variance_preference": 1.3,
            "session_risk": 0.8,
            "certainty_equivalent": 1.2,  # Classic measure
            "bet_volatility": 0.9,
        }

        for q in questions:
            gamma = self.calculate_gamma_from_answer(q)
            if gamma is not None:
                gammas.append(gamma)
                weights.append(question_weights.get(q.question_id, 1.0))
                print(f"{q.question_id}: γ = {gamma:.2f}")

        if not gammas:
            return {"error": "No valid answers provided"}

        # Calculate different aggregations
        results = {
            "mean": np.mean(gammas),
            "weighted_mean": np.average(gammas, weights=weights),
            "median": np.median(gammas),
            "std": np.std(gammas),
            "min": np.min(gammas),
            "max": np.max(gammas),
            "all_gammas": gammas,
            "confidence": (
                1.0 - (np.std(gammas) / np.mean(gammas)) if np.mean(gammas) > 0 else 0
            ),
        }

        # Recommend final gamma
        if results["std"] < 0.5:  # Consistent answers
            results["recommended"] = results["weighted_mean"]
        else:  # Inconsistent, be conservative
            results["recommended"] = results["median"] + 0.5

        return results

    def interactive_assessment(self, bankroll: float = 10000) -> float:
        """
        Run interactive assessment and return recommended gamma.
        """
        print("=" * 60)
        print("RISK TOLERANCE ASSESSMENT FOR BLACKJACK")
        print("=" * 60)
        print(f"\nAssume you have a ${bankroll:,} bankroll for all questions.\n")

        questions = self.create_preference_questions(bankroll)

        for q in questions:
            print(f"\nQuestion: {q.question_text}")
            while True:
                try:
                    answer = float(input("Your answer: "))
                    q.answer = answer
                    break
                except ValueError:
                    print("Please enter a number.")

        print("\n" + "=" * 60)
        print("CALCULATING YOUR RISK PROFILE...")
        print("=" * 60 + "\n")

        results = self.calculate_overall_gamma(questions)

        print(f"\nYour Risk Profile:")
        print(f"Recommended γ (gamma): {results['recommended']:.2f}")
        print(f"Consistency score: {results['confidence']:.1%}")

        print(f"\nWhat this means for your betting:")
        self.explain_gamma(results["recommended"], bankroll)

        return results["recommended"]

    def explain_gamma(self, gamma: float, bankroll: float):
        """
        Explain what a gamma value means in practical terms.
        """
        interpretations = []

        if gamma < 1:
            risk_level = "Aggressive"
            description = "Maximum growth, high variance accepted"
        elif gamma < 2:
            risk_level = "Moderate"
            description = "Balanced growth and risk"
        elif gamma < 3:
            risk_level = "Conservative"
            description = "Steady results preferred"
        else:
            risk_level = "Very Conservative"
            description = "Minimal risk, preservation focused"

        print(f"\nRisk Level: {risk_level}")
        print(f"Description: {description}")

        # Show example bets
        print(f"\nExample bet sizes with ${bankroll:,} bankroll:")

        for edge in [0.005, 0.01, 0.02]:
            variance = 1.3  # Standard BJ
            standard_kelly = edge / variance
            ce_kelly = standard_kelly / (1 + gamma * standard_kelly)
            bet = bankroll * ce_kelly

            print(
                f"  {edge*100:.1f}% edge: ${bet:.0f} per hand ({ce_kelly*100:.2f}% of bankroll)"
            )

    def compare_risk_profiles(self, bankroll: float = 10000):
        """
        Show how different gamma values affect betting.
        """
        edges = np.linspace(0, 0.03, 50)
        variance = 1.3

        plt.figure(figsize=(12, 6))

        # Plot 1: Bet size vs edge for different gammas
        plt.subplot(1, 2, 1)
        for gamma in [0.5, 1.0, 2.0, 3.0, 4.0]:
            kelly_fractions = []
            for edge in edges:
                if edge > 0:
                    sk = edge / variance
                    ck = sk / (1 + gamma * sk)
                    kelly_fractions.append(ck * 100)
                else:
                    kelly_fractions.append(0)

            label = f"γ={gamma}"
            if gamma == 0.5:
                label += " (Aggressive)"
            elif gamma == 2.0:
                label += " (Balanced)"
            elif gamma == 4.0:
                label += " (Conservative)"

            plt.plot(edges * 100, kelly_fractions, label=label)

        plt.xlabel("Player Edge (%)")
        plt.ylabel("Bet Size (% of Bankroll)")
        plt.title("Optimal Bet Sizing by Risk Tolerance")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Growth vs Drawdown tradeoff
        plt.subplot(1, 2, 2)
        gammas = np.linspace(0.5, 4, 50)
        edge = 0.01  # 1% edge

        growth_rates = []
        drawdown_probs = []

        for gamma in gammas:
            sk = edge / variance
            ck = sk / (1 + gamma * sk)

            # Approximate growth rate
            growth = ck * edge - 0.5 * ck**2 * variance
            growth_rates.append(growth * 100)

            # Approximate 20% drawdown probability (simplified)
            dd_prob = np.exp(-0.2 / (ck * 2))  # Rough approximation
            drawdown_probs.append(dd_prob * 100)

        plt.plot(gammas, growth_rates, label="Expected Growth Rate", color="green")
        plt.plot(gammas, drawdown_probs, label="20% Drawdown Risk", color="red")

        plt.xlabel("Risk Aversion (γ)")
        plt.ylabel("Percentage")
        plt.title("Growth vs Risk Tradeoff")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Standalone function for quick assessment
def quick_risk_assessment() -> float:
    """
    Quick version with just 3 key questions.
    """
    print("\nQUICK RISK TOLERANCE ASSESSMENT (3 questions)\n")

    bankroll = 10000
    gammas = []

    # Question 1: Direct bet sizing
    print(f"Q1: With a ${bankroll:,} bankroll and 1.5% edge,")
    bet = float(input("    what would you bet per hand? $"))
    kelly_fraction = bet / bankroll
    standard_kelly = 0.015 / 1.3
    gamma1 = max(0, 2 * (standard_kelly - kelly_fraction))
    gammas.append(gamma1)

    # Question 2: Loss tolerance
    print(f"\nQ2: What's the maximum you could lose from ${bankroll:,}")
    loss = float(input("    before you'd stop playing forever? $"))
    loss_pct = loss / bankroll
    gamma2 = 2.0 / loss_pct  # Simplified mapping
    gammas.append(gamma2)

    # Question 3: Variance preference
    print(f"\nQ3: Which would you prefer?")
    print("    1. Win $100 guaranteed")
    print("    2. 70% chance of $200, 30% chance of -$50")
    choice = input("    Enter 1 or 2: ")
    gamma3 = 3.0 if choice == "1" else 1.0
    gammas.append(gamma3)

    final_gamma = np.median(gammas)

    print(f"\n{'='*40}")
    print(f"Your risk aversion (γ): {final_gamma:.1f}")

    if final_gamma < 1.5:
        print("Profile: AGGRESSIVE - Focus on maximum growth")
    elif final_gamma < 2.5:
        print("Profile: BALANCED - Good mix of growth and safety")
    else:
        print("Profile: CONSERVATIVE - Focus on capital preservation")

    print(
        f"\nWith 1% edge, you should bet {(0.01/1.3)/(1+final_gamma*0.01/1.3)*100:.1f}% of bankroll"
    )
    print(f"{'='*40}\n")

    return final_gamma


# Example usage
if __name__ == "__main__":
    calculator = RiskToleranceCalculator()

    # Run the interactive assessment
    # gamma = calculator.interactive_assessment(bankroll=10000)

    # Or use the quick version
    gamma = quick_risk_assessment()

    print(f"\nUse γ = {gamma:.1f} in your Kelly calculations")
    print(
        f"This gives you CE Kelly fraction = Standard Kelly / (1 + {gamma:.1f} * Standard Kelly)"
    )
