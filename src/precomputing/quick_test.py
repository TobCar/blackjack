"""
Quick test of the precomputation pipeline with minimal configuration.
"""

import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_mechanics.rules import RULES
from precomputing.composition_bucketing import CompositionBucketing, BucketConfig
from precomputing.hand_classification import HandClassifier, HandClass, HandType


def test_bucketing():
    """Test bucketing with small m."""
    print("Testing composition bucketing with m=5...")
    
    bucketing = CompositionBucketing(BucketConfig(m=5))  # Much smaller
    
    if not bucketing.validate_bucket_properties():
        print("❌ Bucket validation failed")
        return False
    
    total_buckets = bucketing.get_bucket_count()
    print(f"✅ Generated {total_buckets} buckets")
    
    # Test basic operations
    if total_buckets > 0:
        bucket_id = 0
        counts = bucketing.bucket_id_to_counts(bucket_id)
        counter = bucketing.bucket_id_to_counter(bucket_id)
        print(f"   Bucket 0: {dict(counter)} (sum={sum(counter.values())})")
        
        # Test roundtrip
        recovered_id = bucketing.counts_to_bucket_id(counter)
        if recovered_id != bucket_id:
            print(f"❌ Roundtrip failed: {bucket_id} != {recovered_id}")
            return False
        
        print(f"✅ Bucket roundtrip test passed")
    
    return True


def test_hand_classification():
    """Test hand classification."""
    print("\nTesting hand classification...")
    
    classifier = HandClassifier()
    all_classes = classifier.get_all_classes()
    print(f"✅ Generated {len(all_classes)} hand classes")
    
    # Test a few specific ones
    test_cases = [
        ("H16", HandClass(HandType.HARD, 16)),
        ("S18", HandClass(HandType.SOFT, 18)),
        ("P_8", HandClass(HandType.PAIR, 8, "8")),
    ]
    
    for class_str, expected in test_cases:
        parsed = classifier.parse_hand_class(class_str)
        if parsed != expected:
            print(f"❌ Parsing failed: {class_str} -> {parsed}, expected {expected}")
            return False
    
    print("✅ Hand classification tests passed")
    return True


def test_ev_computation():
    """Test EV computation on a tiny example."""
    print("\nTesting EV computation (tiny example)...")
    
    from precomputing.ev_orchestrator import EVOrchestrator
    
    bucketing = CompositionBucketing(BucketConfig(m=5))
    orchestrator = EVOrchestrator(RULES, bucketing)
    
    # Just test one situation
    hand_class = HandClass(HandType.HARD, 16)
    dealer_upcard = "10"
    bucket_id = 0
    
    print(f"Computing EV for {hand_class} vs {dealer_upcard}, bucket {bucket_id}")
    
    start_time = time.time()
    result = orchestrator.compute_situation_ev(hand_class, dealer_upcard, bucket_id)
    elapsed = time.time() - start_time
    
    print(f"✅ Computed in {elapsed:.3f} seconds")
    print(f"   Legal mask: {result.legal_mask}")
    print(f"   Stand EV: {result.ev_stand:.4f}" if result.ev_stand is not None else "   Stand EV: None")
    print(f"   Hit EV: {result.ev_hit:.4f}" if result.ev_hit is not None else "   Hit EV: None")
    
    return True


def main():
    """Run quick tests."""
    print("Quick Test of Blackjack EV Precomputation Pipeline")
    print("=" * 50)
    
    tests = [
        test_bucketing,
        test_hand_classification,
        test_ev_computation,
    ]
    
    for i, test_func in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] Running {test_func.__name__}...")
        try:
            success = test_func()
            if not success:
                print(f"❌ Test {test_func.__name__} failed")
                return 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print(f"\n{'='*50}")
    print("✅ All quick tests passed!")
    print("\nTo run the full pipeline, use:")
    print("python -m precomputing.cli --ruleset_id CURRENT --m 20 --device cpu \\")
    print("  --only-upcards A,T --only-hands H12,H16,S18,P_8 --sample-buckets 10000")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)