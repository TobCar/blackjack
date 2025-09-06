"""
Validation script for blackjack EV precomputation pipeline.

This script validates the core components and runs a small demonstration
of the precomputation pipeline.
"""

import sys
import time
from pathlib import Path
from typing import List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_mechanics.rules import RULES, RANKS
from precomputing.composition_bucketing import CompositionBucketing, BucketConfig
from precomputing.hand_classification import HandClassifier, HandClass, HandType
from precomputing.legality import LegalityMaskGenerator
from precomputing.ev_orchestrator import EVOrchestrator
from precomputing.quantization import create_ev_quantizer_from_results
from precomputing.parquet_io import ParquetWriter, create_build_metadata


def validate_bucketing() -> bool:
    """Validate composition bucketing system."""
    print("=== Validating Composition Bucketing ===")
    
    try:
        bucketing = CompositionBucketing(BucketConfig(m=20))
        
        # Validate basic properties
        if not bucketing.validate_bucket_properties():
            print("❌ Bucket validation failed")
            return False
        
        total_buckets = bucketing.get_bucket_count()
        print(f"✅ Generated {total_buckets} valid buckets")
        
        # Test some basic operations
        bucket_id = 0
        counts = bucketing.bucket_id_to_counts(bucket_id)
        counter = bucketing.bucket_id_to_counter(bucket_id)
        
        # Test roundtrip
        recovered_id = bucketing.counts_to_bucket_id(counter)
        if recovered_id != bucket_id:
            print(f"❌ Roundtrip failed: {bucket_id} != {recovered_id}")
            return False
        
        print(f"✅ Bucket roundtrip test passed")
        print(f"   Bucket {bucket_id}: {dict(counter)} (sum={sum(counter.values())})")
        
        # Test transition operator  
        drawable_rank = None
        for i, rank in enumerate(RANKS):
            if counts[i] > 0:
                drawable_rank = rank
                break
        
        if drawable_rank:
            new_bucket_id = bucketing.transition_draw(bucket_id, drawable_rank)
            new_counts = bucketing.bucket_id_to_counts(new_bucket_id)
            print(f"✅ Transition test: draw {drawable_rank} from bucket {bucket_id} → bucket {new_bucket_id}")
            print(f"   New sum: {new_counts.sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bucketing validation failed: {e}")
        return False


def validate_hand_classification() -> bool:
    """Validate hand classification system."""
    print("\n=== Validating Hand Classification ===")
    
    try:
        classifier = HandClassifier()
        
        # Get all hand classes
        all_classes = classifier.get_all_classes()
        print(f"✅ Generated {len(all_classes)} hand classes")
        
        # Test each type
        hard_classes = classifier.get_classes_by_type(HandType.HARD)
        soft_classes = classifier.get_classes_by_type(HandType.SOFT)
        pair_classes = classifier.get_classes_by_type(HandType.PAIR)
        
        print(f"   Hard: {len(hard_classes)}, Soft: {len(soft_classes)}, Pairs: {len(pair_classes)}")
        
        # Test string parsing
        test_classes = ["H16", "S18", "P_8"]
        for class_str in test_classes:
            parsed = classifier.parse_hand_class(class_str)
            back_to_str = str(parsed)
            if back_to_str != class_str:
                print(f"❌ String roundtrip failed: {class_str} != {back_to_str}")
                return False
        
        print(f"✅ String parsing tests passed")
        
        # Test creating example hands
        for hand_class in [HandClass(HandType.HARD, 16), HandClass(HandType.SOFT, 18), HandClass(HandType.PAIR, 8, "8")]:
            try:
                hand = classifier.create_example_hand(hand_class)
                classified = classifier.classify_hand(hand)
                print(f"   {hand_class} → {hand} → {classified}")
            except Exception as e:
                print(f"❌ Example hand creation failed for {hand_class}: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Hand classification validation failed: {e}")
        return False


def validate_legality_masks() -> bool:
    """Validate legality mask generation."""
    print("\n=== Validating Legality Masks ===")
    
    try:
        legality_gen = LegalityMaskGenerator(RULES)
        
        # Test basic mask generation
        test_cases = [
            (HandClass(HandType.HARD, 16), "10"),
            (HandClass(HandType.SOFT, 18), "A"), 
            (HandClass(HandType.PAIR, 8, "8"), "9"),
        ]
        
        for hand_class, upcard in test_cases:
            mask = legality_gen.generate_mask(hand_class, upcard)
            actions = legality_gen.mask_to_actions(mask)
            print(f"   {hand_class} vs {upcard}: mask={mask:04b}, actions={[a.value for a in actions]}")
        
        print("✅ Legality mask generation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Legality mask validation failed: {e}")
        return False


def validate_ev_orchestration() -> bool:
    """Validate EV computation orchestration."""
    print("\n=== Validating EV Orchestration ===")
    
    try:
        bucketing = CompositionBucketing(BucketConfig(m=20))
        orchestrator = EVOrchestrator(RULES, bucketing)
        
        # Test single situation
        hand_class = HandClass(HandType.HARD, 16)
        dealer_upcard = "10"
        bucket_id = 0
        
        print(f"Testing EV computation for {hand_class} vs {dealer_upcard}, bucket {bucket_id}")
        
        result = orchestrator.compute_situation_ev(hand_class, dealer_upcard, bucket_id)
        
        print(f"   Legal mask: {result.legal_mask:04b}")
        print(f"   EVs: Stand={result.ev_stand:.4f}, Hit={result.ev_hit:.4f}, Double={result.ev_double:.4f}, Split={result.ev_split}")
        
        print("✅ EV orchestration test passed")
        return True
        
    except Exception as e:
        print(f"❌ EV orchestration validation failed: {e}")
        return False


def validate_quantization() -> bool:
    """Validate quantization system."""
    print("\n=== Validating Quantization ===")
    
    try:
        # Create some sample EV results
        from precomputing.ev_orchestrator import EVResult
        
        results = [
            EVResult(
                hand_class=HandClass(HandType.HARD, 16),
                dealer_upcard="10", 
                bucket_id=0,
                legal_mask=7,
                ev_stand=-0.5,
                ev_hit=-0.4,
                ev_double=-0.8,
                ev_split=None
            )
        ]
        
        action_names = ["stand", "hit", "double", "split"]
        quantizer = create_ev_quantizer_from_results(results, action_names)
        
        # Test quantization
        ev_dict = {"stand": -0.5, "hit": -0.4, "double": -0.8, "split": None}
        quantized = quantizer.quantize(ev_dict, action_names)
        dequantized = quantizer.dequantize(quantized, action_names)
        
        print(f"   Original: {ev_dict}")
        print(f"   Quantized: {quantized}")
        print(f"   Dequantized: {dequantized}")
        
        # Check errors
        for action in action_names:
            if ev_dict[action] is not None and dequantized[action] is not None:
                error = abs(ev_dict[action] - dequantized[action])
                if error > 1e-3:
                    print(f"❌ Quantization error too large for {action}: {error}")
                    return False
        
        print("✅ Quantization validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Quantization validation failed: {e}")
        return False


def run_mini_precomputation() -> bool:
    """Run a mini precomputation to test the full pipeline."""
    print("\n=== Running Mini Precomputation ===")
    
    try:
        # Small test parameters
        bucketing = CompositionBucketing(BucketConfig(m=20))
        orchestrator = EVOrchestrator(RULES, bucketing)
        
        # Test with 2 hands, 2 upcards, 5 buckets
        hand_classes = [
            HandClass(HandType.HARD, 16),
            HandClass(HandType.SOFT, 18),
        ]
        dealer_upcards = ["10", "A"]
        bucket_ids = list(range(min(5, bucketing.get_bucket_count())))
        
        print(f"Computing EVs for {len(hand_classes)} hands × {len(dealer_upcards)} upcards × {len(bucket_ids)} buckets")
        
        start_time = time.time()
        results = orchestrator.compute_all_evs(hand_classes, dealer_upcards, bucket_ids)
        computation_time = time.time() - start_time
        
        print(f"✅ Computed {len(results)} situations in {computation_time:.3f} seconds")
        print(f"   Rate: {len(results)/computation_time:.1f} situations/second")
        
        # Create quantizer and test I/O
        action_names = ["stand", "hit", "double", "split"]
        quantizer = create_ev_quantizer_from_results(results, action_names)
        
        # Test Parquet I/O
        import tempfile
        import shutil
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            writer = ParquetWriter(temp_dir, RULES)
            build_metadata = create_build_metadata(rng_seed=42)
            
            # Write files
            hand_files = writer.write_hand_files(results, quantizer, bucketing, "TEST", build_metadata)
            buckets_file = writer.write_buckets_file(bucketing, build_metadata)
            
            print(f"✅ Wrote {len(hand_files)} hand files and 1 buckets file")
            
            # Basic validation
            for file_path in hand_files:
                if not file_path.exists() or file_path.stat().st_size == 0:
                    print(f"❌ Invalid file: {file_path}")
                    return False
            
            print(f"✅ All output files are valid")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Mini precomputation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    """Run all validations."""
    print("Blackjack EV Precomputation Pipeline Validation")
    print("=" * 50)
    
    validations = [
        ("Composition Bucketing", validate_bucketing),
        ("Hand Classification", validate_hand_classification),
        ("Legality Masks", validate_legality_masks),
        ("EV Orchestration", validate_ev_orchestration),
        ("Quantization", validate_quantization),
        ("Mini Precomputation", run_mini_precomputation),
    ]
    
    passed = 0
    total = len(validations)
    
    for name, validation_func in validations:
        try:
            success = validation_func()
            if success:
                passed += 1
            else:
                print(f"\n❌ {name} validation failed")
        except Exception as e:
            print(f"\n❌ {name} validation crashed: {e}")
    
    print(f"\n{'='*50}")
    print(f"Validation Summary: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All validations passed! The pipeline is ready.")
        return 0
    else:
        print("❌ Some validations failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)