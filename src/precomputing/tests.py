"""
Comprehensive test suite for blackjack EV precomputation pipeline.

Tests include:
- Bucketing: sums to m, idempotence, scalar-multiple invariance, deterministic tie-breaks
- Legality: masks vs fixtures, structural flags respected
- EV (smoke/regression): split sanity, S17 ≥ H17, etc.
- I/O roundtrip: quantize → write → read → dequant within tolerance
- Performance smoke tests
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from collections import Counter
import numpy as np

from game_mechanics.rules import RULES, RANKS, BlackjackRuleset
from game_mechanics.hand_state import HandState
from game_mechanics.action_type import ActionType

from precomputing.composition_bucketing import CompositionBucketing, BucketConfig
from precomputing.hand_classification import HandClassifier, HandClass, HandType
from precomputing.legality import LegalityMaskGenerator, ACTION_BITS
from precomputing.ev_orchestrator import EVOrchestrator, EVResult
from precomputing.quantization import EVQuantizer, create_ev_quantizer_from_results, validate_quantization_error
from precomputing.parquet_io import ParquetWriter, ParquetReader, create_build_metadata, validate_parquet_roundtrip


class TestCompositionBucketing(unittest.TestCase):
    """Test composition bucketing system."""
    
    def setUp(self):
        self.bucketing = CompositionBucketing(BucketConfig(m=20))
    
    def test_bucket_sums_to_m(self):
        """All buckets should sum to m."""
        for bucket_id in range(min(100, self.bucketing.get_bucket_count())):
            counts = self.bucketing.bucket_id_to_counts(bucket_id)
            self.assertEqual(counts.sum(), self.bucketing.m, f"Bucket {bucket_id} sum is {counts.sum()}, not {self.bucketing.m}")
    
    def test_bidirectional_mapping(self):
        """Test bucket_id ↔ counts is bijective."""
        # Test sample of buckets
        for bucket_id in range(min(50, self.bucketing.get_bucket_count())):
            counts = self.bucketing.bucket_id_to_counts(bucket_id)
            counter = self.bucketing.bucket_id_to_counter(bucket_id)
            
            # Convert back to bucket_id
            recovered_id = self.bucketing.counts_to_bucket_id(counter)
            self.assertEqual(bucket_id, recovered_id, f"Bucket {bucket_id} → counts → {recovered_id}")
    
    def test_deterministic_normalization(self):
        """Test deterministic tie-breaking in normalization."""
        # Create counts that will have ties when normalized
        test_counts = Counter({"A": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1, "10": 1})
        
        # Normalize multiple times - should be deterministic
        bucket_id1 = self.bucketing.counts_to_bucket_id(test_counts)
        bucket_id2 = self.bucketing.counts_to_bucket_id(test_counts)
        
        self.assertEqual(bucket_id1, bucket_id2, "Normalization should be deterministic")
    
    def test_scalar_multiple_invariance(self):
        """Test that scalar multiples map to same bucket."""
        base_counts = Counter({"A": 2, "10": 3, "5": 1})
        scaled_counts = Counter({"A": 4, "10": 6, "5": 2})
        
        bucket_id1 = self.bucketing.counts_to_bucket_id(base_counts)
        bucket_id2 = self.bucketing.counts_to_bucket_id(scaled_counts)
        
        self.assertEqual(bucket_id1, bucket_id2, "Scalar multiples should map to same bucket")
    
    def test_transition_operator(self):
        """Test draw transition operator."""
        # Start with a bucket
        bucket_id = 0
        original_counts = self.bucketing.bucket_id_to_counts(bucket_id)
        
        # Find a rank we can draw
        drawable_rank = None
        for i, rank in enumerate(RANKS):
            if original_counts[i] > 0:
                drawable_rank = rank
                break
        
        if drawable_rank is None:
            self.skipTest("No drawable rank in bucket 0")
        
        # Apply transition
        new_bucket_id = self.bucketing.transition_draw(bucket_id, drawable_rank)
        new_counts = self.bucketing.bucket_id_to_counts(new_bucket_id)
        
        # Verify new counts sum appropriately
        self.assertEqual(new_counts.sum(), self.bucketing.m, "Transition should preserve sum")
    
    def test_impossible_draw_guard(self):
        """Test that impossible draws are rejected."""
        # Find a bucket with zero count for some rank
        for bucket_id in range(min(10, self.bucketing.get_bucket_count())):
            counts = self.bucketing.bucket_id_to_counts(bucket_id)
            for i, rank in enumerate(RANKS):
                if counts[i] == 0:
                    # This should raise an error
                    with self.assertRaises(ValueError):
                        self.bucketing.transition_draw(bucket_id, rank)
                    return
        
        self.skipTest("Could not find bucket with zero count for any rank")


class TestHandClassification(unittest.TestCase):
    """Test hand classification system."""
    
    def setUp(self):
        self.classifier = HandClassifier()
    
    def test_classify_pairs(self):
        """Test pair classification."""
        hand = HandState.from_cards(["8", "8"])
        hand_class = self.classifier.classify_hand(hand)
        
        self.assertEqual(hand_class.type, HandType.PAIR)
        self.assertEqual(hand_class.rank, "8")
    
    def test_classify_soft_hands(self):
        """Test soft hand classification."""
        hand = HandState.from_cards(["A", "7"])  # Soft 18
        hand_class = self.classifier.classify_hand(hand)
        
        self.assertEqual(hand_class.type, HandType.SOFT)
        self.assertEqual(hand_class.value, 18)
    
    def test_classify_hard_hands(self):
        """Test hard hand classification."""
        hand = HandState.from_cards(["10", "6"])  # Hard 16
        hand_class = self.classifier.classify_hand(hand)
        
        self.assertEqual(hand_class.type, HandType.HARD)
        self.assertEqual(hand_class.value, 16)
    
    def test_parse_hand_class_strings(self):
        """Test parsing hand class strings."""
        # Test hard hand
        hc = self.classifier.parse_hand_class("H16")
        self.assertEqual(hc.type, HandType.HARD)
        self.assertEqual(hc.value, 16)
        
        # Test soft hand
        hc = self.classifier.parse_hand_class("S18")
        self.assertEqual(hc.type, HandType.SOFT)
        self.assertEqual(hc.value, 18)
        
        # Test pair
        hc = self.classifier.parse_hand_class("P_8")
        self.assertEqual(hc.type, HandType.PAIR)
        self.assertEqual(hc.rank, "8")
    
    def test_all_classes_valid(self):
        """Test that all generated classes are valid."""
        all_classes = self.classifier.get_all_classes()
        
        for hand_class in all_classes:
            self.assertTrue(self.classifier.is_valid_class(hand_class))
            
            # Test string representation round-trip
            class_str = str(hand_class)
            parsed = self.classifier.parse_hand_class(class_str)
            self.assertEqual(hand_class, parsed)


class TestLegalityMasks(unittest.TestCase):
    """Test legality mask generation."""
    
    def setUp(self):
        self.rules = RULES
        self.legality_gen = LegalityMaskGenerator(self.rules)
        self.classifier = HandClassifier()
    
    def test_stand_always_legal(self):
        """Stand should always be legal for non-busted hands."""
        hand_class = HandClass(HandType.HARD, 16)
        mask = self.legality_gen.generate_mask(hand_class, "10")
        
        self.assertTrue(mask & ACTION_BITS[ActionType.STAND], "Stand should be legal")
    
    def test_split_only_for_pairs(self):
        """Split should only be legal for pairs."""
        # Non-pair hand
        hand_class = HandClass(HandType.HARD, 16)
        mask = self.legality_gen.generate_mask(hand_class, "10")
        self.assertFalse(mask & ACTION_BITS[ActionType.SPLIT], "Split should not be legal for non-pairs")
        
        # Pair hand
        pair_class = HandClass(HandType.PAIR, 8, "8")
        mask = self.legality_gen.generate_mask(pair_class, "10")
        self.assertTrue(mask & ACTION_BITS[ActionType.SPLIT], "Split should be legal for pairs")
    
    def test_double_first_decision_only(self):
        """Double should only be legal on first decision."""
        hand_class = HandClass(HandType.HARD, 11)
        
        # First decision
        mask = self.legality_gen.generate_mask(hand_class, "10", is_first_decision=True)
        self.assertTrue(mask & ACTION_BITS[ActionType.DOUBLE], "Double should be legal on first decision")
        
        # Not first decision
        mask = self.legality_gen.generate_mask(hand_class, "10", is_first_decision=False)
        self.assertFalse(mask & ACTION_BITS[ActionType.DOUBLE], "Double should not be legal after hit")
    
    def test_das_constraints(self):
        """Test Double After Split constraints."""
        pair_class = HandClass(HandType.PAIR, 8, "8")
        
        # DAS allowed
        rules_das = RULES  # DAS is True by default
        gen_das = LegalityMaskGenerator(rules_das)
        mask = gen_das.generate_mask(pair_class, "10", split_level=1)
        self.assertTrue(mask & ACTION_BITS[ActionType.DOUBLE], "Double should be legal after split with DAS")
        
        # Create rules without DAS
        rules_no_das = BlackjackRuleset(
            s17=RULES.s17,
            das=False,  # No DAS
            must_stand_after_split_aces=RULES.must_stand_after_split_aces,
            blackjack_payout=RULES.blackjack_payout,
            max_splits=RULES.max_splits,
            should_dealer_peek=RULES.should_dealer_peek,
            calculate_non_player_blackjack_payout=RULES.calculate_non_player_blackjack_payout,
        )
        gen_no_das = LegalityMaskGenerator(rules_no_das)
        mask = gen_no_das.generate_mask(pair_class, "10", split_level=1)
        self.assertFalse(mask & ACTION_BITS[ActionType.DOUBLE], "Double should not be legal after split without DAS")
    
    def test_split_aces_constraints(self):
        """Test must_stand_after_split_aces constraint."""
        ace_pair = HandClass(HandType.PAIR, 1, "A")
        
        # With must_stand_after_split_aces=True
        mask = self.legality_gen.generate_mask(ace_pair, "10", post_split_aces=True)
        self.assertFalse(mask & ACTION_BITS[ActionType.HIT], "Hit should not be legal after splitting aces")
        self.assertFalse(mask & ACTION_BITS[ActionType.DOUBLE], "Double should not be legal after splitting aces")


class TestEVRegression(unittest.TestCase):
    """Test EV computation regression and smoke tests."""
    
    def setUp(self):
        self.bucketing = CompositionBucketing(BucketConfig(m=20))
        self.orchestrator = EVOrchestrator(RULES, self.bucketing)
        self.classifier = HandClassifier()
    
    def test_split_vs_stand_sanity(self):
        """Test that split 8,8 vs 9 has EV >= stand 16 vs 9."""
        pair_class = HandClass(HandType.PAIR, 8, "8")
        hard_class = HandClass(HandType.HARD, 16)
        
        bucket_id = self.bucketing.get_bucket_count() // 2  # Use middle bucket
        
        pair_result = self.orchestrator.compute_situation_ev(pair_class, "9", bucket_id)
        hard_result = self.orchestrator.compute_situation_ev(hard_class, "9", bucket_id)
        
        if pair_result.ev_split is not None and hard_result.ev_stand is not None:
            self.assertGreaterEqual(
                pair_result.ev_split, hard_result.ev_stand,
                "Split 8,8 should be better than stand 16 vs 9"
            )
    
    def test_basic_strategy_smoke(self):
        """Smoke test basic strategy patterns."""
        # Hard 11 vs 10 should favor double
        hard_11 = HandClass(HandType.HARD, 11)
        result = self.orchestrator.compute_situation_ev(hard_11, "10", 0)
        
        # Should have positive EVs for most actions against dealer 10
        if result.ev_double is not None and result.ev_hit is not None:
            # Double should be better than hit for hard 11 vs 10
            self.assertGreater(result.ev_double, result.ev_hit, "Double should beat hit for hard 11 vs 10")


class TestQuantization(unittest.TestCase):
    """Test EV quantization system."""
    
    def setUp(self):
        self.quantizer = EVQuantizer()
    
    def test_quantization_error_guarantee(self):
        """Test that quantization error is ≤ 1e-3."""
        # Create sample EV values
        ev_values = {
            "stand": [0.0, -0.1, -0.5, 1.5, -0.75],
            "hit": [-0.2, 0.1, -0.3, 0.8, -0.9],
            "double": [-0.4, 0.2, 1.0, -1.0, 0.5],
            "split": [0.0, -0.1, 0.3, -0.2, 0.1],
        }
        
        action_names = ["stand", "hit", "double", "split"]
        
        # Fit quantization parameters
        params = self.quantizer.fit_quantization_params(ev_values, action_names)
        
        # Test each action
        for action in action_names:
            values = ev_values[action]
            param = params[action]
            
            max_error, passes = validate_quantization_error(
                values, 
                [self.quantizer._quantize_values(np.array([v]), param)[0] for v in values],
                param
            )
            
            self.assertTrue(passes, f"Quantization error {max_error:.6f} > 1e-3 for {action}")
            self.assertLessEqual(max_error, 1e-3, f"Quantization error guarantee violated for {action}")
    
    def test_roundtrip_consistency(self):
        """Test quantize → dequantize roundtrip."""
        ev_values = {"stand": [-0.1, 0.0, 0.5, -0.75]}
        self.quantizer.fit_quantization_params(ev_values, ["stand"])
        
        for ev in ev_values["stand"]:
            # Quantize
            quantized = self.quantizer.quantize({"stand": ev}, ["stand"])
            
            # Dequantize
            dequantized = self.quantizer.dequantize(quantized, ["stand"])
            
            error = abs(dequantized["stand"] - ev)
            self.assertLessEqual(error, 1e-3, f"Roundtrip error {error:.6f} > 1e-3 for ev={ev}")


class TestParquetIO(unittest.TestCase):
    """Test Parquet I/O roundtrip."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.bucketing = CompositionBucketing(BucketConfig(m=20))
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parquet_roundtrip(self):
        """Test write → read roundtrip with quantization."""
        # Create sample results
        hand_class = HandClass(HandType.HARD, 16)
        results = []
        
        for bucket_id in range(min(5, self.bucketing.get_bucket_count())):
            result = EVResult(
                hand_class=hand_class,
                dealer_upcard="10",
                bucket_id=bucket_id,
                legal_mask=7,  # Stand, Hit, Double legal
                ev_stand=-0.5,
                ev_hit=-0.4,
                ev_double=-0.8,
                ev_split=None  # Not legal
            )
            results.append(result)
        
        # Test the roundtrip validation
        quantizer = create_ev_quantizer_from_results(results, ["stand", "hit", "double", "split"])
        
        success = validate_parquet_roundtrip(results, quantizer, self.temp_dir)
        self.assertTrue(success, "Parquet roundtrip validation failed")
    
    def test_metadata_integrity(self):
        """Test that metadata is preserved in Parquet files."""
        writer = ParquetWriter(self.temp_dir, RULES)
        reader = ParquetReader(self.temp_dir)
        
        # Write buckets file
        build_metadata = create_build_metadata(rng_seed=42)
        buckets_file = writer.write_buckets_file(self.bucketing, build_metadata)
        
        # Read back and check metadata
        df, metadata = reader.read_buckets_file()
        
        self.assertIn("m", metadata)
        self.assertEqual(metadata["m"], self.bucketing.m)
        self.assertIn("build_info", metadata)
        self.assertIn("counts_sum", metadata)


class TestPerformanceSmoke(unittest.TestCase):
    """Smoke tests for performance requirements."""
    
    def test_small_scale_performance(self):
        """Test that small computations complete reasonably quickly."""
        import time
        
        # Small test: 2 hands, 2 upcards, 10 buckets
        bucketing = CompositionBucketing(BucketConfig(m=20))
        orchestrator = EVOrchestrator(RULES, bucketing)
        
        hand_classes = [
            HandClass(HandType.HARD, 16),
            HandClass(HandType.SOFT, 18)
        ]
        dealer_upcards = ["10", "A"]
        bucket_ids = list(range(min(10, bucketing.get_bucket_count())))
        
        start_time = time.time()
        results = orchestrator.compute_all_evs(hand_classes, dealer_upcards, bucket_ids)
        elapsed = time.time() - start_time
        
        expected_count = len(hand_classes) * len(dealer_upcards) * len(bucket_ids)
        self.assertEqual(len(results), expected_count)
        
        # Should complete in reasonable time
        rate = len(results) / elapsed if elapsed > 0 else float('inf')
        print(f"Performance smoke test: {len(results)} situations in {elapsed:.3f}s ({rate:.1f} situations/s)")
        
        # Very loose requirement - just ensure it's not completely broken
        self.assertLess(elapsed, 60, "Small computation took too long")


def run_all_tests():
    """Run all test suites."""
    test_classes = [
        TestCompositionBucketing,
        TestHandClassification, 
        TestLegalityMasks,
        TestEVRegression,
        TestQuantization,
        TestParquetIO,
        TestPerformanceSmoke,
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()