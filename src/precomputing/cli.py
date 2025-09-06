"""
CLI interface for blackjack EV precomputation.

Implements the command:
precompute --ruleset_id CURRENT --m 20 --device cpu \
  --only-upcards A,T --only-hands H12,H16,S18,P_8 --sample-buckets 10000
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

from game_mechanics.rules import RULES, RANKS
from precomputing.composition_bucketing import CompositionBucketing, BucketConfig
from precomputing.hand_classification import HandClassifier, filter_hand_classes
from precomputing.ev_orchestrator import EVOrchestrator
from precomputing.quantization import create_ev_quantizer_from_results
from precomputing.parquet_io import ParquetWriter, create_build_metadata


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute blackjack EV tables"
    )
    
    parser.add_argument(
        "--ruleset_id",
        default="CURRENT",
        help="Ruleset identifier (default: CURRENT)"
    )
    
    parser.add_argument(
        "--m",
        type=int,
        default=20,
        help="Grid sum for composition bucketing (default: 20)"
    )
    
    parser.add_argument(
        "--device", 
        choices=["cpu"],
        default="cpu",
        help="Computation device (default: cpu)"
    )
    
    parser.add_argument(
        "--only-upcards",
        type=str,
        help="Comma-separated list of dealer upcards to compute (e.g., 'A,T,10')"
    )
    
    parser.add_argument(
        "--only-hands",
        type=str, 
        help="Comma-separated list of hand classes to compute (e.g., 'H12,H16,S18,P_8')"
    )
    
    parser.add_argument(
        "--sample-buckets",
        type=int,
        help="Number of buckets to sample randomly (for testing)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("precomputed_tables"),
        help="Output directory for Parquet files (default: precomputed_tables)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def parse_upcard_list(upcard_str: str) -> List[str]:
    """Parse comma-separated upcard list."""
    upcards = [u.strip() for u in upcard_str.split(",")]
    
    # Normalize T to 10
    normalized_upcards = []
    for u in upcards:
        if u == "T":
            normalized_upcards.append("10")
        else:
            normalized_upcards.append(u)
    
    # Validate upcards
    valid_upcards = set(RANKS)
    invalid = [u for u in normalized_upcards if u not in valid_upcards]
    
    if invalid:
        raise ValueError(f"Invalid upcards: {invalid}. Valid: {list(valid_upcards)} (T maps to 10)")
    
    return normalized_upcards


def progress_callback(processed: int, total: int) -> None:
    """Print progress updates."""
    if processed % max(1, total // 20) == 0 or processed == total:
        percent = 100.0 * processed / total
        print(f"Progress: {processed}/{total} ({percent:.1f}%)")


def main():
    """Main CLI entry point."""
    args = parse_arguments()
    
    # Set random seed for deterministic behavior
    random.seed(args.seed)
    
    print(f"Blackjack EV Precomputation")
    print(f"Ruleset: {args.ruleset_id}")
    print(f"Grid sum (m): {args.m}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Initialize bucketing system
    if args.verbose:
        print("Initializing composition bucketing...")
    
    config = BucketConfig(m=args.m)
    bucketing = CompositionBucketing(config)
    
    if not bucketing.validate_bucket_properties():
        print("Error: Bucket system validation failed")
        return 1
    
    total_buckets = bucketing.get_bucket_count()
    print(f"Generated {total_buckets} composition buckets")
    
    # Determine which buckets to compute
    if args.sample_buckets and args.sample_buckets < total_buckets:
        bucket_ids = random.sample(range(total_buckets), args.sample_buckets)
        print(f"Sampling {len(bucket_ids)} buckets for computation")
    else:
        bucket_ids = list(range(total_buckets))
        print(f"Computing all {len(bucket_ids)} buckets")
    
    # Determine dealer upcards
    if args.only_upcards:
        dealer_upcards = parse_upcard_list(args.only_upcards)
        print(f"Computing for upcards: {dealer_upcards}")
    else:
        dealer_upcards = list(RANKS)
        print(f"Computing for all upcards: {dealer_upcards}")
    
    # Determine hand classes
    classifier = HandClassifier()
    all_hand_classes = classifier.get_all_classes()
    
    if args.only_hands:
        hand_classes = filter_hand_classes(all_hand_classes, args.only_hands.split(","))
        print(f"Computing for {len(hand_classes)} hand classes: {[str(hc) for hc in hand_classes]}")
    else:
        hand_classes = all_hand_classes
        print(f"Computing for all {len(hand_classes)} hand classes")
    
    # Initialize EV orchestrator
    if args.verbose:
        print("Initializing EV computation engine...")
    
    orchestrator = EVOrchestrator(RULES, bucketing)
    
    # Compute EVs
    print(f"\nStarting EV computation...")
    total_combinations = len(hand_classes) * len(dealer_upcards) * len(bucket_ids)
    print(f"Total combinations: {total_combinations}")
    
    start_time = time.time()
    
    results = orchestrator.compute_all_evs(
        hand_classes=hand_classes,
        dealer_upcards=dealer_upcards,
        bucket_ids=bucket_ids,
        progress_callback=progress_callback if args.verbose else None
    )
    
    computation_time = time.time() - start_time
    print(f"\nEV computation completed in {computation_time:.2f} seconds")
    print(f"Rate: {len(results)/computation_time:.1f} situations/second")
    
    # Create quantizer
    if args.verbose:
        print("Fitting quantization parameters...")
    
    action_names = ["stand", "hit", "double", "split"]
    quantizer = create_ev_quantizer_from_results(results, action_names)
    
    # Validate quantization
    params = quantizer.get_params()
    for action, param in params.items():
        print(f"Quantization for {action}: scale={param.scale:.6f}, "
              f"range=[{param.min_value:.3f}, {param.max_value:.3f}]")
    
    # Create build metadata
    build_metadata = create_build_metadata(rng_seed=args.seed)
    
    # Write Parquet files
    print(f"\nWriting Parquet files to {args.output_dir}...")
    
    writer = ParquetWriter(args.output_dir, RULES)
    
    # Write hand files
    hand_files = writer.write_hand_files(
        results=results,
        quantizer=quantizer,
        bucketing=bucketing,
        ruleset_id=args.ruleset_id,
        build_metadata=build_metadata
    )
    
    print(f"Wrote {len(hand_files)} hand files")
    
    
    # Write buckets file
    buckets_file = writer.write_buckets_file(bucketing, build_metadata)
    print(f"Wrote buckets file: {buckets_file.name}")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\nPrecomputation completed successfully in {total_time:.2f} seconds")
    print(f"Output directory: {args.output_dir}")
    print(f"Files generated: {len(hand_files) + 1}")  # +1 for buckets
    
    return 0


def cli_entry_point():
    """Entry point for setuptools console script."""
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli_entry_point()