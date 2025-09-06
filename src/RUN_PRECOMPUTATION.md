# How to Run the Blackjack EV Precomputation

## Option 1: Using the wrapper script (works from anywhere)

```bash
# Navigate to the src directory first
cd /Users/tobias/coding/python/blackjack/src

# Then run any of these commands:

# Quick test with minimal data
python precompute.py --m 5 --only-upcards A,T --only-hands H12,H16 --sample-buckets 5 --verbose

# Acceptance test (as specified in requirements)
python precompute.py --ruleset_id CURRENT --m 20 --device cpu \
  --only-upcards A,T --only-hands H12,H16,S18,P_8 --sample-buckets 10000

# Medium test (more hands and upcards)
python precompute.py --ruleset_id CURRENT --m 20 --device cpu \
  --only-upcards A,2,3,4,5,6,7,8,9,T --only-hands H12,H16,S18,P_8,P_A --sample-buckets 5000

# Full production run (WARNING: will take a very long time and use lots of disk space)
python precompute.py --ruleset_id CURRENT --m 20 --device cpu
```

## Option 2: Using Python module (must be in src directory)

```bash
# Make sure you're in the src directory
cd /Users/tobias/coding/python/blackjack/src

# Then run:
python -m precomputing.cli --ruleset_id CURRENT --m 20 --device cpu \
  --only-upcards A,T --only-hands H12,H16,S18,P_8 --sample-buckets 10000
```

## Option 3: Direct script execution

```bash
cd /Users/tobias/coding/python/blackjack/src
python precomputing/cli.py --help
```

## Verifying Output

After running any precomputation, verify the results:

```bash
cd /Users/tobias/coding/python/blackjack/src
python precomputing/verify_output.py
```

## Understanding the Parameters

- `--ruleset_id`: Identifier for the ruleset (default: CURRENT)
- `--m`: Grid sum for composition bucketing (20 = baseline, 5 = quick test)
- `--device`: Computation device (only 'cpu' supported currently)
- `--only-upcards`: Limit to specific dealer upcards (e.g., 'A,T' means Ace and Ten only)
- `--only-hands`: Limit to specific hand classes (e.g., 'H12,H16,S18,P_8')
- `--sample-buckets`: Randomly sample this many buckets instead of computing all
- `--output-dir`: Where to save the Parquet files (default: precomputed_tables)
- `--seed`: Random seed for reproducible results
- `--verbose`: Show detailed progress information

## Performance Expectations

- **m=5, 5 buckets**: ~0.1 seconds
- **m=20, 1000 buckets, 4 hands, 2 upcards**: ~10 seconds  
- **m=20, 10000 buckets, 4 hands, 2 upcards**: ~60 seconds
- **Full m=20 (10M+ buckets)**: Hours to days depending on hand/upcard selection

## Output Files

The precomputation generates:
- **Hand files**: `{HandClass}_{DealerUpcard}.parquet` (e.g., H16_A.parquet)
- **Buckets file**: `buckets.parquet` (all valid deck compositions)

Each hand file contains 1000 rows per sampled bucket with columns:
- `bucket_id`: Unique bucket identifier
- `c_A`, `c_2`, ..., `c_T`: Card counts in the bucket
- `legal_mask`: Bitmask of legal actions (1=stand, 2=hit, 4=double, 8=split)
- `ev_stand_q`, `ev_hit_q`, `ev_double_q`, `ev_split_q`: Quantized EV values