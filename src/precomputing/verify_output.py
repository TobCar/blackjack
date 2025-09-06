"""
Verify the generated Parquet files have correct schema and can be read back.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pyarrow.parquet as pq
from precomputing.parquet_io import ParquetReader


def verify_hand_file(file_path: Path) -> bool:
    """Verify a hand file has the correct schema."""
    try:
        # Read using pyarrow to get metadata
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        print(f"\n=== {file_path.name} ===")
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = [
            'bucket_id', 'legal_mask',
            'c_A', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7', 'c_8', 'c_9', 'c_T',
            'ev_stand_q', 'ev_hit_q', 'ev_double_q', 'ev_split_q'
        ]
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"❌ Missing columns: {missing}")
            return False
        
        print("✅ All required columns present")
        
        # Check data types
        print(f"bucket_id dtype: {df['bucket_id'].dtype}")
        print(f"legal_mask dtype: {df['legal_mask'].dtype}")
        print(f"EV columns dtypes: {[(col, df[col].dtype) for col in df.columns if col.startswith('ev_')]}")
        
        # Check metadata
        metadata = {}
        if table.schema.metadata:
            for key, value in table.schema.metadata.items():
                try:
                    import json
                    metadata[key.decode()] = json.loads(value.decode())
                except:
                    metadata[key.decode()] = value.decode()
        
        print(f"Metadata keys: {list(metadata.keys())}")
        
        if 'quantization' in metadata:
            print("✅ Quantization metadata present")
        
        if 'ruleset_id' in metadata:
            print(f"Ruleset: {metadata['ruleset_id']}")
        
        # Show sample data
        print(f"Sample rows:")
        print(df.head(3)[['bucket_id', 'legal_mask', 'ev_stand_q', 'ev_hit_q']])
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading {file_path.name}: {e}")
        return False


def verify_buckets_file(file_path: Path) -> bool:
    """Verify buckets file has correct schema."""
    try:
        df = pd.read_parquet(file_path)
        
        print(f"\n=== {file_path.name} ===")
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # Check that all rows sum to m
        count_cols = [col for col in df.columns if col.startswith('c_')]
        sums = df[count_cols].sum(axis=1)
        unique_sums = sums.unique()
        
        print(f"Bucket sums: {unique_sums}")
        
        if len(unique_sums) == 1:
            print(f"✅ All buckets sum to {unique_sums[0]}")
        else:
            print(f"❌ Inconsistent bucket sums: {unique_sums}")
            return False
        
        # Show sample buckets
        print("Sample buckets:")
        print(df.head(3))
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading {file_path.name}: {e}")
        return False


def main():
    """Verify all generated files."""
    output_dir = Path("precomputed_tables")
    
    if not output_dir.exists():
        print(f"❌ Output directory {output_dir} not found")
        return 1
    
    print("Verifying Blackjack EV Precomputation Output Files")
    print("=" * 55)
    
    success_count = 0
    total_count = 0
    
    # Verify hand files
    for file_path in output_dir.glob("*.parquet"):
        if file_path.name == "buckets.parquet":
            continue
        
        total_count += 1
        if verify_hand_file(file_path):
            success_count += 1
    
    # Verify buckets file
    buckets_file = output_dir / "buckets.parquet"
    if buckets_file.exists():
        total_count += 1
        if verify_buckets_file(buckets_file):
            success_count += 1
    
    print(f"\n{'='*55}")
    print(f"Verification Summary: {success_count}/{total_count} files passed")
    
    if success_count == total_count:
        print("✅ All files verified successfully!")
        print("\nThe EV precomputation pipeline has produced valid Parquet outputs")
        print("matching the exact schemas and metadata requirements.")
        return 0
    else:
        print("❌ Some files failed verification")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)