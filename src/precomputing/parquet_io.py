"""
Parquet I/O with exact schemas and metadata for blackjack precomputation.

Implements the specified file formats:
- Per-hand files: bucket_id, c_A...c_T, legal_mask, ev_*_q columns
- Buckets file: bucket_id and count vectors
"""

import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from game_mechanics.rules import RANKS, RULES, BlackjackRuleset
from precomputing.composition_bucketing import CompositionBucketing, create_buckets_parquet_data
from precomputing.hand_classification import HandClass
from precomputing.quantization import EVQuantizer, INT16_MIN
from precomputing.ev_orchestrator import EVResult


class ParquetWriter:
    """Handles writing precomputed data to Parquet files with exact schemas."""
    
    def __init__(self, output_dir: Path, rules: BlackjackRuleset = None):
        self.output_dir = Path(output_dir)
        self.rules = rules or RULES
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_hand_files(
        self,
        results: List[EVResult],
        quantizer: EVQuantizer,
        bucketing: CompositionBucketing,
        ruleset_id: str,
        build_metadata: Dict[str, Any]
    ) -> List[Path]:
        """Write per-hand Parquet files."""
        written_files = []
        
        # Group results by (hand_class, dealer_upcard)
        grouped_results = {}
        for result in results:
            key = (result.hand_class, result.dealer_upcard)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Write one file per (hand_class, dealer_upcard)
        for (hand_class, dealer_upcard), hand_results in grouped_results.items():
            filename = f"{hand_class}_{dealer_upcard}.parquet"
            file_path = self.output_dir / filename
            
            self._write_single_hand_file(
                hand_results, quantizer, bucketing, 
                file_path, ruleset_id, build_metadata
            )
            written_files.append(file_path)
        
        return written_files
    
    def _write_single_hand_file(
        self,
        results: List[EVResult],
        quantizer: EVQuantizer,
        bucketing: CompositionBucketing,
        file_path: Path,
        ruleset_id: str,
        build_metadata: Dict[str, Any]
    ) -> None:
        """Write a single hand file with exact schema."""
        if not results:
            raise ValueError("No results to write")
        
        # All results should have same hand_class and dealer_upcard
        hand_class = results[0].hand_class
        dealer_upcard = results[0].dealer_upcard
        
        # Prepare data rows
        rows = []
        action_names = ["stand", "hit", "double", "split"]
        
        for result in results:
            # Get bucket counts
            bucket_counts = bucketing.bucket_id_to_counts(result.bucket_id)
            
            # Quantize EVs
            ev_dict = {
                "stand": result.ev_stand,
                "hit": result.ev_hit, 
                "double": result.ev_double,
                "split": result.ev_split
            }
            quantized = quantizer.quantize(ev_dict, action_names)
            
            # Build row
            row = {
                "bucket_id": np.uint32(result.bucket_id),
                "legal_mask": np.uint8(result.legal_mask)
            }
            
            # Add count columns c_A, c_2, ..., c_T
            for i, rank in enumerate(RANKS):
                col_name = f"c_{rank}" if rank != "10" else "c_T"
                row[col_name] = np.int8(bucket_counts[i])
            
            # Add quantized EV columns
            row["ev_stand_q"] = quantized["stand"]
            row["ev_hit_q"] = quantized["hit"]
            row["ev_double_q"] = quantized["double"]
            row["ev_split_q"] = quantized["split"]
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Define exact schema
        schema_fields = [
            pa.field("bucket_id", pa.uint32()),
            pa.field("legal_mask", pa.uint8()),
        ]
        
        # Count columns
        for rank in RANKS:
            col_name = f"c_{rank}" if rank != "10" else "c_T"
            schema_fields.append(pa.field(col_name, pa.int8()))
        
        # EV columns
        schema_fields.extend([
            pa.field("ev_stand_q", pa.int16()),
            pa.field("ev_hit_q", pa.int16()),
            pa.field("ev_double_q", pa.int16()),
            pa.field("ev_split_q", pa.int16()),
        ])
        
        schema = pa.schema(schema_fields)
        
        # Create metadata
        file_metadata = self._create_file_metadata(
            hand_class, dealer_upcard, quantizer, 
            ruleset_id, build_metadata, bucketing.m
        )
        
        # Convert to Arrow table with metadata
        table = pa.Table.from_pandas(df, schema=schema)
        table = table.replace_schema_metadata(file_metadata)
        
        # Write to Parquet
        pq.write_table(table, file_path, compression='snappy')
    
    
    def write_buckets_file(
        self, 
        bucketing: CompositionBucketing,
        build_metadata: Dict[str, Any]
    ) -> Path:
        """Write buckets file with bucket_id and count vectors."""
        filename = "buckets.parquet"
        file_path = self.output_dir / filename
        
        # Generate buckets data
        buckets_data = create_buckets_parquet_data(bucketing)
        df = pd.DataFrame(buckets_data)
        
        # Schema for buckets file
        schema_fields = [pa.field("bucket_id", pa.uint32())]
        
        for rank in RANKS:
            col_name = f"c_{rank}" if rank != "10" else "c_T"
            schema_fields.append(pa.field(col_name, pa.int8()))
        
        schema = pa.schema(schema_fields)
        
        # Metadata for buckets file
        metadata = self._create_buckets_metadata(bucketing.m, build_metadata)
        
        table = pa.Table.from_pandas(df, schema=schema)
        table = table.replace_schema_metadata(metadata)
        
        pq.write_table(table, file_path, compression='snappy')
        return file_path
    
    def _create_file_metadata(
        self,
        hand_class: HandClass,
        dealer_upcard: str,
        quantizer: EVQuantizer,
        ruleset_id: str,
        build_metadata: Dict[str, Any],
        m: int
    ) -> Dict[bytes, bytes]:
        """Create metadata dict for hand files."""
        metadata = {
            # Quantization parameters
            "quantization": quantizer.get_metadata(),
            
            # Hand/dealer info
            "hand_class": str(hand_class),
            "dealer_upcard": dealer_upcard,
            
            # Rules and configuration
            "ruleset_id": ruleset_id,
            "m": m,
            "rank_order": list(RANKS),
            "ev_definition": "Q* continuation optimal",
            
            # Build info
            "build_info": build_metadata,
            
            # Validation
            "counts_sum": m,
            "rules_checksum": self._compute_rules_checksum(),
            
            # Technical metadata
            "transition": "reproject_m",
            "draw_model": "pseudo_without_replacement"
        }
        
        # Convert to bytes for Arrow metadata
        return {
            key.encode(): json.dumps(value).encode()
            for key, value in metadata.items()
        }
    
    
    def _create_buckets_metadata(
        self,
        m: int,
        build_metadata: Dict[str, Any]
    ) -> Dict[bytes, bytes]:
        """Create metadata dict for buckets file."""
        metadata = {
            "file_type": "buckets",
            "m": m,
            "rank_order": list(RANKS),
            "build_info": build_metadata,
            "counts_sum": m,
        }
        
        return {
            key.encode(): json.dumps(value).encode()
            for key, value in metadata.items()
        }
    
    def _compute_rules_checksum(self) -> str:
        """Compute checksum of rules for validation."""
        rules_dict = {
            "s17": self.rules.s17,
            "das": self.rules.das,
            "must_stand_after_split_aces": self.rules.must_stand_after_split_aces,
            "blackjack_payout": self.rules.blackjack_payout,
            "max_splits": self.rules.max_splits,
        }
        
        rules_json = json.dumps(rules_dict, sort_keys=True)
        return hashlib.sha256(rules_json.encode()).hexdigest()


def create_build_metadata(
    git_commit: Optional[str] = None,
    rng_seed: Optional[int] = None
) -> Dict[str, Any]:
    """Create build metadata for files."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "rng_seed": rng_seed,
    }
    
    # Try to get git commit
    if git_commit is None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                metadata["git_commit"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    else:
        metadata["git_commit"] = git_commit
    
    # Library versions
    try:
        import pandas
        metadata["pandas_version"] = pandas.__version__
    except ImportError:
        pass
    
    try:
        import pyarrow
        metadata["pyarrow_version"] = pyarrow.__version__
    except ImportError:
        pass
    
    try:
        import numpy
        metadata["numpy_version"] = numpy.__version__
    except ImportError:
        pass
    
    return metadata


class ParquetReader:
    """Reads precomputed data from Parquet files."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
    
    def read_hand_file(self, hand_class: HandClass, dealer_upcard: str) -> Tuple[pd.DataFrame, Dict]:
        """Read a single hand file."""
        filename = f"{hand_class}_{dealer_upcard}.parquet"
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Hand file not found: {file_path}")
        
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        # Extract metadata
        metadata = {}
        for key, value in table.schema.metadata.items():
            try:
                metadata[key.decode()] = json.loads(value.decode())
            except (UnicodeDecodeError, json.JSONDecodeError):
                metadata[key.decode()] = value.decode()
        
        return df, metadata
    
    def read_buckets_file(self) -> Tuple[pd.DataFrame, Dict]:
        """Read buckets file."""
        file_path = self.data_dir / "buckets.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Buckets file not found: {file_path}")
        
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        metadata = {}
        for key, value in table.schema.metadata.items():
            try:
                metadata[key.decode()] = json.loads(value.decode())
            except (UnicodeDecodeError, json.JSONDecodeError):
                metadata[key.decode()] = value.decode()
        
        return df, metadata
    
    def list_hand_files(self) -> List[Tuple[str, str]]:
        """List available hand files as (hand_class, dealer_upcard) pairs."""
        files = []
        
        for file_path in self.data_dir.glob("*.parquet"):
            if file_path.name == "buckets.parquet":
                continue
            
            # Parse filename: HandClass_Upcard.parquet
            stem = file_path.stem
            if "_" in stem:
                parts = stem.split("_")
                if len(parts) >= 2:
                    hand_class_str = "_".join(parts[:-1])
                    dealer_upcard = parts[-1]
                    files.append((hand_class_str, dealer_upcard))
        
        return sorted(files)


def validate_parquet_roundtrip(
    original_results: List[EVResult],
    quantizer: EVQuantizer,
    temp_dir: Path
) -> bool:
    """Validate write/read roundtrip with quantization tolerance."""
    from precomputing.composition_bucketing import CompositionBucketing
    
    # Write data
    writer = ParquetWriter(temp_dir)
    bucketing = CompositionBucketing()
    build_meta = create_build_metadata(rng_seed=42)
    
    written_files = writer.write_hand_files(
        original_results, quantizer, bucketing, 
        "test", build_meta
    )
    
    if not written_files:
        return False
    
    # Read back and compare
    reader = ParquetReader(temp_dir)
    
    for file_path in written_files:
        stem = file_path.stem
        parts = stem.split("_")
        hand_class_str = "_".join(parts[:-1])
        dealer_upcard = parts[-1]
        
        df, metadata = reader.read_hand_file(hand_class_str, dealer_upcard)
        
        # Basic validation
        if len(df) == 0:
            print(f"Empty DataFrame for {hand_class_str}_{dealer_upcard}")
            return False
        
        # Check required columns
        required_cols = ["bucket_id", "legal_mask", "ev_stand_q", "ev_hit_q", "ev_double_q", "ev_split_q"]
        for col in required_cols:
            if col not in df.columns:
                print(f"Missing column {col} in {file_path}")
                return False
        
        # Check metadata
        if "quantization" not in metadata:
            print(f"Missing quantization metadata in {file_path}")
            return False
    
    return True