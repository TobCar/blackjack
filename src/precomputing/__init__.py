"""
Blackjack EV precomputation pipeline.

This package implements ground-up EV precomputation for blackjack,
using ev_engine.py as the authoritative EV source.

Key components:
- composition_bucketing: Deterministic deck state quantization
- hand_classification: Canonical hand descriptors (H12, S18, P_8, etc.)
- legality: Action legality masks based on rules
- ev_orchestrator: Orchestrates calls to ev_engine.py
- quantization: Int16 quantization with d1e-3 error guarantee
- parquet_io: Exact schemas and metadata for output files
- cli: Command-line interface

Usage:
    python -m precomputing.cli --ruleset_id CURRENT --m 20 --device cpu \
      --only-upcards A,T --only-hands H12,H16,S18,P_8 --sample-buckets 10000
"""

from .composition_bucketing import CompositionBucketing, BucketConfig
from .hand_classification import HandClassifier, HandClass, HandType
from .legality import LegalityMaskGenerator, ACTION_BITS
from .ev_orchestrator import EVOrchestrator, EVResult
from .quantization import EVQuantizer, create_ev_quantizer_from_results
from .parquet_io import ParquetWriter, ParquetReader, create_build_metadata

__version__ = "0.1.0"

__all__ = [
    "CompositionBucketing",
    "BucketConfig", 
    "HandClassifier",
    "HandClass",
    "HandType",
    "LegalityMaskGenerator",
    "ACTION_BITS",
    "EVOrchestrator", 
    "EVResult",
    "EVQuantizer",
    "create_ev_quantizer_from_results",
    "ParquetWriter",
    "ParquetReader",
    "create_build_metadata",
]