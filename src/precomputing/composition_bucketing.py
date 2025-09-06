"""
Composition bucketing system for blackjack deck representation.

Implements deterministic normalization to a grid with sum m = 20 (baseline),
with tie-breaking by largest fractional remainder, then fixed rank order.
"""

from typing import Dict, List, Tuple, Optional, Counter as CounterType
from dataclasses import dataclass
from collections import Counter
import numpy as np

from game_mechanics.rules import RANKS

# Fixed rank order for deterministic tie-breaking: A,2,3,4,5,6,7,8,9,T
RANK_ORDER = list(RANKS)


@dataclass(frozen=True)
class BucketConfig:
    """Configuration for composition bucketing."""
    m: int = 20  # Grid sum (baseline=20, extensible to 30/40)
    rank_order: Tuple[str, ...] = tuple(RANK_ORDER)


class CompositionBucketing:
    """
    Handles conversion between card counts and normalized bucket representations.
    
    Key properties:
    - Deterministic normalization to grid sum m
    - Tie-breaking by largest fractional remainder, then rank order
    - Reversible: bucket_id ↔ counts
    - Transition operator for card draws
    """
    
    def __init__(self, config: BucketConfig = None):
        self.config = config or BucketConfig()
        self.m = self.config.m
        self.rank_order = self.config.rank_order
        
        # Pre-compute bucket enumeration for lookups
        self._bucket_to_counts: Dict[int, np.ndarray] = {}
        self._counts_to_bucket: Dict[Tuple[int, ...], int] = {}
        self._build_bucket_mappings()
    
    def _build_bucket_mappings(self):
        """Pre-compute all valid bucket configurations."""
        # Generate all possible count vectors that sum to m
        # This is computationally intensive for large m, but manageable for m=20
        bucket_id = 0
        
        def generate_partitions(remaining_sum: int, num_ranks: int, current: List[int]):
            nonlocal bucket_id
            
            if num_ranks == 1:
                # Last rank gets all remaining
                final_counts = current + [remaining_sum]
                counts_array = np.array(final_counts, dtype=np.int8)
                counts_tuple = tuple(final_counts)
                
                self._bucket_to_counts[bucket_id] = counts_array
                self._counts_to_bucket[counts_tuple] = bucket_id
                bucket_id += 1
                return
            
            # Distribute remaining_sum among remaining ranks
            for count in range(remaining_sum + 1):
                generate_partitions(
                    remaining_sum - count, 
                    num_ranks - 1, 
                    current + [count]
                )
        
        generate_partitions(self.m, len(self.rank_order), [])
        print(f"Generated {bucket_id} buckets for m={self.m}")
    
    def counts_to_bucket_id(self, counts: CounterType[str]) -> int:
        """Convert card counts to bucket ID via deterministic normalization."""
        # Convert to array in rank order
        count_array = np.array([counts.get(rank, 0) for rank in self.rank_order], dtype=float)
        total_cards = count_array.sum()
        
        if total_cards == 0:
            # Empty deck maps to zero bucket
            zero_counts = tuple([0] * len(self.rank_order))
            return self._counts_to_bucket[zero_counts]
        
        # Normalize to target sum m
        normalized = count_array * (self.m / total_cards)
        
        # Deterministic rounding with tie-breaking
        rounded = self._deterministic_round(normalized)
        
        # Convert to tuple for lookup
        rounded_tuple = tuple(rounded.astype(int))
        
        if rounded_tuple not in self._counts_to_bucket:
            raise ValueError(f"Invalid bucket configuration: {rounded_tuple}")
        
        return self._counts_to_bucket[rounded_tuple]
    
    def bucket_id_to_counts(self, bucket_id: int) -> np.ndarray:
        """Convert bucket ID to normalized count vector."""
        if bucket_id not in self._bucket_to_counts:
            raise ValueError(f"Invalid bucket ID: {bucket_id}")
        
        return self._bucket_to_counts[bucket_id].copy()
    
    def bucket_id_to_counter(self, bucket_id: int) -> Counter[str]:
        """Convert bucket ID to Counter of ranks."""
        counts = self.bucket_id_to_counts(bucket_id)
        return Counter({
            rank: int(count) 
            for rank, count in zip(self.rank_order, counts)
            if count > 0
        })
    
    def _deterministic_round(self, normalized: np.ndarray) -> np.ndarray:
        """
        Deterministically round to integers summing to m.
        
        Tie-breaking:
        1. Largest fractional remainder gets priority
        2. Among ties, fixed rank order (A,2,3,4,5,6,7,8,9,T) breaks ties
        """
        # Start with floor
        floored = np.floor(normalized)
        remainder = normalized - floored
        
        # How many units need to be distributed
        shortfall = int(self.m - floored.sum())
        
        if shortfall == 0:
            return floored
        
        # Create priority list: (fractional_remainder, -rank_priority, index)
        # Negative rank priority so higher priority ranks come first in sort
        priorities = [
            (remainder[i], -i, i)  # -i gives A highest priority (index 0)
            for i in range(len(remainder))
        ]
        
        # Sort by fractional remainder (descending), then by rank priority (ascending due to -)
        priorities.sort(reverse=True)
        
        # Distribute the shortfall
        result = floored.copy()
        for i in range(shortfall):
            _, _, idx = priorities[i]
            result[idx] += 1
        
        return result
    
    def transition_draw(self, bucket_id: int, drawn_rank: str) -> int:
        """
        Apply transition operator for drawing a card.
        
        Process:
        1. Get counts from bucket_id
        2. Decrement the drawn rank (guard against impossible draws)
        3. Rescale to m/(m-1) and re-normalize to sum m
        4. Return new bucket_id
        """
        counts = self.bucket_id_to_counts(bucket_id)
        
        # Find rank index
        if drawn_rank not in self.rank_order:
            raise ValueError(f"Invalid rank for draw: {drawn_rank}")
        
        rank_idx = self.rank_order.index(drawn_rank)
        
        # Guard against impossible draw
        if counts[rank_idx] <= 0:
            raise ValueError(f"Cannot draw {drawn_rank} from bucket {bucket_id}: count is {counts[rank_idx]}")
        
        # Decrement count
        new_counts = counts.copy().astype(float)
        new_counts[rank_idx] -= 1
        
        # Rescale: multiply by m/(m-1) to maintain proportions
        if self.m == 1:
            # Special case: if m=1, after draw we have empty deck
            new_counts = np.zeros_like(new_counts)
        else:
            scale_factor = self.m / (self.m - 1)
            new_counts = new_counts * scale_factor
        
        # Re-normalize and convert back to bucket_id
        normalized = self._deterministic_round(new_counts)
        normalized_tuple = tuple(normalized.astype(int))
        
        if normalized_tuple not in self._counts_to_bucket:
            raise ValueError(f"Transition produced invalid bucket: {normalized_tuple}")
        
        return self._counts_to_bucket[normalized_tuple]
    
    def get_bucket_count(self) -> int:
        """Get total number of buckets."""
        return len(self._bucket_to_counts)
    
    def validate_bucket_properties(self) -> bool:
        """Validate bucket system properties."""
        # Check that all buckets sum to m
        for bucket_id, counts in self._bucket_to_counts.items():
            if counts.sum() != self.m:
                print(f"Bucket {bucket_id} sum is {counts.sum()}, expected {self.m}")
                return False
        
        # Check bidirectional mapping consistency
        for bucket_id, counts in self._bucket_to_counts.items():
            counts_tuple = tuple(counts)
            if self._counts_to_bucket[counts_tuple] != bucket_id:
                print(f"Bidirectional mapping inconsistent for bucket {bucket_id}")
                return False
        
        print(f"Bucket system validated: {self.get_bucket_count()} buckets, all sum to {self.m}")
        return True


def create_buckets_parquet_data(bucketing: CompositionBucketing) -> List[Dict]:
    """Create data for buckets.parquet file."""
    data = []
    
    for bucket_id in range(bucketing.get_bucket_count()):
        counts = bucketing.bucket_id_to_counts(bucket_id)
        
        row = {"bucket_id": bucket_id}
        
        # Add individual rank counts as c_A, c_2, ..., c_T
        for rank, count in zip(RANK_ORDER, counts):
            col_name = f"c_{rank}" if rank != "10" else "c_T"
            row[col_name] = int(count)
        
        data.append(row)
    
    return data