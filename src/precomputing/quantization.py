"""
Quantization system for EV values to int16 with guaranteed ≤1e-3 error.

Implements linear quantization with scale/offset parameters stored in metadata.
Uses INT16_MIN as sentinel value for NA/unavailable actions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Sentinel value for unavailable actions
INT16_MIN = np.iinfo(np.int16).min  # -32768
INT16_MAX = np.iinfo(np.int16).max  # 32767

# Reserve sentinel range to avoid conflicts
QUANTIZATION_MIN = INT16_MIN + 1  # -32767
QUANTIZATION_MAX = INT16_MAX  # 32767

# Error tolerance guarantee
MAX_DEQUANT_ERROR = 1e-3


@dataclass
class QuantizationParams:
    """Parameters for linear quantization."""

    scale: float
    offset: float
    min_value: float
    max_value: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "scale": self.scale,
            "offset": self.offset,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "QuantizationParams":
        """Create from dictionary."""
        return cls(
            scale=data["scale"],
            offset=data["offset"],
            min_value=data["min_value"],
            max_value=data["max_value"],
        )


class EVQuantizer:
    """Quantizes EV values to int16 with error guarantees."""

    def __init__(self):
        self._params: Dict[str, QuantizationParams] = {}

    def fit_quantization_params(
        self, ev_values: Dict[str, List[float]], action_names: List[str]
    ) -> Dict[str, QuantizationParams]:
        """
        Fit quantization parameters for each action to guarantee ≤1e-3 error.

        Args:
            ev_values: Dict mapping action names to lists of EV values
            action_names: List of action names to quantize

        Returns:
            Dictionary of quantization parameters per action
        """
        params = {}

        for action in action_names:
            if action not in ev_values or not ev_values[action]:
                # No values for this action, use default params
                params[action] = QuantizationParams(
                    scale=MAX_DEQUANT_ERROR / 2,  # Conservative default
                    offset=0.0,
                    min_value=-1.0,
                    max_value=2.0,
                )
                continue

            values = np.array(ev_values[action])
            # Remove any NaN/inf values
            values = values[np.isfinite(values)]

            if len(values) == 0:
                # No valid values
                params[action] = QuantizationParams(
                    scale=MAX_DEQUANT_ERROR / 2,
                    offset=0.0,
                    min_value=-1.0,
                    max_value=2.0,
                )
                continue

            min_val = float(values.min())
            max_val = float(values.max())

            # Handle edge case where all values are the same
            if max_val - min_val < 1e-10:
                # All values essentially identical
                params[action] = QuantizationParams(
                    scale=MAX_DEQUANT_ERROR / 2,
                    offset=min_val,
                    min_value=min_val,
                    max_value=max_val,
                )
                continue

            # Calculate scale to guarantee error ≤ MAX_DEQUANT_ERROR
            # quantization_error = scale / 2, so scale = 2 * MAX_DEQUANT_ERROR
            target_scale = 2.0 * MAX_DEQUANT_ERROR

            # But we also need to fit the range in the available quantization space
            value_range = max_val - min_val
            available_range = QUANTIZATION_MAX - QUANTIZATION_MIN
            required_scale = value_range / available_range

            # Use the more restrictive constraint
            scale = max(target_scale, required_scale)

            # Calculate offset to center the range
            offset = min_val

            params[action] = QuantizationParams(
                scale=scale, offset=offset, min_value=min_val, max_value=max_val
            )

            # Validate the quantization will work
            self._validate_params(values, params[action], action)

        self._params = params
        return params

    def _validate_params(
        self, values: np.ndarray, params: QuantizationParams, action_name: str
    ) -> None:
        """Validate that quantization parameters will achieve error guarantee."""
        # Test quantize/dequantize on the actual values
        quantized = self._quantize_values(values, params)
        dequantized = self._dequantize_values(quantized, params)

        errors = np.abs(dequantized - values)
        max_error = errors.max()

        if max_error > MAX_DEQUANT_ERROR * 1.001:  # Small tolerance for floating point
            raise ValueError(
                f"Quantization validation failed for {action_name}: "
                f"max_error={max_error:.6f} > {MAX_DEQUANT_ERROR:.6f}"
            )

    def quantize(
        self, ev_values: Dict[str, Optional[float]], action_names: List[str]
    ) -> Dict[str, np.int16]:
        """
        Quantize EV values using fitted parameters.

        Args:
            ev_values: Dict mapping action names to EV values (None for unavailable)
            action_names: List of action names

        Returns:
            Dict mapping action names to quantized int16 values
        """
        quantized = {}

        for action in action_names:
            if action not in self._params:
                raise ValueError(f"No quantization parameters for action: {action}")

            ev_val = ev_values.get(action)

            if ev_val is None or not np.isfinite(ev_val):
                # Unavailable action
                quantized[action] = np.int16(INT16_MIN)
            else:
                params = self._params[action]
                q_val = self._quantize_values(np.array([ev_val]), params)[0]
                quantized[action] = np.int16(q_val)

        return quantized

    def dequantize(
        self, quantized_values: Dict[str, np.int16], action_names: List[str]
    ) -> Dict[str, Optional[float]]:
        """
        Dequantize int16 values back to floats.

        Args:
            quantized_values: Dict mapping action names to quantized values
            action_names: List of action names

        Returns:
            Dict mapping action names to dequantized values (None for unavailable)
        """
        dequantized = {}

        for action in action_names:
            if action not in self._params:
                raise ValueError(f"No quantization parameters for action: {action}")

            q_val = quantized_values.get(action, INT16_MIN)

            if q_val == INT16_MIN:
                # Unavailable action
                dequantized[action] = None
            else:
                params = self._params[action]
                deq_val = self._dequantize_values(np.array([q_val]), params)[0]
                dequantized[action] = float(deq_val)

        return dequantized

    def _quantize_values(
        self, values: np.ndarray, params: QuantizationParams
    ) -> np.ndarray:
        """Quantize array of values using given parameters."""
        # Linear quantization: q = round((value - offset) / scale)
        normalized = (values - params.offset) / params.scale
        quantized = np.round(normalized).astype(np.int32)

        # Clamp to valid range
        quantized = np.clip(quantized, QUANTIZATION_MIN, QUANTIZATION_MAX)

        return quantized.astype(np.int16)

    def _dequantize_values(
        self, quantized: np.ndarray, params: QuantizationParams
    ) -> np.ndarray:
        """Dequantize array of values using given parameters."""
        # Linear dequantization: value = q * scale + offset
        return quantized.astype(np.float64) * params.scale + params.offset

    def get_params(self) -> Dict[str, QuantizationParams]:
        """Get fitted quantization parameters."""
        return self._params.copy()

    def set_params(self, params: Dict[str, QuantizationParams]) -> None:
        """Set quantization parameters."""
        self._params = params.copy()

    def get_metadata(self) -> Dict:
        """Get quantization metadata for Parquet file."""
        return {action: params.to_dict() for action, params in self._params.items()}

    def load_metadata(self, metadata: Dict) -> None:
        """Load quantization parameters from metadata."""
        self._params = {
            action: QuantizationParams.from_dict(params_dict)
            for action, params_dict in metadata.items()
        }


def create_ev_quantizer_from_results(
    results: List, action_names: List[str]
) -> EVQuantizer:
    """
    Create and fit quantizer from list of EVResult objects.

    Args:
        results: List of EVResult objects
        action_names: List of action names to quantize

    Returns:
        Fitted EVQuantizer
    """
    # Collect all EV values by action
    ev_values = {action: [] for action in action_names}

    for result in results:
        for action in action_names:
            ev_val = result.get_action_ev(action)
            if ev_val is not None and np.isfinite(ev_val):
                ev_values[action].append(ev_val)

    # Create and fit quantizer
    quantizer = EVQuantizer()
    quantizer.fit_quantization_params(ev_values, action_names)

    return quantizer


def validate_quantization_error(
    original_values: List[float],
    quantized_values: List[np.int16],
    params: QuantizationParams,
) -> Tuple[float, bool]:
    """
    Validate quantization error against guarantee.

    Returns:
        (max_error, passes_guarantee)
    """
    if len(original_values) != len(quantized_values):
        raise ValueError("Array lengths must match")

    # Filter out sentinel values
    valid_pairs = [
        (orig, quant)
        for orig, quant in zip(original_values, quantized_values)
        if quant != INT16_MIN and np.isfinite(orig)
    ]

    if not valid_pairs:
        return 0.0, True

    orig_array = np.array([pair[0] for pair in valid_pairs])
    quant_array = np.array([pair[1] for pair in valid_pairs], dtype=np.int16)

    # Dequantize
    dequant_array = quant_array.astype(np.float64) * params.scale + params.offset

    # Calculate errors
    errors = np.abs(dequant_array - orig_array)
    max_error = errors.max()

    passes = max_error <= MAX_DEQUANT_ERROR

    return float(max_error), passes
