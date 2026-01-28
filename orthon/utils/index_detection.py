"""
ORTHON Index Detection
======================

Time: auto-detectable (ISO 8601, Unix epoch, date strings, Excel serial)
Other: requires user input (space, frequency, scale, cycle)

Time is the one index dimension where ORTHON can be smart.
Everything else, ask the user.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import re

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class IndexDetectionResult:
    """Result of index column detection."""
    column: Optional[str]
    index_type: str  # timestamp, unix_seconds, unix_ms, cycle, integer_sequence, unknown
    dimension: str   # time, space, frequency, scale, unknown
    confidence: str  # high, medium, low
    needs_user_input: bool
    message: Optional[str] = None

    # Sampling info (for time indices)
    sampling_interval_seconds: Optional[float] = None
    sampling_unit: Optional[str] = None
    sampling_value: Optional[float] = None
    regularity: Optional[str] = None  # regular, mostly_regular, irregular

    # Detected format (for timestamps)
    detected_format: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class IndexDetector:
    """
    Detect index column type and sampling interval.

    Time: auto-detected from standard formats
    Other: requires user input
    """

    # Common timestamp column names
    TIME_COLUMNS = {
        'timestamp', 'time', 'datetime', 'date', 't', 'ts',
        'time_stamp', 'sample_time', 'record_time', 'created_at',
        'measured_at', 'observation_time', 'recorded_at', 'dt',
        'obs_date', 'obs_time', 'observation_date', 'record_date'
    }

    # Substrings that indicate time columns (for fuzzy matching)
    TIME_SUBSTRINGS = {'date', 'time', 'timestamp', 'datetime'}

    # Common cycle/sequence column names
    CYCLE_COLUMNS = {
        'cycle', 'cycles', 'sample', 'index', 'i', 'n',
        'step', 'iteration', 'sequence', 'row', 'sample_id'
    }

    # Spatial column names (need user confirmation)
    SPATIAL_COLUMNS = {
        'x', 'y', 'z', 'position', 'distance', 'depth', 'height',
        'length', 'radius', 'r', 'location', 'station', 'chainage'
    }

    # Frequency column names (need user confirmation)
    FREQUENCY_COLUMNS = {
        'frequency', 'freq', 'f', 'hz', 'khz', 'bin', 'band'
    }

    # Timestamp formats to try (Python strptime)
    TIME_FORMATS = [
        ('%Y-%m-%dT%H:%M:%S.%fZ', 'ISO 8601 with ms'),
        ('%Y-%m-%dT%H:%M:%SZ', 'ISO 8601'),
        ('%Y-%m-%dT%H:%M:%S', 'ISO 8601 no Z'),
        ('%Y-%m-%d %H:%M:%S.%f', 'Datetime with ms'),
        ('%Y-%m-%d %H:%M:%S', 'Datetime'),
        ('%Y-%m-%d', 'Date only'),
        ('%m/%d/%Y %H:%M:%S', 'US datetime'),
        ('%m/%d/%Y', 'US date'),
        ('%d/%m/%Y %H:%M:%S', 'EU datetime'),
        ('%d/%m/%Y', 'EU date'),
        ('%d-%b-%Y %H:%M:%S', 'DD-Mon-YYYY datetime'),
        ('%d-%b-%Y', 'DD-Mon-YYYY'),
        ('%Y%m%d', 'YYYYMMDD compact'),
        ('%Y%m%d%H%M%S', 'YYYYMMDDHHMMSS compact'),
    ]

    def detect_index_column(self, df) -> IndexDetectionResult:
        """
        Detect which column is the index and what type it is.

        Args:
            df: polars or pandas DataFrame

        Returns:
            IndexDetectionResult with detection info
        """
        # Convert to consistent format
        columns, get_col, get_dtype = self._normalize_dataframe(df)

        candidates = []

        for col in columns:
            result = self._analyze_column(get_col(col), col, get_dtype(col))
            if result.index_type != 'not_index':
                candidates.append(result)

        # No candidates found
        if not candidates:
            return IndexDetectionResult(
                column=None,
                index_type='unknown',
                dimension='unknown',
                confidence='low',
                needs_user_input=True,
                message='No index column detected. Please specify which column is the index.'
            )

        # Sort by confidence
        confidence_order = {'high': 0, 'medium': 1, 'low': 2}
        candidates.sort(key=lambda x: confidence_order.get(x.confidence, 3))

        return candidates[0]

    def _normalize_dataframe(self, df) -> Tuple:
        """Normalize DataFrame access for polars or pandas."""
        if HAS_POLARS and isinstance(df, pl.DataFrame):
            columns = df.columns
            get_col = lambda c: df[c].to_list()
            get_dtype = lambda c: str(df[c].dtype)
        elif HAS_PANDAS and hasattr(df, 'columns'):
            columns = list(df.columns)
            get_col = lambda c: df[c].tolist()
            get_dtype = lambda c: str(df[c].dtype)
        else:
            raise ValueError("DataFrame must be polars or pandas")

        return columns, get_col, get_dtype

    def _analyze_column(self, values: List, name: str, dtype: str) -> IndexDetectionResult:
        """Analyze a single column for index potential."""

        name_lower = name.lower()

        # Filter out nulls for analysis
        values = [v for v in values if v is not None and str(v) != 'nan']
        if len(values) < 2:
            return IndexDetectionResult(
                column=name, index_type='not_index', dimension='unknown',
                confidence='low', needs_user_input=True
            )

        # Check if monotonically increasing (for numeric types)
        is_monotonic = self._is_monotonic(values)

        # === TIME DETECTION (auto-detectable) ===

        # Check for native Date/Datetime dtypes (pandas/polars)
        dtype_lower = dtype.lower()
        is_native_datetime = any(x in dtype_lower for x in ['date', 'datetime', 'datetime64'])

        if is_native_datetime:
            # Convert native dates to datetime objects
            try:
                if 'datetime64' in dtype_lower:
                    # pandas datetime64
                    datetimes = [v.to_pydatetime() if hasattr(v, 'to_pydatetime') else v for v in values]
                else:
                    # Polars Date or python date objects
                    datetimes = [datetime(v.year, v.month, v.day) if hasattr(v, 'year') else v for v in values]
                interval = self._detect_interval_from_datetimes(datetimes)
                return IndexDetectionResult(
                    column=name,
                    index_type='timestamp',
                    dimension='time',
                    confidence='high',
                    needs_user_input=False,
                    detected_format='Native date/datetime',
                    **interval
                )
            except (AttributeError, TypeError, ValueError):
                pass

        # Check for time column names (exact match or substring)
        is_time_name = name_lower in self.TIME_COLUMNS or any(sub in name_lower for sub in self.TIME_SUBSTRINGS)

        if is_time_name:
            parsed, fmt = self._try_parse_timestamps(values)
            if parsed is not None:
                interval = self._detect_interval_from_datetimes(parsed)
                return IndexDetectionResult(
                    column=name,
                    index_type='timestamp',
                    dimension='time',
                    confidence='high',
                    needs_user_input=False,
                    detected_format=fmt,
                    **interval
                )

        # Check for Unix timestamps (numeric)
        if dtype in ['int64', 'float64', 'Int64', 'Float64']:
            try:
                sample = float(values[0])

                # Unix seconds (1970-2100 range)
                if 1e9 < sample < 4.1e9 and is_monotonic:
                    datetimes = [datetime.fromtimestamp(float(v)) for v in values]
                    interval = self._detect_interval_from_datetimes(datetimes)
                    return IndexDetectionResult(
                        column=name,
                        index_type='unix_seconds',
                        dimension='time',
                        confidence='medium',
                        needs_user_input=False,
                        detected_format='Unix epoch (seconds)',
                        **interval
                    )

                # Unix milliseconds
                if 1e12 < sample < 4.1e15 and is_monotonic:
                    datetimes = [datetime.fromtimestamp(float(v) / 1000) for v in values]
                    interval = self._detect_interval_from_datetimes(datetimes)
                    return IndexDetectionResult(
                        column=name,
                        index_type='unix_milliseconds',
                        dimension='time',
                        confidence='medium',
                        needs_user_input=False,
                        detected_format='Unix epoch (milliseconds)',
                        **interval
                    )

                # Excel serial date (1900-2100 range: ~0 to ~72000)
                if 1 < sample < 72000 and is_monotonic and not self._is_integer_sequence(values):
                    # Excel dates have fractional parts for time
                    if any('.' in str(v) for v in values[:10]):
                        return IndexDetectionResult(
                            column=name,
                            index_type='excel_serial',
                            dimension='time',
                            confidence='low',
                            needs_user_input=True,
                            message=f'Column "{name}" may be Excel serial dates. Please confirm.'
                        )
            except (ValueError, TypeError):
                pass

        # Try to parse string values as timestamps
        if dtype in ['str', 'object', 'Utf8', 'String']:
            parsed, fmt = self._try_parse_timestamps(values)
            if parsed is not None:
                interval = self._detect_interval_from_datetimes(parsed)
                return IndexDetectionResult(
                    column=name,
                    index_type='timestamp',
                    dimension='time',
                    confidence='medium' if fmt else 'low',
                    needs_user_input=False,
                    detected_format=fmt,
                    **interval
                )

        # === OTHER INDEX TYPES (need user input) ===

        # Cycle/sequence column
        if name_lower in self.CYCLE_COLUMNS and is_monotonic:
            return IndexDetectionResult(
                column=name,
                index_type='cycle',
                dimension='time',  # Cycles are time-like
                confidence='high',
                needs_user_input=True,
                message=f'Column "{name}" appears to be a cycle counter. What is the duration per cycle?'
            )

        # Spatial column
        if name_lower in self.SPATIAL_COLUMNS and is_monotonic:
            return IndexDetectionResult(
                column=name,
                index_type='spatial',
                dimension='space',
                confidence='medium',
                needs_user_input=True,
                message=f'Column "{name}" appears to be spatial. What is the unit? (m, ft, km, etc.)'
            )

        # Frequency column
        if name_lower in self.FREQUENCY_COLUMNS and is_monotonic:
            return IndexDetectionResult(
                column=name,
                index_type='frequency',
                dimension='frequency',
                confidence='medium',
                needs_user_input=True,
                message=f'Column "{name}" appears to be frequency bins. What is the unit? (Hz, kHz, etc.)'
            )

        # Generic integer sequence (fallback)
        if self._is_integer_sequence(values) and is_monotonic:
            return IndexDetectionResult(
                column=name,
                index_type='integer_sequence',
                dimension='unknown',
                confidence='low',
                needs_user_input=True,
                message=f'Column "{name}" is a monotonic integer sequence. What does it represent?'
            )

        # Monotonic but unknown
        if is_monotonic:
            return IndexDetectionResult(
                column=name,
                index_type='unknown',
                dimension='unknown',
                confidence='low',
                needs_user_input=True,
                message=f'Column "{name}" is monotonic but type is unclear. What does it represent?'
            )

        return IndexDetectionResult(
            column=name, index_type='not_index', dimension='unknown',
            confidence='low', needs_user_input=True
        )

    def _is_monotonic(self, values: List) -> bool:
        """Check if values are monotonically increasing."""
        try:
            for i in range(1, len(values)):
                if values[i] < values[i-1]:
                    return False
            return True
        except (TypeError, ValueError):
            return False

    def _is_integer_sequence(self, values: List) -> bool:
        """Check if values form a clean integer sequence."""
        try:
            for v in values[:100]:  # Check first 100
                if float(v) != int(float(v)):
                    return False
            return True
        except (ValueError, TypeError):
            return False

    def _try_parse_timestamps(self, values: List) -> Tuple[Optional[List[datetime]], Optional[str]]:
        """Try to parse string values as timestamps."""
        sample = values[:10]  # Test on first 10 values

        for fmt, fmt_name in self.TIME_FORMATS:
            try:
                parsed = [datetime.strptime(str(v), fmt) for v in sample]
                # If sample works, parse all
                all_parsed = [datetime.strptime(str(v), fmt) for v in values]
                return all_parsed, fmt_name
            except (ValueError, TypeError):
                continue

        # Try ISO format with fromisoformat
        try:
            parsed = [datetime.fromisoformat(str(v).replace('Z', '+00:00')) for v in values]
            return parsed, 'ISO 8601'
        except (ValueError, TypeError):
            pass

        return None, None

    def _detect_interval_from_datetimes(self, datetimes: List[datetime]) -> Dict[str, Any]:
        """Detect sampling interval from parsed datetimes."""
        if len(datetimes) < 2:
            return {}

        # Calculate differences in seconds
        diffs = []
        for i in range(1, len(datetimes)):
            diff = (datetimes[i] - datetimes[i-1]).total_seconds()
            diffs.append(diff)

        if not diffs:
            return {}

        # Find modal (most common) interval
        from collections import Counter
        rounded_diffs = [round(d, 1) for d in diffs]
        counter = Counter(rounded_diffs)
        modal_diff = counter.most_common(1)[0][0]

        mean_diff = sum(diffs) / len(diffs)

        # Classify the interval
        seconds = modal_diff
        if seconds < 1:
            unit, value = 'milliseconds', seconds * 1000
        elif seconds < 60:
            unit, value = 'seconds', seconds
        elif seconds < 3600:
            unit, value = 'minutes', seconds / 60
        elif seconds < 86400:
            unit, value = 'hours', seconds / 3600
        elif seconds < 604800:
            unit, value = 'days', seconds / 86400
        elif seconds < 2592000:
            unit, value = 'weeks', seconds / 604800
        else:
            unit, value = 'months', seconds / 2592000

        # Check regularity (coefficient of variation)
        if len(diffs) > 1:
            std_diff = (sum((d - mean_diff)**2 for d in diffs) / len(diffs))**0.5
            cv = std_diff / max(mean_diff, 1e-10)

            if cv < 0.01:
                regularity = 'regular'
            elif cv < 0.1:
                regularity = 'mostly_regular'
            else:
                regularity = 'irregular'
        else:
            regularity = 'unknown'

        return {
            'sampling_interval_seconds': seconds,
            'sampling_unit': unit,
            'sampling_value': round(value, 2),
            'regularity': regularity
        }


# =============================================================================
# LLM CONCIERGE PROMPTS FOR INDEX DETECTION
# =============================================================================

PROMPT_UNKNOWN_INDEX = """
I detected a potential index column "{column}" but I'm not sure what it represents.

Sample values: {samples}

What type of index is this?
1. Time (seconds, minutes, hours, days)
2. Cycles (engine cycles, iterations)
3. Spatial (distance along pipe, position)
4. Frequency (Hz bins)
5. Other (please specify)
"""

PROMPT_CYCLE_DURATION = """
The column "{column}" appears to be a cycle counter (1, 2, 3, ...).

To convert this to time for causality analysis, I need to know:
- How long is each cycle? (e.g., "3 minutes", "1 flight hour", "1 sample")
"""

PROMPT_SPATIAL_INDEX = """
The column "{column}" appears to be a spatial index.

Please specify:
- What unit? (meters, feet, kilometers, etc.)
- What dimension? (length along pipe, height, radial distance)
"""

PROMPT_FREQUENCY_INDEX = """
The column "{column}" appears to be frequency bins.

Please specify:
- What unit? (Hz, kHz, MHz)
- What is the frequency resolution?
"""


def get_index_detection_prompt(result: IndexDetectionResult) -> Optional[str]:
    """Get the appropriate LLM prompt for user clarification."""
    if not result.needs_user_input:
        return None

    if result.index_type == 'cycle':
        return PROMPT_CYCLE_DURATION.format(column=result.column)
    elif result.index_type == 'spatial':
        return PROMPT_SPATIAL_INDEX.format(column=result.column)
    elif result.index_type == 'frequency':
        return PROMPT_FREQUENCY_INDEX.format(column=result.column)
    elif result.index_type in ('unknown', 'integer_sequence'):
        return PROMPT_UNKNOWN_INDEX.format(
            column=result.column,
            samples=result.message or 'N/A'
        )

    return result.message


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def detect_index(df) -> IndexDetectionResult:
    """
    Convenience function to detect index from DataFrame.

    Usage:
        result = detect_index(df)
        if result.needs_user_input:
            # Ask user via concierge
            prompt = get_index_detection_prompt(result)
        else:
            # Auto-detected time index
            print(f"Index: {result.column} ({result.detected_format})")
            print(f"Sampling: {result.sampling_value} {result.sampling_unit}")
    """
    detector = IndexDetector()
    return detector.detect_index_column(df)
