"""
ORTHON PRISM Runner
===================

Interface to PRISM compute engine.
Consumes PRISM status stream for progress display.

PRISM computes. ORTHON interprets.
"""

from pathlib import Path
from typing import Generator, Dict, Any
import tempfile

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    import json


def run_prism(
    data_path: Path,
    config: dict,
    output_dir: Path,
) -> Generator[Dict[str, Any], None, None]:
    """
    Run PRISM and yield status updates.

    Args:
        data_path: Path to input data (parquet/csv)
        config: Configuration dict (window_size, window_stride, etc.)
        output_dir: Directory for PRISM outputs

    Yields:
        Status dicts: {'stage': str, 'message': str, 'progress': int, ...}
        Final yield: {'stage': 'complete', 'output_dir': str}

    Raises:
        ImportError: If prism package not installed
    """
    try:
        import prism
    except ImportError:
        raise ImportError(
            "prism-engine not installed.\n"
            "Install with: pip install prism-engine\n"
            "Or: pip install git+https://github.com/prism-engines/prism.git"
        )

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        if HAS_YAML:
            yaml.dump(config, f)
        else:
            import json
            json.dump(config, f)
        config_path = Path(f.name)

    try:
        # Consume PRISM generator
        for update in prism.run(data_path, config_path, output_dir):
            if hasattr(update, 'to_dict'):
                yield update.to_dict()
            elif isinstance(update, dict):
                yield update
            else:
                # Final yield is the output path
                yield {'stage': 'complete', 'output_dir': str(update)}
    finally:
        config_path.unlink(missing_ok=True)


def run_prism_blocking(
    data_path: Path,
    config: dict,
    output_dir: Path,
) -> Path:
    """
    Run PRISM and block until complete.

    Returns:
        Path to output directory
    """
    result_dir = None
    for update in run_prism(data_path, config, output_dir):
        if update.get('stage') == 'complete':
            result_dir = Path(update.get('output_dir', output_dir))

    return result_dir or output_dir
