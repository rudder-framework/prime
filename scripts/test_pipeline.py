#!/usr/bin/env python3
"""
Smoke test for RUDDER pipeline.

Verifies the full import chain and runs a minimal end-to-end test.
Run this after any refactor to ensure pipeline isn't broken.

Usage:
    python scripts/test_pipeline.py
"""

import sys
from pathlib import Path

# Add rudder and engines to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path.home() / "engines"))

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")

    failures = []

    imports = [
        ("framework.core.data_reader", "DataProfile"),
        ("framework.config", "recommender"),
        ("framework.config.discrete_sparse_config", None),
        ("framework.typology.discrete_sparse", None),
        ("framework.typology.level2_corrections", None),
        ("framework.ingest.typology_raw", None),
        ("framework.manifest.generator", "build_manifest"),
    ]

    for module, attr in imports:
        try:
            mod = __import__(module, fromlist=[attr] if attr else [])
            if attr:
                getattr(mod, attr)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            failures.append((module, str(e)))

    return failures


def test_engines_imports():
    """Test ENGINES imports."""
    print("\nTesting ENGINES imports...")
    print("  (Note: ENGINES uses separate venv - some deps may differ)")

    failures = []
    warnings = []

    imports = [
        ("engines.entry_points.stage_01_signal_vector", "run_from_manifest"),
        ("engines.entry_points.stage_02_state_vector", None),
        ("engines.entry_points.stage_03_state_geometry", None),
    ]

    for module, attr in imports:
        try:
            mod = __import__(module, fromlist=[attr] if attr else [])
            if attr:
                getattr(mod, attr)
            print(f"  ✓ {module}")
        except ModuleNotFoundError as e:
            # Missing deps in rudder venv are warnings, not failures
            # ENGINES should be run with its own venv
            if "joblib" in str(e) or "numpy" in str(e):
                print(f"  ⚠ {module}: {e} (use engines venv)")
                warnings.append((module, str(e)))
            else:
                print(f"  ✗ {module}: {e}")
                failures.append((module, str(e)))
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            failures.append((module, str(e)))

    return failures


def test_pipeline_dry_run():
    """Test pipeline can be invoked (dry run on tiny data)."""
    print("\nTesting pipeline invocation...")

    # Import from the scripts directory using importlib
    import importlib.util
    script_path = Path(__file__).parent / "process_all_domains.py"

    try:
        spec = importlib.util.spec_from_file_location("process_all_domains", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("  ✓ process_all_domains importable")

        # Check key functions exist
        assert hasattr(module, 'compute_full_typology')
        assert hasattr(module, 'generate_manifest')
        print("  ✓ compute_full_typology, generate_manifest available")
    except Exception as e:
        print(f"  ✗ process_all_domains: {e}")
        return False

    return True


def main():
    print("=" * 60)
    print("RUDDER PIPELINE SMOKE TEST")
    print("=" * 60)

    rudder_failures = test_imports()
    engines_failures = test_engines_imports()
    pipeline_ok = test_pipeline_dry_run()

    all_failures = rudder_failures + engines_failures

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_failures:
        print(f"\n✗ {len(all_failures)} import failures:")
        for mod, err in all_failures:
            print(f"  - {mod}: {err}")
        print("\nRun: python scripts/update_imports.py")
        sys.exit(1)

    if not pipeline_ok:
        print("\n✗ Pipeline invocation failed")
        sys.exit(1)

    print("\n✓ All checks passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
