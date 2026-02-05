#!/usr/bin/env python3
"""
Update imports after directory restructure.
Run from repo root: python scripts/update_imports.py
"""

import os
import re
from pathlib import Path

# Import mappings: old → new
IMPORT_MAPPINGS = {
    # Root → core
    'from orthon.pipeline': 'from orthon.core.pipeline',
    'from orthon.data_reader': 'from orthon.core.data_reader',
    'from orthon.validation': 'from orthon.core.validation',
    'from orthon.prism_client': 'from orthon.core.prism_client',
    'from orthon.api': 'from orthon.core.api',
    'from orthon.server': 'from orthon.core.server',
    'import orthon.pipeline': 'import orthon.core.pipeline',
    'import orthon.data_reader': 'import orthon.core.data_reader',

    # analysis → cohorts
    'from orthon.analysis.cohort_detection': 'from orthon.cohorts.detection',
    'from orthon.analysis.baseline_discovery': 'from orthon.cohorts.baseline',
    'from orthon.cohort_discovery': 'from orthon.cohorts.discovery',
    'from orthon.analysis': 'from orthon.cohorts',

    # intake → ingest
    'from orthon.intake.upload': 'from orthon.ingest.upload',
    'from orthon.intake.transformer': 'from orthon.ingest.transform',
    'from orthon.intake.validate': 'from orthon.ingest.validate',
    'from orthon.intake': 'from orthon.ingest',
    'import orthon.intake': 'import orthon.ingest',

    # window → manifest
    'from orthon.window_recommender': 'from orthon.manifest.window_recommender',
    'from orthon.window.characteristic_time': 'from orthon.manifest.characteristic_time',
    'from orthon.window.domain_clock': 'from orthon.manifest.domain_clock',
    'from orthon.window.system_window': 'from orthon.manifest.system_window',
    'from orthon.window.manifest_generator': 'from orthon.manifest.generator',
    'from orthon.window': 'from orthon.manifest',
    'import orthon.window': 'import orthon.manifest',

    # Root concierge → services
    'from orthon.concierge': 'from orthon.services.concierge',

    # static → explorer/static
    'orthon/static/': 'orthon/explorer/static/',
}


def update_file(filepath: Path) -> int:
    """Update imports in a single file. Returns number of changes."""
    if filepath.suffix != '.py':
        return 0

    try:
        content = filepath.read_text()
    except Exception:
        return 0

    original = content
    for old, new in IMPORT_MAPPINGS.items():
        content = content.replace(old, new)

    if content != original:
        filepath.write_text(content)
        return 1
    return 0


def main():
    """Update all imports in the repository."""
    repo_root = Path(__file__).parent.parent
    orthon_dir = repo_root / 'orthon'

    if not orthon_dir.exists():
        print("Error: orthon/ directory not found")
        return

    changed = 0
    for filepath in orthon_dir.rglob('*.py'):
        if '_legacy' in str(filepath) or '__pycache__' in str(filepath):
            continue
        if update_file(filepath):
            print(f"Updated: {filepath}")
            changed += 1

    # Also update tests/
    tests_dir = repo_root / 'tests'
    if tests_dir.exists():
        for filepath in tests_dir.rglob('*.py'):
            if '__pycache__' in str(filepath):
                continue
            if update_file(filepath):
                print(f"Updated: {filepath}")
                changed += 1

    # Also update scripts/
    scripts_dir = repo_root / 'scripts'
    if scripts_dir.exists():
        for filepath in scripts_dir.rglob('*.py'):
            if filepath.name == 'update_imports.py':
                continue
            if '__pycache__' in str(filepath):
                continue
            if update_file(filepath):
                print(f"Updated: {filepath}")
                changed += 1

    print(f"\nTotal files updated: {changed}")


if __name__ == '__main__':
    main()
