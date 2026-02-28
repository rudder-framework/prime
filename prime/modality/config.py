"""
Modality configuration: unit-based signal grouping.

Auto-discovers signal modalities from signals.parquet unit column.
Optional YAML override for merge/split/rename.
signal_0 is always excluded (ordering axis, not a sensor).
Null/empty unit → 'unclassified' group.
Singletons (1 signal) are processed with a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import polars as pl


@dataclass
class ModalityConfig:
    name: str           # e.g. "thermal"
    unit: str           # e.g. "Rankine"
    signals: list[str]  # e.g. ["T2", "T24", "T30", "T50"]
    is_singleton: bool  # True if len(signals) == 1


def resolve_modalities(
    signals_path: Path,
    override_yaml: Optional[Path] = None,
) -> list[ModalityConfig]:
    """
    Read signals.parquet, group by unit column.

    Rules:
    - signal_0 excluded (ordering axis)
    - Null/empty unit → 'unclassified'
    - YAML overrides applied (merge/split/rename)
    - Singletons marked but included

    Returns sorted list[ModalityConfig] by name.
    """
    signals_df = pl.read_parquet(signals_path)

    if "unit" not in signals_df.columns:
        # No unit info — everything goes in unclassified
        signal_ids = [
            s for s in signals_df["signal_id"].to_list()
            if s != "signal_0"
        ]
        is_singleton = len(signal_ids) == 1
        return [ModalityConfig(
            name="unclassified",
            unit="",
            signals=signal_ids,
            is_singleton=is_singleton,
        )]

    # Exclude signal_0
    df = signals_df.filter(pl.col("signal_id") != "signal_0")

    # Normalize unit: null/empty → 'unclassified'
    df = df.with_columns(
        pl.when(pl.col("unit").is_null() | (pl.col("unit") == ""))
        .then(pl.lit("unclassified"))
        .otherwise(pl.col("unit"))
        .alias("unit_norm")
    )

    # Group by normalized unit
    unit_groups: dict[str, list[str]] = {}
    for row in df.iter_rows(named=True):
        unit_key = row["unit_norm"]
        if unit_key not in unit_groups:
            unit_groups[unit_key] = []
        unit_groups[unit_key].append(row["signal_id"])

    # Apply YAML overrides if provided
    if override_yaml is not None and override_yaml.exists():
        unit_groups = _apply_overrides(unit_groups, override_yaml)

    # Build ModalityConfig list
    modalities: list[ModalityConfig] = []
    for unit_key, signal_list in sorted(unit_groups.items()):
        # Derive name from unit (sanitize for use as column prefix)
        name = _unit_to_name(unit_key)
        is_singleton = len(signal_list) == 1
        modalities.append(ModalityConfig(
            name=name,
            unit=unit_key,
            signals=sorted(signal_list),
            is_singleton=is_singleton,
        ))

    return sorted(modalities, key=lambda m: m.name)


def _unit_to_name(unit: str) -> str:
    """Convert a unit string to a safe column prefix name."""
    if unit == "unclassified":
        return "unclassified"
    # Replace special chars, lowercase, strip leading/trailing underscores
    name = unit.lower()
    for ch in ["/", " ", "-", "(", ")", ".", ",", "*", "^", "°"]:
        name = name.replace(ch, "_")
    name = name.strip("_")
    # Collapse multiple underscores
    while "__" in name:
        name = name.replace("__", "_")
    return name or "unclassified"


def _apply_overrides(
    unit_groups: dict[str, list[str]],
    override_yaml: Path,
) -> dict[str, list[str]]:
    """
    Apply YAML override file to unit groups.

    YAML format (example):
        overrides:
          - action: rename
            from: Rankine
            to: thermal
          - action: merge
            sources: [psia, psid]
            to: pressure
          - action: split
            source: unclassified
            groups:
              speed: [Nf, Nc]
              other: []   # remainder

    Actions are applied in order. Unknown actions are skipped.
    """
    try:
        import yaml
    except ImportError:
        print("  [modality] WARNING: PyYAML not available — skipping overrides")
        return unit_groups

    try:
        with open(override_yaml) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"  [modality] WARNING: could not load override YAML: {e}")
        return unit_groups

    overrides = config.get("overrides", [])
    groups = {k: list(v) for k, v in unit_groups.items()}

    for override in overrides:
        action = override.get("action", "")

        if action == "rename":
            src = override.get("from", "")
            dst = override.get("to", "")
            if src in groups and src != dst:
                signals = groups.pop(src)
                existing = groups.get(dst, [])
                groups[dst] = existing + signals

        elif action == "merge":
            sources = override.get("sources", [])
            dst = override.get("to", "")
            merged: list[str] = []
            for src in sources:
                if src in groups:
                    merged.extend(groups.pop(src))
            existing = groups.get(dst, [])
            groups[dst] = existing + merged

        elif action == "split":
            source = override.get("source", "")
            sub_groups = override.get("groups", {})
            if source not in groups:
                continue
            remaining = list(groups.pop(source))
            assigned: set[str] = set()
            for sub_name, sub_signals in sub_groups.items():
                if sub_signals:  # explicit list
                    groups[sub_name] = sub_signals
                    assigned.update(sub_signals)
                # else: gets remainder below
            # Assign remainder to any empty sub_group
            for sub_name, sub_signals in sub_groups.items():
                if not sub_signals:
                    leftover = [s for s in remaining if s not in assigned]
                    groups[sub_name] = leftover
                    break

    return groups
