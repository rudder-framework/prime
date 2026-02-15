"""Generate SQL_README.md documenting all Prime SQL views and their schemas."""

import duckdb
from pathlib import Path


def generate_sql_readme(con: duckdb.DuckDBPyConnection, output_dir: Path) -> None:
    """Write SQL_README.md into output_dir/sql/ describing all views in the session."""

    output_dir = Path(output_dir)

    # Get all views
    views = con.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_type = 'VIEW'
          AND table_name LIKE 'v_%'
        ORDER BY table_name
    """).fetchall()

    lines = [
        "# Prime SQL Analysis Output",
        "",
        "Generated from Manifold output in this directory.",
        f"Total views: {len(views)}",
        "",
        "SQL source files are alongside this README.",
        "",
        "---",
        "",
    ]

    # Group views by prefix
    groups = {
        'geometry': [],
        'signal': [],
        'anomaly': [],
        'coupling': [],
        'canary': [],
        'curvature': [],
        'brittleness': [],
        'transition': [],
        'regime': [],
        'break': [],
        'deviation': [],
        'dashboard': [],
        'threshold': [],
        'trajectory': [],
        'stability': [],
        'health': [],
        'chart': [],
        'graph': [],
        'other': [],
    }

    for (view_name,) in views:
        placed = False
        for prefix in groups:
            if prefix in view_name:
                groups[prefix].append(view_name)
                placed = True
                break
        if not placed:
            groups['other'].append(view_name)

    for group_name, view_names in groups.items():
        if not view_names:
            continue

        lines.append(f"## {group_name.replace('_', ' ').title()}")
        lines.append("")

        for view_name in sorted(view_names):
            lines.append(f"### {view_name}")
            lines.append("")

            # Get columns and types
            try:
                cols = con.execute(f"""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = '{view_name}'
                    ORDER BY ordinal_position
                """).fetchall()

                lines.append("| Column | Type |")
                lines.append("|--------|------|")
                for col_name, col_type in cols:
                    lines.append(f"| {col_name} | {col_type} |")
                lines.append("")

                # Row count
                count = con.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
                lines.append(f"Rows: {count:,}")
                lines.append("")

                # Sample
                lines.append("Sample (first 5 rows):")
                lines.append("```")
                sample = con.execute(f"SELECT * FROM {view_name} LIMIT 5").fetchdf()
                lines.append(sample.to_string(index=False))
                lines.append("```")
                lines.append("")

            except Exception as e:
                lines.append(f"*Error reading view: {e}*")
                lines.append("")

    sql_out = output_dir / "sql"
    sql_out.mkdir(parents=True, exist_ok=True)
    readme_path = sql_out / "SQL_README.md"
    readme_path.write_text("\n".join(lines))
    print(f"  → SQL_README.md written ({len(views)} views documented)")
