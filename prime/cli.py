"""
Prime CLI
=========

Interpret Manifold parquet files.

Usage:
    python -m prime.interpret --data data/
    python -m prime.interpret --data data/ --entity engine_1
    python -m prime.interpret --data data/ --alerts
"""

import argparse
from pathlib import Path

from .db.connection import connect
from .db.schema import detect
from .views.views import (
    SignalTypologyView,
    GeometryView,
    DynamicsView,
    PhysicsView,
    UnifiedView,
)


def main():
    parser = argparse.ArgumentParser(
        description='Prime - Interpret Manifold parquet outputs'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Directory containing Manifold parquet files'
    )
    parser.add_argument(
        '--entity', '-e',
        type=str,
        default=None,
        help='Filter to specific entity_id'
    )
    parser.add_argument(
        '--view', '-v',
        type=str,
        choices=['typology', 'geometry', 'dynamics', 'physics', 'all'],
        default='all',
        help='Which view to display'
    )
    parser.add_argument(
        '--alerts',
        action='store_true',
        help='Show only critical alerts'
    )
    parser.add_argument(
        '--sql',
        action='store_true',
        help='Print SQL instead of executing'
    )
    parser.add_argument(
        '--schema',
        action='store_true',
        help='Print schema summary and exit'
    )

    args = parser.parse_args()

    # Detect schema
    print(f"Reading Manifold outputs from: {args.data}")
    schema = detect(args.data)

    if args.schema:
        print(schema.summary())
        return

    # Connect
    db = connect(args.data)
    print(f"Available tables: {db.tables()}")

    # Initialize views
    unified = UnifiedView(db, schema)

    # Handle alerts shortcut
    if args.alerts:
        if unified.dynamics:
            print("\n=== CRITICAL ALERTS ===")
            alerts = unified.dynamics.critical_entities()
            print(alerts)
        else:
            print("No dynamics.parquet found - cannot show alerts")
        return

    # Show requested view(s)
    if args.view == 'all':
        if args.entity:
            print(f"\n=== ENTITY: {args.entity} ===")
            result = unified.entity_summary(args.entity)
            for view_name, df in result.items():
                print(f"\n--- {view_name.upper()} ---")
                print(df)
        else:
            # Show samples from each view
            for view_name in ['typology', 'geometry', 'dynamics', 'physics']:
                view = getattr(unified, view_name, None)
                if view:
                    print(f"\n=== {view_name.upper()} (sample) ===")
                    if args.sql:
                        print(view.get_sql())
                    else:
                        print(view.get().head(10))
    else:
        view = getattr(unified, args.view, None)
        if view:
            if args.sql:
                print(view.get_sql(args.entity))
            else:
                print(view.get(args.entity))
        else:
            print(f"View '{args.view}' not available (missing parquet file)")

    db.close()


if __name__ == '__main__':
    main()
