"""
Prime Manifold Explorer CLI
============================

Command-line interface for manifold visualization.
"""

import argparse
from pathlib import Path

from .loader import ManifoldLoader
from .renderer import ManifoldRenderer
from .models import ExplorerConfig


def main():
    parser = argparse.ArgumentParser(
        description='Prime Manifold Explorer - Visualize behavioral dynamics'
    )
    parser.add_argument(
        'data_dir',
        help='Directory containing PRISM parquet outputs'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (png, pdf, svg)'
    )
    parser.add_argument(
        '--no-velocity',
        action='store_true',
        help='Hide velocity vectors'
    )
    parser.add_argument(
        '--no-force',
        action='store_true',
        help='Hide force vectors'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Hide entity labels'
    )
    parser.add_argument(
        '--2d',
        dest='two_d',
        action='store_true',
        help='Render 2D projection instead of 3D'
    )
    parser.add_argument(
        '--axes',
        type=str,
        default='0,1',
        help='Axes for 2D projection (e.g., "0,1" for PC1 vs PC2)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List entities and exit'
    )

    args = parser.parse_args()

    config = ExplorerConfig(
        show_velocities=not args.no_velocity,
        show_forces=not args.no_force,
        show_labels=not args.no_labels,
    )

    print(f"Loading from {args.data_dir}...")
    loader = ManifoldLoader(Path(args.data_dir), config)

    if args.list:
        entities = loader.get_entities()
        print(f"\nEntities ({len(entities)}):")
        for eid in entities:
            print(f"  {eid}")
        return

    state = loader.load_state()

    print(f"\nManifold State:")
    print(f"  Entities: {len(state.entities)}")
    print(f"  Healthy:  {state.n_healthy}")
    print(f"  Degraded: {state.n_degraded}")
    print(f"  Critical: {state.n_critical}")
    print(f"  Mean hd_slope: {state.mean_hd_slope:.4f}")

    # List critical entities
    critical = [e for e in state.entities.values() if e.regime == 2]
    if critical:
        print(f"\nCritical Entities:")
        for e in sorted(critical, key=lambda x: x.hd_slope):
            print(f"  {e.entity_id}: hd_slope={e.hd_slope:.4f}")

    renderer = ManifoldRenderer(config)

    if args.two_d:
        axes = tuple(int(x) for x in args.axes.split(','))
        renderer.render_2d_projection(state, axes=axes, output_path=args.output)
    else:
        renderer.render(state, output_path=args.output, show=not args.output)


if __name__ == '__main__':
    main()
