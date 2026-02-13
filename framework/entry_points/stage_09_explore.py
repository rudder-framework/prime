"""
09: Explore Entry Point
========================

Pure orchestration - calls manifold explorer.
Visualizes behavioral dynamics from PRISM outputs.

Stages: PRISM output dir → manifold visualization

Renders 2D/3D phase portraits, velocity fields, and trajectories.
"""

from pathlib import Path
from typing import Optional, Tuple

from framework.explorer.loader import ManifoldLoader
from framework.explorer.renderer import ManifoldRenderer
from framework.explorer.models import ExplorerConfig


def run(
    data_dir: str,
    output_path: Optional[str] = None,
    two_d: bool = False,
    axes: Tuple[int, int] = (0, 1),
    show_velocity: bool = True,
    show_force: bool = True,
    show_labels: bool = True,
    verbose: bool = True,
) -> None:
    """
    Run manifold explorer visualization.

    Args:
        data_dir: Path to PRISM output directory
        output_path: Save visualization to file (png, pdf, svg)
        two_d: Render 2D projection instead of 3D
        axes: Axes for 2D projection
        show_velocity: Show velocity vectors
        show_force: Show force vectors
        show_labels: Show entity labels
        verbose: Print progress
    """
    if verbose:
        print("=" * 70)
        print("09: EXPLORE - Manifold Visualization")
        print("=" * 70)

    config = ExplorerConfig(
        show_velocity=show_velocity,
        show_force=show_force,
        show_labels=show_labels,
    )

    loader = ManifoldLoader(Path(data_dir), config)
    state = loader.load_state()

    if verbose:
        print(f"  Entities: {len(state.entities)}")
        print(f"  Dimensions: {state.n_dims}")

    renderer = ManifoldRenderer()

    if two_d:
        renderer.render_2d_projection(state, axes=axes, output_path=output_path)
    else:
        renderer.render(state, output_path=output_path, show=output_path is None)

    if output_path and verbose:
        print(f"\nSaved: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="09: Manifold Explorer")
    parser.add_argument('data_dir', help='PRISM output directory')
    parser.add_argument('-o', '--output', help='Output file (png, pdf, svg)')
    parser.add_argument('--2d', dest='two_d', action='store_true',
                        help='2D projection')
    parser.add_argument('--axes', default='0,1',
                        help='Axes for 2D projection (e.g. "0,1")')
    parser.add_argument('--no-velocity', action='store_true')
    parser.add_argument('--no-force', action='store_true')
    parser.add_argument('--no-labels', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()
    axes = tuple(int(x) for x in args.axes.split(','))

    run(
        args.data_dir,
        output_path=args.output,
        two_d=args.two_d,
        axes=axes,
        show_velocity=not args.no_velocity,
        show_force=not args.no_force,
        show_labels=not args.no_labels,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
