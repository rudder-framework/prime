"""
RUDDER Manifold Renderer
========================

Render manifold visualization with matplotlib.
Zero calculations — just draw what's there.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional

from .models import ManifoldState, ExplorerConfig


class ManifoldRenderer:
    """Render manifold visualization."""

    def __init__(self, config: ExplorerConfig = None):
        self.config = config or ExplorerConfig()

    def render(
        self,
        state: ManifoldState,
        output_path: Optional[str] = None,
        show: bool = True,
    ):
        """Render manifold state to figure."""

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 1. Draw regime centers
        if len(state.regime_centers) > 0:
            ax.scatter(
                state.regime_centers[:, 0],
                state.regime_centers[:, 1],
                state.regime_centers[:, 2],
                marker='s', s=200, c='gray', alpha=0.4,
                label='Regime Centers'
            )

        # 2. Draw attractors
        ax.scatter(
            *state.healthy_attractor,
            marker='o', s=400, c='green',
            edgecolors='black', linewidths=2,
            label='Healthy Attractor'
        )
        ax.scatter(
            *state.failure_attractor,
            marker='X', s=400, c='red',
            edgecolors='black', linewidths=2,
            label='Failure Attractor'
        )

        # 3. Draw entities
        for eid, entity in state.entities.items():
            ax.scatter(
                entity.position[0],
                entity.position[1],
                entity.position[2],
                c=[entity.color], s=entity.size * 200,
                alpha=0.8, edgecolors='black', linewidths=0.5
            )

            # Label critical entities
            if entity.regime == 2 and self.config.show_labels:
                ax.text(
                    entity.position[0],
                    entity.position[1],
                    entity.position[2] + 0.5,
                    f"{eid}\n({entity.hd_slope:.3f})",
                    fontsize=8, ha='center', color='red'
                )

        # 4. Draw velocity vectors
        if self.config.show_velocities:
            for entity in state.entities.values():
                if np.linalg.norm(entity.velocity) > 0.01:
                    ax.quiver(
                        entity.position[0],
                        entity.position[1],
                        entity.position[2],
                        entity.velocity[0],
                        entity.velocity[1],
                        entity.velocity[2],
                        color='blue', alpha=0.6,
                        arrow_length_ratio=0.3
                    )

        # 5. Draw force vectors
        if self.config.show_forces:
            for entity in state.entities.values():
                if np.linalg.norm(entity.force) > 0.01:
                    ax.quiver(
                        entity.position[0],
                        entity.position[1],
                        entity.position[2],
                        entity.force[0] * 0.5,
                        entity.force[1] * 0.5,
                        entity.force[2] * 0.5,
                        color='orange', alpha=0.4,
                        arrow_length_ratio=0.2
                    )

        # Styling
        ax.set_xlabel('PC1 (Primary Behavioral Axis)')
        ax.set_ylabel('PC2 (Secondary Axis)')
        ax.set_zlabel('PC3 (Tertiary Axis)')

        ax.set_title(
            f'Behavioral Manifold\n'
            f'Healthy: {state.n_healthy} | '
            f'Degraded: {state.n_degraded} | '
            f'Critical: {state.n_critical}',
            fontsize=14
        )

        ax.legend(loc='upper left')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")

        if show and not output_path:
            plt.show()

        return fig

    def render_2d_projection(
        self,
        state: ManifoldState,
        axes: tuple = (0, 1),
        output_path: Optional[str] = None,
    ):
        """Render 2D projection of manifold."""

        fig, ax = plt.subplots(figsize=(12, 10))

        ax_labels = ['PC1', 'PC2', 'PC3']
        i, j = axes

        # Draw entities
        for eid, entity in state.entities.items():
            ax.scatter(
                entity.position[i],
                entity.position[j],
                c=[entity.color], s=entity.size * 200,
                alpha=0.8, edgecolors='black', linewidths=0.5
            )

            # Label critical
            if entity.regime == 2:
                ax.annotate(
                    f"{eid}\n({entity.hd_slope:.3f})",
                    (entity.position[i], entity.position[j]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    color='red'
                )

        # Draw attractors
        ax.scatter(
            state.healthy_attractor[i],
            state.healthy_attractor[j],
            marker='o', s=300, c='green',
            edgecolors='black', linewidths=2,
            label='Healthy', zorder=10
        )
        ax.scatter(
            state.failure_attractor[i],
            state.failure_attractor[j],
            marker='X', s=300, c='red',
            edgecolors='black', linewidths=2,
            label='Failure', zorder=10
        )

        ax.set_xlabel(ax_labels[i])
        ax.set_ylabel(ax_labels[j])
        ax.set_title(
            f'Manifold Projection ({ax_labels[i]} vs {ax_labels[j]})\n'
            f'Healthy: {state.n_healthy} | '
            f'Degraded: {state.n_degraded} | '
            f'Critical: {state.n_critical}'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        else:
            plt.show()

        return fig
