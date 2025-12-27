from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _add_box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    bullets: list[str],
    face: str,
    edge: str,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=2.2,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(box)

    ax.text(
        x + 0.22,
        y + h - 0.34,
        title,
        fontsize=14,
        fontweight="bold",
        color="#0B1220",
        va="top",
    )
    ax.text(
        x + 0.22,
        y + h - 0.72,
        subtitle,
        fontsize=10.5,
        color="#334155",
        va="top",
    )

    bullet_text = "\n".join([f"• {b}" for b in bullets])
    ax.text(
        x + 0.22,
        y + h - 1.18,
        bullet_text,
        fontsize=11,
        color="#0B1220",
        va="top",
        linespacing=1.35,
    )


def _add_arrow(
    ax: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    color: str = "#0B1220",
    linestyle: str = "-",
    connectionstyle: str | None = None,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.0,
        linestyle=linestyle,
        color=color,
        shrinkA=6,
        shrinkB=6,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)

    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + (0.32 if (connectionstyle is None) else 0.45),
            label,
            fontsize=10.5,
            color="#0B1220",
            ha="center",
            va="bottom",
        )


def main() -> None:
    out_path = Path("results/report/figures/architecture_pipeline.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(17, 5.4), dpi=220, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    fig.text(
        0.5,
        0.94,
        "End-to-End Privacy Benchmarking Pipeline (Federated Learning)",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#0B1220",
    )
    fig.text(
        0.5,
        0.90,
        "FL simulation → captured updates → optional defenses → gradient inversion attack → metrics & stakeholder-ready artifacts",
        ha="center",
        va="top",
        fontsize=11.5,
        color="#334155",
    )

    y = 1.2
    w = 3.55
    h = 2.75

    x1 = 0.6
    x2 = x1 + w + 0.6
    x3 = x2 + w + 0.6
    x4 = x3 + w + 0.6

    _add_box(
        ax,
        x=x1,
        y=y,
        w=w,
        h=h,
        title="Federated Learning Simulation",
        subtitle="fl_system.py",
        bullets=[
            "data loading + transforms",
            "client training loop (FedAvg)",
            "capture gradients / updates",
        ],
        face="#EFF6FF",
        edge="#2563EB",
    )

    _add_box(
        ax,
        x=x2,
        y=y,
        w=w,
        h=h,
        title="Privacy Layer (Optional)",
        subtitle="differential_privacy.py · homomorphic_encryptor.py",
        bullets=[
            "DP: clipping + Gaussian noise (ε,δ)",
            "HE: additive prototype + fast simulation",
            "toggleable per experiment run",
        ],
        face="#FFF7ED",
        edge="#F97316",
    )

    _add_box(
        ax,
        x=x3,
        y=y,
        w=w,
        h=h,
        title="Gradient Inversion Attack",
        subtitle="gradient_attack.py",
        bullets=[
            "DLG / iDLG optimization",
            "TV regularization + restarts",
            "metric matching (L2 / cosine)",
        ],
        face="#F5F3FF",
        edge="#7C3AED",
    )

    _add_box(
        ax,
        x=x4,
        y=y,
        w=w,
        h=h,
        title="Artifacts & Reporting",
        subtitle="scripts/* · results/report/*",
        bullets=[
            "baseline_attack_result.png",
            "metrics.txt + config.json",
            "summary.csv + dashboard + poster",
        ],
        face="#ECFDF5",
        edge="#10B981",
    )

    mid_y = y + h / 2

    _add_arrow(
        ax,
        start=(x1 + w, mid_y),
        end=(x2, mid_y),
        label="captured gradients / updates",
        color="#0B1220",
    )
    _add_arrow(
        ax,
        start=(x2 + w, mid_y),
        end=(x3, mid_y),
        label="defended signal",
        color="#0B1220",
    )
    _add_arrow(
        ax,
        start=(x3 + w, mid_y),
        end=(x4, mid_y),
        label="reconstruct + score",
        color="#0B1220",
    )

    _add_arrow(
        ax,
        start=(x1 + w, y + h - 0.25),
        end=(x3, y + h - 0.25),
        label="baseline (no defense)",
        color="#334155",
        linestyle=(0, (5, 4)),
        connectionstyle="arc3,rad=0.18",
    )

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

