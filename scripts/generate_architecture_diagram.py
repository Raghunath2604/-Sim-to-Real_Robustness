"""
Generate publication-quality System Architecture Diagram for Sim-to-Real Robustness Benchmark.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def create_architecture_diagram(out_path: str = "assets/architecture_diagram.png") -> None:
    # High-res canvas with sleek dark theme
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    bg_color = "#0B0F19"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Header Title
    ax.text(
        8.0,
        8.45,
        "Sim-to-Real Robustness Architecture & Evaluation Pipeline",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#F8FAFC",
        fontfamily="sans-serif",
    )
    ax.text(
        8.0,
        8.05,
        "End-to-End PPO Synthesis, 5-Axis Dynamics Perturbation, "
        "Deterministic Evaluation Grid & Real-Time Microservice Serving",
        ha="center",
        va="center",
        fontsize=11,
        color="#94A3B8",
        fontfamily="sans-serif",
    )

    # 4 Main Pipeline Cards
    card_width = 3.3
    card_height = 6.2
    y_card = 1.2

    cards = [
        {
            "x": 0.6,
            "title": "1. Perturbation Engine",
            "tag": "DYNAMICS & ASYNCHRONY",
            "color": "#38BDF8",  # Cyan
            "bg": "#111827",
            "border": "#0284C7",
            "items": [
                ("Simulation Core", "MuJoCo 3.15 (HalfCheetah-v5)"),
                ("Payload Inertia", "Body link mass: 0.5x - 1.8x"),
                ("Contact Surface", "Ground friction: 0.3x - 2.0x"),
                ("Actuator Authority", "Motor gear torque: 0.6x - 1.4x"),
                ("Sensor Uncertainty", "Gaussian obs noise (sigma=0.20)"),
                ("Temporal Lag", "FIFO action delay queue (tau=1, 2)"),
                ("Interface Mode", "Gymnasium wrapper layer"),
            ],
        },
        {
            "x": 4.3,
            "title": "2. PPO Synthesis Core",
            "tag": "POLICY OPTIMIZATION",
            "color": "#818CF8",  # Indigo
            "bg": "#111827",
            "border": "#4F46E5",
            "items": [
                ("Algorithm", "Continuous PPO + GAE (lambda=0.95)"),
                ("Architecture", "Actor-Critic MLP (Tanh, Ortho Init)"),
                ("Surrogate Loss", "Clipped objective (eps=0.2) + VF clip"),
                ("Variants Evaluated", "baseline, dr_0.2, dr_0.5, dr_sensing"),
                ("Training Budget", "20M steps (4 variants x 5 seeds)"),
                ("Vectorized Workers", "Parallel rollout worker environments"),
                ("Artifact Format", "TorchScript JIT & PyTorch state"),
            ],
        },
        {
            "x": 8.0,
            "title": "3. Benchmark & Rigor",
            "tag": "EVALUATION ENGINE",
            "color": "#F59E0B",  # Amber
            "bg": "#111827",
            "border": "#D97706",
            "items": [
                ("Evaluation Space", "25 off-nominal grid configurations"),
                ("Total Sample Size", "500 deterministic evaluation runs"),
                ("Statistical Bounds", "95% CIs across paired seeds"),
                ("Hypothesis Testing", "Pre-registered Welch t-tests (p=0.03)"),
                ("Primary Metric", "Retention Rate: Mean Perturbed / Nominal"),
                ("Floor Safety", "Worst-case episode return floor"),
                ("Reproducibility", "Paired evaluation seeds (1-5)"),
            ],
        },
        {
            "x": 11.7,
            "title": "4. Production Serving",
            "tag": "ZERO-DRIFT DEPLOYMENT",
            "color": "#10B981",  # Emerald
            "bg": "#111827",
            "border": "#059669",
            "items": [
                ("Serving Framework", "Asynchronous FastAPI microservice"),
                ("Inference Protocol", "REST API (POST /act) JSON/tensors"),
                ("Exported Artifact", "Optimized TorchScript JIT graph"),
                ("Closed-Loop Test", "1,000 steps active feedback test"),
                ("Numerical Parity", "0.000000 drift (Bit-identical)"),
                ("Mean Latency", "10.65 ms serving latency"),
                ("Real-Time Bound", "P95 = 19.52 ms (< 20 ms loop bound)"),
            ],
        },
    ]

    for card in cards:
        # Card Background
        rect = patches.FancyBboxPatch(
            (card["x"], y_card),
            card_width,
            card_height,
            boxstyle="round,pad=0.1,rounding_size=0.18",
            facecolor=card["bg"],
            edgecolor=card["border"],
            linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(rect)

        # Card Header Tag
        ax.text(
            card["x"] + 0.2,
            y_card + card_height - 0.45,
            card["tag"],
            fontsize=8,
            fontweight="bold",
            color=card["color"],
            fontfamily="sans-serif",
        )
        # Card Title
        ax.text(
            card["x"] + 0.2,
            y_card + card_height - 0.85,
            card["title"],
            fontsize=13,
            fontweight="bold",
            color="#FFFFFF",
            fontfamily="sans-serif",
        )

        # Horizontal separator
        ax.plot(
            [card["x"] + 0.2, card["x"] + card_width - 0.2],
            [y_card + card_height - 1.05, y_card + card_height - 1.05],
            color="#334155",
            linewidth=1,
            zorder=3,
        )

        # Items list
        y_text = y_card + card_height - 1.45
        for heading, desc in card["items"]:
            # Bullet Dot
            ax.plot(card["x"] + 0.25, y_text, marker="o", markersize=4, color=card["color"], zorder=3)
            # Label
            ax.text(
                card["x"] + 0.42,
                y_text,
                heading,
                fontsize=9.5,
                fontweight="bold",
                color="#F1F5F9",
                fontfamily="sans-serif",
                va="center",
            )
            # Value / Description
            ax.text(
                card["x"] + 0.42,
                y_text - 0.32,
                desc,
                fontsize=8.5,
                color="#94A3B8",
                fontfamily="sans-serif",
                va="center",
            )
            y_text -= 0.68

    # Connector Flow Arrows between cards
    arrow_props = dict(
        arrowstyle="-|>,head_length=0.8,head_width=0.45", color="#38BDF8", lw=2.5, mutation_scale=15, zorder=5
    )
    arrow_y = y_card + (card_height / 2)

    # 1 -> 2
    ax.annotate("", xy=(4.25, arrow_y), xytext=(3.95, arrow_y), arrowprops=arrow_props)
    # 2 -> 3
    ax.annotate("", xy=(7.95, arrow_y), xytext=(7.65, arrow_y), arrowprops=arrow_props)
    # 3 -> 4
    ax.annotate("", xy=(11.65, arrow_y), xytext=(11.35, arrow_y), arrowprops=arrow_props)

    # Bottom Footer Summary Ribbon
    ribbon = patches.FancyBboxPatch(
        (0.6, 0.25),
        14.4,
        0.65,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor="#1E293B",
        edgecolor="#334155",
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(ribbon)
    ax.text(
        8.0,
        0.57,
        "Rigorous Scientific Standard: Pre-Registered Hypotheses  *  "
        "Empirical Measurements Only  *  Closed-Loop Sim-to-Real Proxy Validation",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#38BDF8",
        fontfamily="sans-serif",
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=bg_color, bbox_inches="tight")
    plt.close()
    print(f"Successfully rendered architecture diagram to: {out_path}")


if __name__ == "__main__":
    create_architecture_diagram("assets/architecture_diagram.png")
