#!/usr/bin/env python3
"""
Draft figures comparing the MF solver variants (steepest descent, conjugate
gradient, trust regions), from the output of analyze_mf_pilot.py.

Usage: plot_mf_variants.py ANALYSIS_DIR [OUTPUT_DIR] [--variants SD,CG-HS,TR-0.3]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (registers the 'science' style)

# Same style as notebooks/04_Final.qmd
plt.style.use(["science"])
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 18,
    "axes.titlesize": 22,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 18,
    "figure.titlesize": 24,
    "text.usetex": True,
    "text.latex.preamble": "\\usepackage{libertinus}",
    "pdf.fonttype": 42,
})

# Colours as METHOD_COLORS in notebooks/04_Final.qmd (seaborn "deep" 1, 5, 6)
STYLE = {
    "SD": dict(label="MF-SD", color="#dd8452", marker="o"),
    "CG-HS": dict(label="MF-CG", color="#937860", marker="s"),
    "TR-0.3": dict(label="MF-TR", color="#da8bc3", marker="^"),
}
# Vertical offsets (points) for end labels of the time plot, where SD and CG nearly coincide
LABEL_NUDGE = {"SD": 7, "CG-HS": -7}
GRAPH_ORDER = ["ER", "BA", "GEO", "GEO3D", "DD", "HK", "LRM_ER"]
GRAPH_LABEL = {"GEO": "GEO2D", "LRM_ER": "LRM"}
KEY = ["graph_type", "n_nodes", "density", "rew", "sim_id"]


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return x.mean(), half


def legend_below(fig, axes, variants):
    handles = [plt.Line2D([], [], color=STYLE[v]["color"], marker=STYLE[v]["marker"], lw=2, ms=8)
               for v in variants]
    fig.legend(handles, [STYLE[v]["label"] for v in variants], loc="lower center",
               ncol=len(variants), frameon=False, bbox_to_anchor=(0.5, -0.02))


def plot_best_symmetry(best, variants, out):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    axes = axes.ravel()
    offsets = np.linspace(-0.12, 0.12, len(variants))
    sizes = sorted(best.n_nodes.unique())
    for ax, gt in zip(axes, GRAPH_ORDER):
        g = best[best.graph_type == gt]
        for off, v in zip(offsets, variants):
            stats = [mean_ci(g[(g.variant == v) & (g.n_nodes == n)].best) for n in sizes]
            m, h = np.array(stats).T
            x = np.arange(len(sizes)) + off
            ax.errorbar(x, m, yerr=h, color=STYLE[v]["color"], marker=STYLE[v]["marker"],
                        ms=8, lw=2, capsize=3)
        ax.set_title(GRAPH_LABEL.get(gt, gt), fontsize=18)
        ax.set_xticks(range(len(sizes)), [str(n) for n in sizes])
        ax.grid(axis="y", color="0.9", lw=0.8)
    axes[-1].axis("off")
    for ax in axes[4:7]:
        ax.set_xlabel("$n$")
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("best symmetry $S$")
    legend_below(fig, axes, variants)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def plot_paired_difference(best, variants, out):
    wide = best.pivot_table(index=KEY, columns="variant", values="best", dropna=False).reset_index()
    others = [v for v in variants if v != "SD"]
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.32
    offsets = np.linspace(-width / 2, width / 2, len(others)) if len(others) > 1 else [0]
    for off, v in zip(offsets, others):
        data = [(wide[wide.graph_type == gt][v] - wide[wide.graph_type == gt]["SD"]).dropna()
                for gt in GRAPH_ORDER]
        pos = np.arange(len(GRAPH_ORDER)) + off
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True,
                        showfliers=False, medianprops=dict(color="white", lw=2))
        for box in bp["boxes"]:
            box.set(facecolor=STYLE[v]["color"], edgecolor=STYLE[v]["color"])
        for part in ("whiskers", "caps"):
            for line in bp[part]:
                line.set(color=STYLE[v]["color"], lw=1.5)
    ax.axhline(0, color="0.3", lw=1, ls="--")
    ax.set_xticks(range(len(GRAPH_ORDER)), [GRAPH_LABEL.get(g, g) for g in GRAPH_ORDER])
    ax.set_ylabel(r"$S_{\mathrm{variant}} - S_{\mathrm{SD}}$")
    ax.grid(axis="y", color="0.9", lw=0.8)
    # win/loss share per box, as text under the axis labels' baseline
    ymin = ax.get_ylim()[0]
    for off, v in zip(offsets, others):
        for i, gt in enumerate(GRAPH_ORDER):
            d = (wide[wide.graph_type == gt][v] - wide[wide.graph_type == gt]["SD"]).dropna()
            ax.text(i + off, ymin, f"{100 * (d < 0).mean():.0f}\\%", ha="center", va="bottom",
                    fontsize=11, color="0.25")
    handles = [plt.Rectangle((0, 0), 1, 1, color=STYLE[v]["color"]) for v in others]
    ax.legend(handles, [STYLE[v]["label"] for v in others], loc="upper right", frameon=False, fontsize=14)
    ax.set_title("Paired difference to MF-SD per instance (below 0: better); \\% = share of wins",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_time(runs, variants, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = sorted(runs.n_nodes.unique())
    for v in variants:
        g = runs[runs.variant == v].groupby("n_nodes").time
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax.plot(sizes, med[sizes], color=STYLE[v]["color"], marker=STYLE[v]["marker"], ms=8, lw=2)
        ax.fill_between(sizes, q1[sizes], q3[sizes], color=STYLE[v]["color"], alpha=0.15, lw=0)
        ax.annotate(STYLE[v]["label"], (sizes[-1], med[sizes[-1]]), xytext=(8, LABEL_NUDGE.get(v, 0)),
                    textcoords="offset points", va="center", fontsize=13, color="0.2")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xlim(sizes[0] - 5, sizes[-1] + 22)
    ax.set_xlabel("$n$")
    ax.set_ylabel("time per run [s]")
    ax.grid(axis="y", which="major", color="0.9", lw=0.8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("output_dir", type=Path, nargs="?")
    parser.add_argument("--variants", default="SD,CG-HS,TR-0.3")
    args = parser.parse_args()
    out = args.output_dir or args.analysis_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    variants = args.variants.split(",")

    best = pd.read_csv(args.analysis_dir / "mf_variants_best.csv")
    best = best[best.variant.isin(variants)]
    runs = pd.read_pickle(args.analysis_dir / "mf_variants_runs.pkl")
    runs = runs[runs.variant.isin(variants)]

    plot_best_symmetry(best, variants, out / "mf_variants_best_symmetry.pdf")
    plot_paired_difference(best, variants, out / "mf_variants_paired_difference.pdf")
    plot_time(runs, variants, out / "mf_variants_time.pdf")
    print("wrote", sorted(p.name for p in out.glob("mf_variants_*.pdf")))


if __name__ == "__main__":
    main()
