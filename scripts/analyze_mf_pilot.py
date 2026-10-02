#!/usr/bin/env python3
"""
Compare Manifold (MF) solver variants against the paper's steepest-descent MF.

Loads results produced by run.py for the CG / trust-region variants, joins them
with the per-run steepest-descent results in the processed results pickle, and
applies the paper's per-graph-type selection of the penalty c:

  - a simulation is a success if any of its runs is a non-identity permutation,
  - only c values with success rate >= 0.99 over the graph type are eligible,
  - the eligible c with the lowest mean symmetry over all runs is selected.

Variants are then compared on the best non-identity symmetry per simulation at
their selected c. Steepest descent uses the c selected in the paper.

Usage: analyze_mf_pilot.py RESULTS_DIR [OUTPUT_DIR]
"""

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "kubek"
SD_PICKLE = REPO / "notebooks" / "processed_data" / "graph_symmetry_results_all.pkl"

# c selected for MF (steepest descent) in the paper, per graph type
PAPER_SD_C = {"ER": 0.1, "BA": 0.4, "GEO": 0.1, "GEO3D": 0.1, "DD": 0.1, "HK": 0.4, "LRM_ER": 0.1}

NAME = re.compile(
    r"(?P<gt>.+?)_nNodes(?P<n>\d+)_density(?P<d>\d+)(?:.*?_rew(?P<rew>\d+))?.*_sim(?P<sim>\d+)$"
)
GT_DIR_TO_PKL = {"LRM_ER_rewired": "LRM_ER"}
KEY = ["graph_type", "n_nodes", "density", "rew", "sim_id"]


def variant_name(cfg):
    m = cfg["method_config"]
    if m["optimizer"] == "conjugate_gradient":
        rule = m.get("optimizer_kwargs", {}).get("beta_rule", "HestenesStiefel")
        return "CG-" + "".join(ch for ch in rule if ch.isupper())
    if m["optimizer"] == "trust_regions":
        return f"TR-{m.get('run_kwargs', {}).get('Delta_bar', 'default'):g}"
    return m["optimizer"]


def load_variants(results_dir):
    graphs = {}
    rows = []
    for cdir in sorted((results_dir / "Manifold").iterdir()):
        cfg = json.load(open(cdir / "config.json"))
        variant = variant_name(cfg)
        for gdir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            for pf in sorted((gdir / "Permutations").glob("*.csv")):
                match = NAME.match(pf.stem)
                sim = int(match["sim"])
                instance_file = DATA / gdir.name / (pf.stem.rsplit("_sim", 1)[0] + ".p")
                if instance_file not in graphs:
                    with open(instance_file, "rb") as f:
                        graphs[instance_file] = pickle.load(f)
                A = graphs[instance_file][sim].astype(float)
                n = A.shape[0]
                perms = np.loadtxt(pf, delimiter=",", dtype=int).reshape(n, -1)
                metrics = pd.read_csv(gdir / "Metrics" / pf.name)
                for run in range(perms.shape[1]):
                    p = perms[:, run]
                    # (P A P^T)_ij = A[p_i, p_j] for P_{i, p_i} = 1
                    error = 0.25 * np.sum((A - A[np.ix_(p, p)]) ** 2)
                    rows.append(
                        dict(
                            variant=variant,
                            c=cfg["c"],
                            graph_type=GT_DIR_TO_PKL.get(gdir.name, gdir.name),
                            n_nodes=n,
                            density=int(match["d"]),
                            rew=float(match["rew"]) if match["rew"] else np.nan,
                            sim_id=sim,
                            run_id=run,
                            symmetry=4 * error / (n * (n - 1)),
                            fixed_points=int(np.sum(p == np.arange(n))),
                            time=metrics.time[run],
                            iterations=metrics.iterations[run],
                            stop=metrics.stopping_criterion[run].split(" after")[0],
                        )
                    )
    return pd.DataFrame(rows)


def load_steepest_descent(sims, columns):
    sd = pd.read_pickle(SD_PICKLE)
    sd = sd[(sd.method == "Manifold") & sd.sim_id.isin(sims)].copy()
    sd = sd.rename(columns={"metric_time": "time", "metric_iterations": "iterations"})
    sd["variant"] = "SD"
    return sd[[c for c in columns if c in sd.columns]]


def select_c(df):
    df = df.assign(nonid=df.fixed_points != df.n_nodes)
    success = df.groupby(["variant", "c"] + KEY, dropna=False).nonid.any().rename("success").reset_index()
    rows = []
    for (variant, gt), g in df.groupby(["variant", "graph_type"]):
        rate = success[(success.variant == variant) & (success.graph_type == gt)].groupby("c").success.mean()
        eligible = rate[rate >= 0.99].index
        if variant == "SD":
            c = PAPER_SD_C[gt]
        elif len(eligible):
            c = g[g.c.isin(eligible)].groupby("c").symmetry.mean().idxmin()
        else:
            c = np.nan
        rows.append(dict(variant=variant, graph_type=gt, c=c, success=rate.get(c, np.nan)))
    return pd.DataFrame(rows)


def main():
    results_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else results_dir
    pd.set_option("display.width", 200)

    new = load_variants(results_dir)
    sims = sorted(new.sim_id.unique())
    df = pd.concat([load_steepest_descent(sims, new.columns), new], ignore_index=True)
    df["c"] = df.c.round(3)
    df["nonid"] = df.fixed_points != df.n_nodes
    print(f"{len(new)} new runs; variants {sorted(new.variant.unique())}; sims {sims}")
    print("runs per (variant, c):", new.groupby(["variant", "c"]).size().agg(["min", "max"]).to_dict())

    sel = select_c(df)
    print("\n== selected c per graph type (SD: the paper's choice)")
    print(sel.pivot(index="graph_type", columns="variant", values="c"))
    print("\n== success rate at the selected c")
    print(sel.pivot(index="graph_type", columns="variant", values="success").round(3))

    best = (
        df[df.nonid]
        .merge(sel[["variant", "graph_type", "c"]], on=["variant", "graph_type", "c"])
        .groupby(["variant"] + KEY, dropna=False)
        .symmetry.min()
        .rename("best")
        .reset_index()
    )
    print("\n== mean best-of-runs symmetry at the selected c (lower is better)")
    print(best.pivot_table(index="graph_type", columns="variant", values="best", aggfunc="mean").round(4))
    print(best.pivot_table(index="n_nodes", columns="variant", values="best", aggfunc="mean").round(4))

    wide = best.pivot_table(index=KEY, columns="variant", values="best", dropna=False)
    rows = []
    for variant in [v for v in wide.columns if v != "SD"]:
        groups = list(wide.groupby(level="graph_type")) + [("ALL", wide)]
        for gt, g in groups:
            x = g[[variant, "SD"]].dropna()
            d = x[variant] - x.SD
            p = wilcoxon(d).pvalue if (d != 0).sum() >= 6 else np.nan
            rows.append(
                dict(variant=variant, graph_type=gt, n=len(x), win=(d < -1e-12).mean(),
                     loss=(d > 1e-12).mean(), mean_diff=d.mean(), p=p)
            )
    print("\n== paired with SD per instance (both at selected c): win / loss / mean diff / Wilcoxon p")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n== median time per run (s), all c")
    print(df.pivot_table(index="n_nodes", columns="variant", values="time", aggfunc="median").round(2))
    print("\n== median iterations, all c")
    print(df.pivot_table(index="n_nodes", columns="variant", values="iterations", aggfunc="median"))
    print("\n== stopping criteria (new variants)")
    print(new.groupby(["variant", "stop"]).size().unstack(0).fillna(0).astype(int))
    print("\n== identity-run rate by c")
    grid = sorted(new.c.round(3).unique())
    print(df[df.c.isin(grid)].assign(identity=lambda x: ~x.nonid)
          .pivot_table(index="c", columns="variant", values="identity", aggfunc="mean").round(3))

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_dir / "mf_variants_runs.pkl")
    best.to_csv(out_dir / "mf_variants_best.csv", index=False)
    sel.to_csv(out_dir / "mf_variants_selected_c.csv", index=False)


if __name__ == "__main__":
    main()
