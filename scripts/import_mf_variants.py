#!/usr/bin/env python3
"""
Import Manifold CG / trust-region results into the notebooks' results tree.

run.py stores every Manifold optimizer under the method directory "Manifold".
The analysis notebooks identify a method by the "method" field of config.json
and filter on method == "Manifold", so the variants would be pooled with the
steepest-descent results. This script copies each configuration directory to
a separate method directory (ManifoldCG, ManifoldTR) with the same layout,
sets "method" in config.json accordingly, and flattens the nested solver
options into plain method_config keys (so they load as scalar columns).

Usage: import_mf_variants.py SRC_RESULTS DEST_RESULTS
  e.g. import_mf_variants.py results/mf-full notebooks/results/all
"""

import json
import shutil
import sys
from pathlib import Path

METHOD_FOR_OPTIMIZER = {
    "conjugate_gradient": "ManifoldCG",
    "trust_regions": "ManifoldTR",
}
NESTED_OPTIONS = ("optimizer_kwargs", "run_kwargs", "manifold_kwargs")


def flatten_method_config(method_config):
    flat = {k: v for k, v in method_config.items() if k not in NESTED_OPTIONS}
    for key in NESTED_OPTIONS:
        for name, value in method_config.get(key, {}).items():
            if name in flat:
                raise ValueError(f"option {name} clashes with an existing method_config key")
            flat[name] = value
    return flat


def main():
    src, dest = Path(sys.argv[1]) / "Manifold", Path(sys.argv[2])
    counts = {}
    for cdir in sorted(d for d in src.iterdir() if d.is_dir()):
        config = json.load(open(cdir / "config.json"))
        method = METHOD_FOR_OPTIMIZER.get(config["method_config"]["optimizer"])
        if method is None:
            print(f"skipping {cdir.name}: optimizer {config['method_config']['optimizer']}")
            continue
        target = dest / method / cdir.name
        if target.exists():
            raise FileExistsError(f"{target} already exists; not overwriting")
        shutil.copytree(cdir, target)
        config["method"] = method
        config["method_config"] = flatten_method_config(config["method_config"])
        with open(target / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        counts[method] = counts.get(method, 0) + 1
    print("imported configuration directories:", counts)


if __name__ == "__main__":
    main()
