#!/usr/bin/env python3
"""
Process extra method directories with the loader of 00_Load_and_Process.qmd
and append the rows to an existing processed results file (.pkl and .csv).

The loader's definition chunks (up to and including "load-all-data") are
executed as they are, so the new rows have exactly the notebook's schema.
The previous .pkl and .csv are kept with a ".bak-<timestamp>" suffix.

Usage (from the notebooks directory):
  ../scripts/append_processed_results.py results/all graph_symmetry_results_all \
      ManifoldCG ManifoldTR

--output-dir writes to a copy instead (the existing file is read from there too).
--replace drops the existing rows of the methods first, but only for the graph
types present in the new results, so rows of other graph types are kept.
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd

LOADER = Path("00_Load_and_Process.qmd")
LAST_DEFINITION_CHUNK = "load-all-data"


def loader_namespace():
    text = LOADER.read_text()
    chunks = re.findall(r"^```\{python\}\n(.*?)^```", text, flags=re.S | re.M)
    namespace = {"__name__": "loader"}
    for chunk in chunks:
        exec(compile(chunk, str(LOADER), "exec"), namespace)
        if re.search(rf"label:\s*{LAST_DEFINITION_CHUNK}\b", chunk):
            return namespace
    raise RuntimeError(f"chunk {LAST_DEFINITION_CHUNK!r} not found in {LOADER}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_name")
    parser.add_argument("methods", nargs="+")
    parser.add_argument("--output-dir", type=Path, help="default: the loader's OUTPUT_DIR")
    parser.add_argument("--replace", action="store_true",
                        help="replace the methods' rows for the graph types in results_dir")
    args = parser.parse_args()
    results_dir, output_name, methods = args.results_dir, args.output_name, args.methods
    ns = loader_namespace()
    ns["METHOD_DIRS"] = methods
    output_dir = args.output_dir or ns["OUTPUT_DIR"]
    pkl, csv = output_dir / f"{output_name}.pkl", output_dir / f"{output_name}.csv"

    existing = pd.read_pickle(pkl)
    clash = set(existing.method.unique()) & set(methods)
    if clash and not args.replace:
        raise SystemExit(f"{pkl} already contains {sorted(clash)}; not appending twice")

    new = ns["load_all_data"](results_dir, ns["DATA_DIR"])
    if new.empty:
        raise SystemExit("no rows loaded")
    if args.replace:
        replaced = existing.method.isin(methods) & existing.graph_type.isin(new.graph_type.unique())
        print(f"replacing {replaced.sum()} rows of {sorted(clash)} for {sorted(new.graph_type.unique())}")
        existing = existing[~replaced]
    combined = pd.concat([existing, new], ignore_index=True)
    # Keep the existing column types, e.g. a column of None in the new rows
    # would otherwise turn a float column into object
    for col, dtype in existing.dtypes.items():
        if combined[col].dtype != dtype:
            combined[col] = combined[col].astype(dtype)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    for path in (pkl, csv):
        if path.exists():
            path.rename(path.with_name(f"{path.name}.bak-{stamp}"))
    combined.to_pickle(pkl)
    combined.to_csv(csv, index=False)
    print(f"{'replaced with' if args.replace else 'appended'} {len(new)} rows ({sorted(new.method.unique())}) -> {len(combined)} rows in {pkl} and {csv}")


if __name__ == "__main__":
    main()
