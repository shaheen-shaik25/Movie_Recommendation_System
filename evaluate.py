"""Evaluate the recommender offline:  python evaluate.py [--k 10] [--sample 1000]"""
import argparse
import time

import pandas as pd

from recommender import Recommender
from recommender.evaluation import evaluate, franchise_groups

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="list length to evaluate (default 10)")
    ap.add_argument("--sample", type=int, default=None, help="evaluate on a random sample of N movies (default: all)")
    a = ap.parse_args()

    rec = Recommender.load_or_build()
    t = time.time()
    table = evaluate(rec, k=a.k, sample=a.sample)
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 200
    print(f"\nEvaluated {a.sample or len(rec)} query movies at k={a.k} in {time.time() - t:.0f}s")
    print(f"Franchise ground truth covers {len(franchise_groups(rec.movies))} movies\n")
    print(table.T.to_string())
    out = f"docs/evaluation_k{a.k}.csv"
    table.to_csv(out)
    print(f"\nSaved -> {out}")
