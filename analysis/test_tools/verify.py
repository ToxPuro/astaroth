#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
import argparse

#We filter shock out since it can have very large values that with smoothing produce numerical noise that can be zero or a small value that is nonzero
def filter_shock(df):
    cols_to_keep = [c for c in df.columns if "SHOCK" not in c]
    return df[cols_to_keep]

def test(reference_path, result_path, tol=1e-8):
    reference = filter_shock(pd.read_csv(reference_path, sep=r"\s+"))
    res = filter_shock(pd.read_csv(result_path, sep=r"\s+"))

    # numeric-only to avoid string subtraction errors
    reference = reference.select_dtypes(include="number")
    res = res.select_dtypes(include="number")

    # compute relative difference safely
    eps = 1e-15
    denominator = np.maximum(reference.abs(), res.abs()).replace(0, eps)
    rel_diff = (reference - res).abs() / denominator

    # maximum relative difference
    max_val = rel_diff.to_numpy().max()

    # find location (row, col) of maximum difference
    max_loc = np.unravel_index(rel_diff.to_numpy().argmax(), rel_diff.shape)
    row, col = max_loc
    col_name = rel_diff.columns[col]
    row_idx = rel_diff.index[row]

    print(f"Maximum relative difference: {max_val}")
    print(f"At row {row_idx}, column '{col_name}'")
    print(f"Reference value: {reference.iloc[row, col]}, Result value: {res.iloc[row, col]}")

    if max_val > tol:
        print("Did not match reference data for:", reference_path)
        print("Maximum relative error was:", max_val)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare timeseries CSV files")
    parser.add_argument("reference", help="Path to reference timeseries CSV")
    parser.add_argument("result", help="Path to result timeseries CSV")
    parser.add_argument("--tol", type=float, default=1e-8,
                        help="Tolerance for maximum relative difference (default=1e-8)")

    args = parser.parse_args()
    test(args.reference, args.result, args.tol)

