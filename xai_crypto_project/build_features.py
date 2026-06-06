"""
Phase 2 — Feature Extraction (standalone script)
Run this instead of the Phase 2 notebook cell.
Reads data/dataset.csv, extracts features (including seq_len), writes data/features.csv.
"""
import ast
import pandas as pd
from feature_extraction import extract_features

raw_df = pd.read_csv("data/dataset.csv")

records = [
    {
        **extract_features(ast.literal_eval(row.bytes), seq_len=int(row.seq_len)),
        "label": row.label,
    }
    for row in raw_df.itertuples(index=False)
]

feature_df = pd.DataFrame(records)
feature_df.to_csv("data/features.csv", index=False)
print(feature_df.head())
print(feature_df[["seq_len", "label"]].groupby("label").agg(["mean", "min", "max"]))
print(feature_df["label"].value_counts())
