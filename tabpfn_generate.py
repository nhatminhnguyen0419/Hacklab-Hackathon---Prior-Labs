"""
Synthetic data generation with TabPFN-3.5 (tabpfn-extensions, unsupervised model).

Learns the joint distribution of the per-episode table (5 shape features + object
label) from real simulator episodes and samples new rows from it.
Fidelity check: "train on synthetic, test on real" - a classifier trained ONLY on
the generated rows is scored on real episodes it never saw.

  python tabpfn_generate.py --n 500        # writes data/tabpfn_synthetic.csv
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

from classifier import DATA, FEATURES
from common import OBJECTS

CONTEXT = 500  # real rows TabPFN learns from; more is slower on CPU (see classifier.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="synthetic rows to generate")
    ap.add_argument("--out", default="data/tabpfn_synthetic.csv")
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    df["label"] = df["object_type"].map(OBJECTS.index)
    real_tr, real_te = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=0)
    ctx = real_tr.sample(CONTEXT, random_state=0)

    cols = FEATURES + ["label"]
    fast = ModelVersion.V3_5_FAST  # CPU-friendly; use ModelVersion.V3_5 on the GPU laptop
    model = TabPFNUnsupervisedModel(
        tabpfn_clf=TabPFNClassifier.create_default_for_version(fast, device="auto"),
        tabpfn_reg=TabPFNRegressor.create_default_for_version(fast, device="auto"),
    )
    model.set_categorical_features([cols.index("label")])
    model.fit(torch.tensor(ctx[cols].to_numpy(), dtype=torch.float32))
    syn = pd.DataFrame(model.generate_synthetic_data(n_samples=args.n).cpu().numpy(), columns=cols)

    syn["label"] = syn["label"].round().clip(0, len(OBJECTS) - 1).astype(int)
    syn["object_type"] = syn["label"].map(dict(enumerate(OBJECTS)))
    syn[FEATURES + ["object_type"]].to_csv(args.out, index=False)
    print(f"Wrote {len(syn)} synthetic rows to {args.out}")

    # Fidelity: per-object feature means, synthetic vs real
    print("\nmean features per object (real / synthetic):")
    for o in OBJECTS:
        r = real_tr[real_tr.object_type == o][FEATURES].mean().to_numpy()
        s = syn[syn.object_type == o][FEATURES].mean().to_numpy()
        print(f"  {o:7s} n_syn={int((syn.object_type == o).sum()):4d}  "
              + "  ".join(f"{a:.3f}/{b:.3f}" for a, b in zip(r, s)))

    # Train on synthetic, test on real (vs train on the same amount of real)
    def score(train):
        rf = RandomForestClassifier(200, random_state=0).fit(train[FEATURES], train["object_type"])
        return rf.score(real_te[FEATURES], real_te["object_type"])
    tstr, trtr = score(syn), score(real_tr.sample(len(syn), random_state=1))
    print(f"\naccuracy on {len(real_te)} real test episodes: trained on synthetic {tstr:.3f}"
          f"   trained on {len(syn)} real {trtr:.3f}")
    assert np.isfinite(syn[FEATURES].to_numpy()).all(), "generator produced NaN/inf"


if __name__ == "__main__":
    main()
