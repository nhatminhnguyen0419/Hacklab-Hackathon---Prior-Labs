"""
Object classifier (spec section 3) with TabPFN-3.5: which of the 5 objects is
approaching, from the 5 noisy shape features. Replaces the rule-based classify().

  python classifier.py                 # train on 80%, report accuracy on 20% vs the rule
  python classifier.py --fast          # same with the fast checkpoint on 500 examples (CPU-friendly)

From other code:
  from classifier import load_classifier
  clf = load_classifier()
  clf.predict([[h, w, d, elongation, roundness]])  # -> array(["bottle"])
"""

import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from common import OBJECTS, classify

FEATURES = ["feat_height", "feat_width", "feat_depth", "feat_elongation", "feat_roundness"]
DATA = "data/run1_features.csv"


FAST_SAMPLES = 500  # CPU timing per prediction: full + 3000 examples ~130 s, fast + 500 ~3 s


def make_model(fast=False):
    version = ModelVersion.V3_5_FAST if fast else ModelVersion.V3_5
    return TabPFNClassifier.create_default_for_version(version, device="auto")


def load_classifier(path=DATA, fast=False):
    """TabPFN learns in-context: 'training' just stores the examples, and the work happens at
    predict time. fast=True (fast checkpoint, 500 examples) for live use without a GPU."""
    df = pd.read_csv(path)
    if fast:
        df = df.sample(FAST_SAMPLES, random_state=0)
    return make_model(fast).fit(df[FEATURES].to_numpy(), df["object_type"].to_numpy())


def main():
    fast = "--fast" in sys.argv
    df = pd.read_csv(DATA)
    X, y = df[FEATURES].to_numpy(), df["object_type"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    if fast:
        X_tr, y_tr = X_tr[:FAST_SAMPLES], y_tr[:FAST_SAMPLES]

    pred = make_model(fast).fit(X_tr, y_tr).predict(X_te)
    rule = [classify(tuple(row)) for row in X_te]

    print(f"TabPFN-3.5{'-fast' if fast else ''} on {len(y_tr)} examples, accuracy: {accuracy_score(y_te, pred):.3f}   rule baseline: {accuracy_score(y_te, rule):.3f}"
          f"   ({len(y_te)} test episodes)")
    print(classification_report(y_te, pred, labels=OBJECTS, digits=3))
    print("confusion (rows = true, cols = predicted):", OBJECTS)
    print(confusion_matrix(y_te, pred, labels=OBJECTS))
    assert accuracy_score(y_te, pred) > 0.9, "classifier much worse than expected - check the data"


if __name__ == "__main__":
    main()
