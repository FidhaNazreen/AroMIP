"""
train_test_W.py  –  Full 20-amino-acid model for Tryptophan (W) on Human IDR data

Dataset   : 9-residue windows centred on W residues, extracted from human IDRs.
            Pre-split into 80 % training and 20 % test sets.

Data files:
    data/W_rows_train.csv  (space-separated, no header)
    data/W_rows_test.csv   (same format)
    col 0 – 9-residue sequence (middle pos 5 = W)
    col 3 – PPM3 membrane insertion depth score

Model     : P(insert | seq) = 1 / (1 + (Q_th / Q)^n)
            Q = product of q[aa] over 8 context positions  (Q_th=1, n=1 fixed)

Amino-acid groups:
    group1  VLIMFWY  hydrophobic / aromatic  (highest q)
    group2  GPCA     helix-breaker / small apolar
    group3  STNQH    polar uncharged
    group4  KR       positively charged
    group5  DE       negatively charged      (lowest q)

Scaling   : SCALE_FACTOR = (10^6)^(1/8) ≈ 7.5  (same as F)

Outputs   : results/W_params.csv     –  trained q values for all 20 AAs
            Printed: training/test accuracy, confusion matrix, top-10 sequences
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

TRAIN_FILE  = os.path.join(os.path.dirname(__file__), "data", "W_rows_train.csv")
TEST_FILE   = os.path.join(os.path.dirname(__file__), "data", "W_rows_test.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_CSV     = os.path.join(RESULTS_DIR, "W_params.csv")

MIDDLE_POS  = 5

# Parameter ordering: hydrophobic residues first – matches how initial guesses
# are set and makes the output CSV easier to read.
AA_LIST = list("FWLIMVYPRTKASHCGQNDE")

AA_GROUPS = {
    "group1": list("VLIMFWY"),
    "group2": list("GPCA"),
    "group3": list("STNQH"),
    "group4": list("KR"),
    "group5": list("DE"),
}
AA_TO_GROUP = {aa: g for g, aas in AA_GROUPS.items() for aa in aas}

SCALE_FACTOR = (1e6) ** (1 / 8)

Q_TH_FIXED   = 1.0
N_FIXED      = 1.0

BASE_THRESHOLD = 40
MIN_THRESHOLD  = 9
R_WEIGHT       = 15
K_WEIGHT       = 10

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def generalized_sigmoid(Q, Q_th=1.0, n=1.0):
    """Generalised sigmoid: insertion probability from context-residue product Q."""
    if Q <= 0:
        return 0.0
    return 1.0 / (1.0 + (Q_th / Q) ** n)


def model_prediction(seq, params, Q_th=1.0, n=1.0):
    """Compute insertion probability for one sequence.

    Builds a {aa: q} map, multiplies q values over context positions,
    clamps each to 1e-12 to guard against numerical underflow, then
    passes the product through the sigmoid.
    """
    param_map = {aa: params[i] for i, aa in enumerate(AA_LIST)}
    product = 1.0
    for pos, aa in enumerate(seq, start=1):
        if pos == MIDDLE_POS or aa not in param_map:
            continue
        product *= max(1e-12, param_map[aa])
    return generalized_sigmoid(product, Q_th=Q_th, n=n)


def compute_dynamic_threshold(seq):
    """Lower insertion threshold for R/K-containing sequences."""
    n_r = seq.count('R')
    n_k = seq.count('K')
    return max(BASE_THRESHOLD - R_WEIGHT * n_r - K_WEIGHT * n_k, MIN_THRESHOLD)


def assign_labels(sequences, values):
    """Binary labels: 1 = inserted, 0 = not inserted."""
    labels = []
    for seq, val in zip(sequences, values):
        if 'R' not in seq and 'K' not in seq:
            threshold = BASE_THRESHOLD
        else:
            threshold = compute_dynamic_threshold(seq)
        labels.append(1 if val >= threshold else 0)
    return labels


def sum_squared_residuals(params, sequences, labels):
    """Objective for least_squares: squared prediction residuals."""
    return [
        (label - model_prediction(seq, params, Q_th=Q_TH_FIXED, n=N_FIXED)) ** 2
        for seq, label in zip(sequences, labels)
    ]


def confusion_matrix_stats(preds, labels):
    """TP, TN, FP, FN, and accuracy (%) from prediction and true-label lists."""
    TP = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    TN = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    FP = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    FN = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    accuracy = (TP + TN) / len(labels) * 100
    return TP, TN, FP, FN, accuracy


# ─────────────────────────────────────────────
# Bounds and initial guesses
# ─────────────────────────────────────────────

GROUP_PARAMS = {
    "group1": (0.10, 5.0, 1.87),
    "group2": (0.01, 2.0, 0.28),
    "group3": (0.01, 1.5, 0.21),
    "group4": (0.01, 1.0, 0.32),
    "group5": (0.01, 1.0, 0.19),
}

lower_bounds, upper_bounds, initial_guess = [], [], []
for aa in AA_LIST:
    group = AA_TO_GROUP.get(aa, "group5")
    lb, ub, ig = GROUP_PARAMS[group]
    lower_bounds.append(lb * SCALE_FACTOR)
    upper_bounds.append(ub * SCALE_FACTOR)
    initial_guess.append(ig * SCALE_FACTOR)

# ─────────────────────────────────────────────
# Load training data
# ─────────────────────────────────────────────

print("Loading training data ...")
df_train = pd.read_csv(TRAIN_FILE, header=None, sep=r"\s+")
df_train = df_train[~df_train[0].str.contains("X")]

train_seqs   = df_train[0].tolist()
train_values = df_train[3].tolist()
train_labels = assign_labels(train_seqs, train_values)

print(f"  {len(train_seqs):,} training sequences  |  "
      f"{sum(train_labels):,} positive ({100*sum(train_labels)/len(train_labels):.1f} %)")

# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────

print("\nOptimising parameters ...")
result = least_squares(
    sum_squared_residuals,
    initial_guess,
    bounds=(lower_bounds, upper_bounds),
    args=(train_seqs, train_labels),
    method='trf'
)

if not result.success:
    print(f"WARNING: optimisation may not have converged – {result.message}")

trained_params = result.x

# ─────────────────────────────────────────────
# Training-set evaluation
# ─────────────────────────────────────────────

train_scores = [model_prediction(s, trained_params) for s in train_seqs]
train_preds  = [1 if sc >= 0.5 else 0 for sc in train_scores]
TP, TN, FP, FN, acc = confusion_matrix_stats(train_preds, train_labels)

print("\n===== TRAINING SET RESULTS =====")
print(f"Accuracy : {acc:.2f} %")
print(f"TP {TP}  TN {TN}  FP {FP}  FN {FN}")

# ─────────────────────────────────────────────
# Save trained parameters
# ─────────────────────────────────────────────

params_df = pd.DataFrame({
    "AA":    AA_LIST + ["Q_th", "n", "Train_Accuracy", "TP", "TN", "FP", "FN"],
    "Value": list(trained_params) + [Q_TH_FIXED, N_FIXED, acc, TP, TN, FP, FN]
})
params_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved parameters to {OUT_CSV}")

# ─────────────────────────────────────────────
# Test-set evaluation
# ─────────────────────────────────────────────

print("\nLoading test data ...")
df_test = pd.read_csv(TEST_FILE, header=None, sep=r"\s+")
df_test = df_test[~df_test[0].str.contains("X")]

test_seqs   = df_test[0].tolist()
test_values = df_test[3].tolist()
test_labels = assign_labels(test_seqs, test_values)

print(f"  {len(test_seqs):,} test sequences  |  "
      f"{sum(test_labels):,} positive ({100*sum(test_labels)/len(test_labels):.1f} %)")

test_scores = [model_prediction(s, trained_params) for s in test_seqs]
test_preds  = [1 if sc >= 0.5 else 0 for sc in test_scores]
TP, TN, FP, FN, acc = confusion_matrix_stats(test_preds, test_labels)

print("\n===== TEST SET RESULTS =====")
print(f"Accuracy : {acc:.2f} %")
print(f"TP {TP}  TN {TN}  FP {FP}  FN {FN}")

# ─────────────────────────────────────────────
# Top 10 highest-scoring test sequences
# ─────────────────────────────────────────────

ranked = sorted(zip(test_seqs, test_scores), key=lambda x: x[1], reverse=True)
print("\n===== TOP 10 TEST SEQUENCES BY INSERTION SCORE =====")
for i, (seq, score) in enumerate(ranked[:10], 1):
    print(f"  {i:2d}.  {seq}   score = {score:.6f}")
