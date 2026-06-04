"""
train_test_F.py  –  Full 20-amino-acid model for Phenylalanine (F) on Human IDR data

Dataset   : 9-residue windows centred on F residues, extracted from human
            Intrinsically Disordered Regions (IDRs).  The dataset is pre-split into
            80 % training and 20 % test sets.

Data files:
    data/F_rows_train.csv  –  training set  (space-separated, no header)
    data/F_rows_test.csv   –  test set      (same format)
    col 0 – 9-residue sequence string (middle position = 5 is always F)
    col 1, 2 – intermediate PPM3 values (not used)
    col 3 – PPM3 membrane insertion depth score (higher = deeper insertion)

Model     : P(insert | seq) = sigmoid( product of q[aa] for context positions )
            where sigmoid(Q) = 1 / (1 + (Q_th / Q)^n)  with Q_th = 1, n = 1 (fixed)

            All 20 standard amino acids have their own q parameter.
            Amino acids are partitioned into 5 chemical groups to set initial guesses
            and bounds that reflect their known membrane-interaction properties:
              group1  VLIMFWY  – hydrophobic / aromatic  (high q → promotes insertion)
              group2  GPCA     – helix-breaker / small apolar
              group3  STNQH    – polar uncharged
              group4  KR       – positively charged
              group5  DE       – negatively charged      (low q → opposes insertion)

Scaling   : SCALE_FACTOR = (10^6)^(1/8) ≈ 7.5
            Parameters are optimised in the scaled space so that the 8-position
            product stays near 1 at the decision boundary.

Outputs   : results/F_params.csv     –  trained q values for all 20 AAs
            Printed: training accuracy, test accuracy, confusion matrix, top-10 sequences
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

TRAIN_FILE  = os.path.join(os.path.dirname(__file__), "data", "F_rows_train.csv")
TEST_FILE   = os.path.join(os.path.dirname(__file__), "data", "F_rows_test.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_CSV     = os.path.join(RESULTS_DIR, "F_params.csv")

MIDDLE_POS  = 5      # 1-indexed position of the fixed F residue

# Alphabetical order – used consistently for parameter indexing
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

# Chemical-group assignments for setting per-group bounds and initial guesses
AA_GROUPS = {
    "group1": list("VLIMFWY"),   # hydrophobic / aromatic
    "group2": list("GPCA"),      # helix-breaker / small apolar
    "group3": list("STNQH"),     # polar uncharged
    "group4": list("KR"),        # positively charged
    "group5": list("DE"),        # negatively charged
}
AA_TO_GROUP = {aa: g for g, aas in AA_GROUPS.items() for aa in aas}

# SCALE = (10^6)^(1/8)  –  keeps the optimised product near 1
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
    """Generalised sigmoid: P(insert) given the per-position product Q."""
    if Q <= 0:
        return 0.0
    return 1.0 / (1.0 + (Q_th / Q) ** n)


def model_prediction(seq, params, Q_th=1.0, n=1.0):
    """Compute insertion probability for a single sequence.

    Multiplies the q values at every context position (not MIDDLE_POS),
    clamping each value to at least 1e-12 to avoid numerical underflow in
    the product before the sigmoid.
    """
    param_map = {aa: params[i] for i, aa in enumerate(AA_LIST)}
    product = 1.0
    for pos, aa in enumerate(seq, start=1):
        if pos == MIDDLE_POS or aa not in param_map:
            continue
        product *= max(1e-12, param_map[aa])
    return generalized_sigmoid(product, Q_th=Q_th, n=n)


def compute_dynamic_threshold(seq):
    """Lower the insertion threshold for R/K-rich sequences."""
    n_r = seq.count('R')
    n_k = seq.count('K')
    return max(BASE_THRESHOLD - R_WEIGHT * n_r - K_WEIGHT * n_k, MIN_THRESHOLD)


def assign_labels(sequences, values):
    """Convert PPM3 depth scores to binary labels."""
    labels = []
    for seq, val in zip(sequences, values):
        if 'R' not in seq and 'K' not in seq:
            threshold = BASE_THRESHOLD
        else:
            threshold = compute_dynamic_threshold(seq)
        labels.append(1 if val >= threshold else 0)
    return labels


def sum_squared_residuals(params, sequences, labels):
    """Objective for least_squares: vector of squared prediction errors."""
    return [
        (label - model_prediction(seq, params, Q_th=Q_TH_FIXED, n=N_FIXED)) ** 2
        for seq, label in zip(sequences, labels)
    ]


def confusion_matrix_stats(preds, labels):
    """Return TP, TN, FP, FN and accuracy (%) from prediction and label lists."""
    TP = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    TN = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    FP = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    FN = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    accuracy = (TP + TN) / len(labels) * 100
    return TP, TN, FP, FN, accuracy


# ─────────────────────────────────────────────
# Bounds and initial guesses
# ─────────────────────────────────────────────

# Per-group (lb, ub, initial_guess) in unscaled space; scaled up before passing
# to the optimiser.  Hydrophobic residues get a higher starting q (more membrane-friendly).
GROUP_PARAMS = {
    "group1": (0.10, 5.0, 1.87),   # hydrophobic / aromatic
    "group2": (0.01, 2.0, 0.28),   # helix-breaker / small apolar
    "group3": (0.01, 1.5, 0.21),   # polar uncharged
    "group4": (0.01, 1.0, 0.32),   # positively charged
    "group5": (0.01, 1.0, 0.19),   # negatively charged
}

lower_bounds, upper_bounds, initial_guess = [], [], []
for aa in AA_LIST:
    group = AA_TO_GROUP.get(aa, "group5")
    lb, ub, ig = GROUP_PARAMS[group]
    lower_bounds.append(lb * SCALE_FACTOR)
    upper_bounds.append(ub * SCALE_FACTOR)
    initial_guess.append(ig * SCALE_FACTOR)

# ─────────────────────────────────────────────
# Load and label training data
# ─────────────────────────────────────────────

print("Loading training data ...")
df_train = pd.read_csv(TRAIN_FILE, header=None, sep=r"\s+")
# Remove sequences containing non-standard amino acid X
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
    "AA":       AA_LIST + ["Q_th", "n", "Train_Accuracy", "TP", "TN", "FP", "FN"],
    "Value":    list(trained_params) + [Q_TH_FIXED, N_FIXED, acc, TP, TN, FP, FN]
})
params_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved parameters to {OUT_CSV}")

# ─────────────────────────────────────────────
# Load and evaluate test set
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
