"""
train_F.py  –  5-letter alphabet model training for Phenylalanine (F)

Dataset   : 9-residue synthetic sequences with F fixed at the middle position (pos 5).
            The 8 context positions are drawn from a 5-letter reduced alphabet: L, R, G, N, E.
Data file : data/F_rows.csv  (space-separated, no header)
            col 0 – sequence string
            col 1, 2 – intermediate PPM3 values (not used here)
            col 3 – membrane insertion depth score from PPM3 (higher = deeper insertion)

Model     : P(insert | seq) = sigmoid( product of q[aa] for all context positions )
            where sigmoid(Q) = 1 / (1 + (Q_th / Q)^n)  with Q_th = 1, n = 1 (fixed)

Labels    : binary (1 = inserted, 0 = not inserted) assigned by a dynamic depth threshold
            that is lowered for sequences containing positively-charged residues R or K,
            since these are known to facilitate membrane insertion.

Training  : scipy.optimize.least_squares (Trust-Region Reflective method) minimises the
            sum of squared residuals between model predictions and binary labels.
            The loop repeats training at increasing dataset sizes to check parameter
            convergence (1 000, 10 000, 100 000, and the full dataset).

Outputs   : results/F_param_vs_training_size.csv  –  q values at each training size
            results/F_q_vs_training_size.png       –  convergence plot
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DATA_FILE      = os.path.join(os.path.dirname(__file__), "data", "F_rows.csv")
RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")
OUT_CSV        = os.path.join(RESULTS_DIR, "F_param_vs_training_size.csv")
OUT_PLOT       = os.path.join(RESULTS_DIR, "F_q_vs_training_size.png")

MIDDLE_POS     = 5          # 1-indexed position of the fixed aromatic residue
AA_LIST        = list("LRGNE")  # reduced 5-letter alphabet for context positions

# Scaling transforms the optimised parameters into a numerically well-conditioned
# range around 1.  For F: SCALE = (10^6)^(1/8) ≈ 7.5
SCALE_FACTOR   = (1e6) ** (1 / 8)

# Fixed sigmoid parameters (not optimised)
Q_TH_FIXED     = 1.0
N_FIXED        = 1.0

# Dynamic labelling thresholds
BASE_THRESHOLD = 40   # default insertion depth cutoff (PPM3 score)
MIN_THRESHOLD  = 9    # floor – never go below this even for R/K-rich sequences
R_WEIGHT       = 15   # each R in the sequence lowers the threshold by this amount
K_WEIGHT       = 10   # each K in the sequence lowers the threshold by this amount

# Training-size convergence study
TRAINING_SIZES = [1000, 10_000, 100_000]   # the full dataset size is appended automatically
RANDOM_SEED    = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def generalized_sigmoid(Q, Q_th=1.0, n=1.0):
    """Generalised sigmoid used as the insertion probability model.
    Returns 0 immediately for non-positive Q to avoid log/division errors."""
    if Q <= 0:
        return 0.0
    return 1.0 / (1.0 + (Q_th / Q) ** n)


def model_prediction(seq, params, Q_th=1.0, n=1.0):
    """Compute insertion probability for one sequence.

    The product of the per-amino-acid q parameters at all context positions
    (excluding the fixed aromatic at MIDDLE_POS) is passed through the sigmoid.
    Positions occupied by amino acids outside AA_LIST contribute nothing (q=1).
    """
    param_map = dict(zip(AA_LIST, params))   # {aa: q_value}
    product = 1.0
    for pos, aa in enumerate(seq, start=1):
        if pos == MIDDLE_POS or aa not in param_map:
            continue
        product *= param_map[aa]
    return generalized_sigmoid(product, Q_th=Q_th, n=n)


def compute_dynamic_threshold(seq):
    """Lower the insertion threshold for sequences containing R or K.

    R and K carry positive charge which is known to promote membrane
    interactions; their presence makes insertion more likely even at
    shallower PPM3 depths.
    """
    n_r = seq.count('R')
    n_k = seq.count('K')
    return max(BASE_THRESHOLD - R_WEIGHT * n_r - K_WEIGHT * n_k, MIN_THRESHOLD)


def assign_labels(sequences, values):
    """Convert raw PPM3 depth scores to binary labels using the dynamic threshold."""
    labels = []
    for seq, val in zip(sequences, values):
        if 'R' not in seq and 'K' not in seq:
            threshold = BASE_THRESHOLD
        else:
            threshold = compute_dynamic_threshold(seq)
        labels.append(1 if val >= threshold else 0)
    return labels


def sum_squared_residuals(params, sequences, labels, Q_th, n):
    """Objective function: vector of squared residuals for least_squares."""
    return [
        (label - model_prediction(seq, params, Q_th=Q_th, n=n)) ** 2
        for seq, label in zip(sequences, labels)
    ]


# ─────────────────────────────────────────────
# Bounds and initial guesses (scaled)
# ─────────────────────────────────────────────

# Initial q values are from prior optimisation runs; scaled up by SCALE_FACTOR
# so that the product of 8 context positions stays near 1 at the decision boundary.
INITIAL_GUESS_DICT = {'L': 0.41, 'R': 0.27, 'E': 0.11, 'G': 0.15, 'N': 0.12}

lower_bounds, upper_bounds, initial_guess = [], [], []
for aa in AA_LIST:
    if aa == 'L':
        lower_bounds.append(0.10 * SCALE_FACTOR)
        upper_bounds.append(3.00 * SCALE_FACTOR)
    else:
        lower_bounds.append(0.01 * SCALE_FACTOR)
        upper_bounds.append(0.50 * SCALE_FACTOR)
    initial_guess.append(INITIAL_GUESS_DICT[aa] * SCALE_FACTOR)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

print("Loading data ...")
df = pd.read_csv(DATA_FILE, header=None, sep=r"\s+")
sequences = np.array(df[0].tolist())
values    = df[3].tolist()

labels = np.array(assign_labels(sequences, values))
total_size = len(sequences)
print(f"  {total_size:,} sequences  |  "
      f"{labels.sum():,} positive ({100*labels.mean():.1f} %)")

# Append full-dataset size to the convergence study
training_sizes = sorted(set(TRAINING_SIZES + [total_size]))

# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

np.random.seed(RANDOM_SEED)
results_list = []

for size in training_sizes:
    print(f"\nTraining on {size:,} sequences ...")

    # Draw a random subset (without replacement) for this run
    idx = np.random.choice(total_size, size=size, replace=False)
    seq_sub   = sequences[idx]
    label_sub = labels[idx]

    result = least_squares(
        sum_squared_residuals,
        initial_guess,
        bounds=(lower_bounds, upper_bounds),
        args=(seq_sub, label_sub, Q_TH_FIXED, N_FIXED),
        method='trf'          # Trust-Region Reflective – handles bounds efficiently
    )

    row = {"Training_Size": size}
    if result.success:
        for aa, val in zip(AA_LIST, result.x):
            row[aa] = val
        print(f"  Converged.  Cost = {result.cost:.4f}")
    else:
        print(f"  WARNING: optimisation did not converge – {result.message}")
        for aa in AA_LIST:
            row[aa] = np.nan

    results_list.append(row)

# ─────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────

param_df = pd.DataFrame(results_list)
param_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved parameter table to {OUT_CSV}")
print(param_df.to_string(index=False))

# ─────────────────────────────────────────────
# Convergence plot
# ─────────────────────────────────────────────

colours = {'L': 'black', 'R': 'blue', 'G': 'gray', 'N': 'green', 'E': 'red'}

fig, ax = plt.subplots(figsize=(8, 5))
for aa in AA_LIST:
    ax.plot(param_df["Training_Size"], param_df[aa],
            marker='o', linewidth=2, markersize=10,
            color=colours[aa], label=aa)

ax.set_xlabel("Training size", fontsize=20, fontweight="bold")
ax.set_ylabel("q",             fontsize=20, fontweight="bold")
ax.set_xscale("log")
ax.legend(loc='upper center', ncol=5, fontsize=16, framealpha=1)
ax.set_yticks([0, 1, 2, 3])
ax.tick_params(labelsize=20)
fig.tight_layout()
fig.savefig(OUT_PLOT, dpi=500, bbox_inches='tight')
plt.show()
print(f"Saved convergence plot to {OUT_PLOT}")
