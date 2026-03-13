"""
Steam Cracker Coil Health Monitoring — Main Analysis Pipeline
CHEMENG 277, Winter 2026 | Author: Rupin Vaidya

Runs the full ML pipeline in sequence:
  1. Feature engineering   — lag and rolling features, per-cycle, per-phase
  2. Regression modelling  — Ridge/Lasso per target, cracking and
                             decoking independently (48 models total)
  3. Residual analysis     — post-equilibration residuals, CUSUM, per-cycle
                             trend statistics
  4. Classification        — residual-based degradation labelling and
                             leave-one-cycle-out logistic regression

Inputs (from data_prep.py):
  prepared_data/df_cracking_ss.csv
  prepared_data/df_decoking_ss.csv
  prepared_data/df_cycle_meta.csv

Outputs:
  figures/   — PNG plots
  results/   — CSV result tables + printed summary

See README.md for full documentation of all design decisions.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (mean_squared_error, r2_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings("ignore")

# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

# --- Paths -------------------------------------------------------------------
# Root directory: same folder that contains data.xlsx and prepared_data/
ROOT_DIR      = r"C:\Users\rupin\Documents\Documents\VS Code\CHEMENG 277\Project"
PREP_DIR      = os.path.join(ROOT_DIR, "prepared_data")
FIGURES_DIR   = os.path.join(ROOT_DIR, "figures")
RESULTS_DIR   = os.path.join(ROOT_DIR, "results")

# --- Feature engineering -----------------------------------------------------
LAG_STEPS        = [1, 5, 10, 30]   # minutes; lag-2 dropped (near-identical to
                                     # lag-1, adds collinearity without signal)
ROLL_WINDOW      = 30                   # minutes; smooths slow thermal drift

# --- Regression --------------------------------------------------------------
# Alpha candidates for 5-fold CV selection (one alpha per model)
ALPHA_CANDIDATES = [0.01, 0.1, 1, 10]
N_CV_FOLDS       = 3

# --- Residual analysis -------------------------------------------------------
# Baseline: first N training cycles used to establish nominal residual
# distribution (mean and std). Degradation threshold = mean + 1σ.
N_BASELINE_CYCLES       = 12   # Run 1 used 8 -- too narrow; inflated false positive
                               # rate to ~77%. Widened to 12 (3 full weeks) for a
                               # more stable nominal residual distribution.
RESID_THRESHOLD_SIGMA   = 2.0  # Run 1 used 1sigma -- near-trivial classifier (77%
                               # base rate). Raised to 2sigma (~2.3% exceedance
                               # under normality) for a meaningful anomaly flag.
RESID_ROLL_WINDOW       = 30   # minutes; rolling window for residual mean/std

# --- Classification ----------------------------------------------------------
CLF_N_BASELINE = N_BASELINE_CYCLES   # must always match N_BASELINE_CYCLES

# --- Targets -----------------------------------------------------------------
# 6 per-element average temperatures
ELEM_TARGETS = ["Elem1_Top", "Elem2", "Elem3", "Elem4", "Elem5", "Elem6_Bot"]

# 6 per-element max-zone temperatures (hottest zone within each element).
# Reduces zone targets from 18 to 6 while preserving full spatial resolution
# at the element level — the operationally meaningful unit for coil monitoring.
ZONE_TARGETS = ["Elem1_Top_maxzone", "Elem2_maxzone", "Elem3_maxzone",
                "Elem4_maxzone",    "Elem5_maxzone", "Elem6_Bot_maxzone"]

ALL_TARGETS = ELEM_TARGETS + ZONE_TARGETS   # 12 targets total

# --- Features ----------------------------------------------------------------
# Input features shared across all target models
SP_COL      = "Setpoint %"
TI_COLS     = ["TI 144 Avg_lag5min", "TI 145 Avg_lag5min"]

# Element averages used as lagged features (thermal state of neighbouring zones)
ELEM_AVG_COLS = ELEM_TARGETS   # same columns, used as lagged inputs

# =============================================================================
# 1.  SETUP
# =============================================================================

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "axes.titlesize":     11,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
})

def savefig(name):
    """Save current figure to the figures directory."""
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"    Saved: figures/{name}")

def savecsv(df, name):
    """Save a DataFrame to the results directory."""
    path = os.path.join(RESULTS_DIR, name)
    df.to_csv(path, index=False)
    print(f"    Saved: results/{name}")

# =============================================================================
# 2.  LOAD PREPARED DATA
# =============================================================================

print("=" * 60)
print("LOADING PREPARED DATA")
print("=" * 60)

df_crack = pd.read_csv(os.path.join(PREP_DIR, "df_cracking_ss.csv"),
                       parse_dates=["Timestamp"])
df_decok = pd.read_csv(os.path.join(PREP_DIR, "df_decoking_ss.csv"),
                       parse_dates=["Timestamp"])
df_meta  = pd.read_csv(os.path.join(PREP_DIR, "df_cycle_meta.csv"),
                       parse_dates=["start", "end"])

print(f"  Cracking SS : {len(df_crack):,} rows, "
      f"{df_crack['cycle_id'].nunique()} cycles")
print(f"  Decoking SS : {len(df_decok):,} rows, "
      f"{df_decok['cycle_id'].nunique()} cycles")
print(f"  Cycle meta  : {len(df_meta)} rows")

# =============================================================================
# 3.  FEATURE ENGINEERING
# =============================================================================
# Features are built per-cycle so that lag and rolling computations do not
# cross cycle boundaries (the furnace state resets between cycles).
#
# Feature set for each target:
#   - Setpoint % at current timestep
#   - Setpoint % at lags [1, 2, 5, 10, 30] min
#   - Each element average at lags [1, 2, 5, 10, 30] min
#   - Rolling 30-min mean and std of furnace_avg
#   - TI-144 Avg (5-min lag) and TI-145 Avg (5-min lag)    [already lagged]

print("\n" + "=" * 60)
print("STEP 1: FEATURE ENGINEERING")
print("=" * 60)

def build_features(df_phase, meta_phase, phase_label):
    """
    Build the feature matrix and target matrix for one phase (cracking or
    decoking). Operates per-cycle so lags never cross cycle boundaries.

    Parameters
    ----------
    df_phase    : DataFrame — steady-state rows for this phase
    meta_phase  : DataFrame — cycle metadata rows for this phase
    phase_label : str       — 'cracking_ss' or 'decoking_ss' (for logging)

    Returns
    -------
    features : DataFrame — feature matrix (rows = timesteps after dropna)
    targets  : DataFrame — target matrix aligned with features
    """
    feat_frames = []

    for cyc in sorted(df_phase["cycle_id"].unique()):
        sub = df_phase[df_phase["cycle_id"] == cyc].copy().reset_index(drop=True)

        # Retrieve equilibration time for this cycle from metadata
        meta_row = meta_phase[meta_phase["cycle"] == cyc]
        equil_min = int(meta_row["equil_time_min"].values[0]) if len(meta_row) else 0

        # Crop the equilibration transient — model trains/predicts on
        # post-equilibration steady state only
        sub = sub.iloc[equil_min:].reset_index(drop=True)

        f = pd.DataFrame()
        f["Timestamp"]  = sub["Timestamp"]
        f["cycle_id"]   = cyc
        f["split"]      = sub["split"]

        # --- Base feature: current setpoint ----------------------------------
        f[SP_COL] = sub[SP_COL]

        # --- Lagged setpoint -------------------------------------------------
        for lag in LAG_STEPS:
            f[f"sp_lag{lag}"] = sub[SP_COL].shift(lag)

        # --- Lagged element averages -----------------------------------------
        # Captures thermal inertia: where was each element N minutes ago?
        for elem in ELEM_AVG_COLS:
            for lag in LAG_STEPS:
                f[f"{elem}_lag{lag}"] = sub[elem].shift(lag)

        # --- Rolling statistics on furnace_avg -------------------------------
        # Provides a smoothed thermal state signal over the past 30 minutes
        f["furnace_roll_mean"] = (sub["furnace_avg"]
                                  .rolling(ROLL_WINDOW, min_periods=10).mean())
        f["furnace_roll_std"]  = (sub["furnace_avg"]
                                  .rolling(ROLL_WINDOW, min_periods=10).std())

        # --- Lagged refractory temperatures (already 5-min lagged in prep) --
        for ti_col in TI_COLS:
            f[ti_col] = sub[ti_col]

        # --- Target columns --------------------------------------------------
        for tgt in ALL_TARGETS:
            f[tgt] = sub[tgt]

        feat_frames.append(f)

    out = pd.concat(feat_frames, ignore_index=True).dropna()

    # Separate feature columns from target columns
    meta_cols = ["Timestamp", "cycle_id", "split"] + ALL_TARGETS
    feat_cols = [c for c in out.columns if c not in meta_cols]

    print(f"  {phase_label}: {len(out):,} rows × {len(feat_cols)} features "
          f"after dropna  "
          f"(train={( out['split']=='train').sum():,}, "
          f"test={( out['split']=='test').sum():,})")

    return out[["Timestamp", "cycle_id", "split"] + feat_cols], out[ALL_TARGETS]


meta_crack = df_meta[df_meta["phase"] == "cracking_ss"].reset_index(drop=True)
meta_decok = df_meta[df_meta["phase"] == "decoking_ss"].reset_index(drop=True)

feat_crack, tgt_crack = build_features(df_crack, meta_crack, "cracking_ss")
feat_decok, tgt_decok = build_features(df_decok, meta_decok, "decoking_ss")

FEAT_COLS = [c for c in feat_crack.columns
             if c not in ["Timestamp", "cycle_id", "split"]]

print(f"\n  Feature columns ({len(FEAT_COLS)}):")
print(f"    {FEAT_COLS}")

# =============================================================================
# 4.  REGRESSION MODELLING
# =============================================================================
# One model per target x per phase = 24 models total (12 targets x 2 phases).
# Train: cycles 1-17, Test: cycles 18-28.
#
# Alpha selection (shared per phase):
#   A single shared alpha is selected for Ridge and Lasso via
#   5-fold CV on furnace_avg as the representative target. All 12 targets then
#   use this shared alpha. This is justified because all targets share the same
#   feature matrix -- regularisation strength is a property of the feature
#   space, not the individual target. Reduces CV fits from ~1440 to ~30/phase.
#
# Models compared: OLS (baseline), Ridge, Lasso (best by test R2).
# ElasticNet excluded — adds little over Lasso when features are collinear
# and Lasso already provides the sparse solution if needed.

print("\n" + "=" * 60)
print("STEP 2: REGRESSION MODELLING")
print("=" * 60)

def select_shared_alphas(X_train_s, y_rep_train):
    """
    Select one alpha per regularised model type via 5-fold CV, using
    furnace_roll_mean as the representative target. The chosen alphas are
    then applied to all 12 targets, since all share the same feature matrix.

    Parameters
    ----------
    X_train_s    : ndarray — scaled training feature matrix
    y_rep_train  : ndarray — training values for the representative target

    Returns
    -------
    dict mapping {Ridge: best_alpha, Lasso: best_alpha}
    """
    shared = {}
    for model_cls in [Ridge, Lasso]:
        best_alpha, best_score = ALPHA_CANDIDATES[0], -np.inf
        for a in ALPHA_CANDIDATES:
            scores = cross_val_score(
                model_cls(alpha=a, max_iter=10000),
                X_train_s, y_rep_train,
                cv=N_CV_FOLDS, scoring="r2"
            )
            if scores.mean() > best_score:
                best_score, best_alpha = scores.mean(), a
        shared[model_cls] = best_alpha
        print(f"    {model_cls.__name__:<12s}: best alpha={best_alpha}  "
              f"(CV R²={best_score:.5f})")
    return shared


def fit_phase_models(feat_df, tgt_df, phase_label):
    """
    Fit OLS, Ridge, and Lasso for every target in ALL_TARGETS.
    Alpha is selected once per phase via CV on furnace_roll_mean, then
    shared across all 12 targets (same feature matrix for all).

    Parameters
    ----------
    feat_df     : DataFrame — feature matrix with Timestamp/cycle_id/split cols
    tgt_df      : DataFrame — target matrix aligned row-for-row with feat_df
    phase_label : str

    Returns
    -------
    model_results : dict  — {target: {model_name: {rmse, r2}}}
    best_models   : dict  — {target: fitted sklearn model (best by test R²)}
    scalers       : dict  — {target: fitted StandardScaler}
    best_names    : dict  — {target: name of best model}
    """
    train_mask = feat_df["split"] == "train"
    test_mask  = feat_df["split"] == "test"

    X_train = feat_df.loc[train_mask, FEAT_COLS].values
    X_test  = feat_df.loc[test_mask,  FEAT_COLS].values

    # One scaler for the whole phase — all targets share the same feature matrix
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_train)
    Xte_s  = scaler.transform(X_test)

    print(f"\n  {phase_label}  "
          f"(train={train_mask.sum():,} rows, test={test_mask.sum():,} rows)")
    print(f"  Selecting shared alphas via CV on furnace_roll_mean ...")
    y_rep         = feat_df.loc[train_mask, "furnace_roll_mean"].values
    shared_alphas = select_shared_alphas(Xtr_s, y_rep)

    ar = shared_alphas[Ridge]
    al = shared_alphas[Lasso]

    model_results = {}
    best_models   = {}
    best_names    = {}

    print(f"  {'Target':<25s}  {'Best model':<22s}  "
          f"{'RMSE (°C)':>10s}  {'R²':>10s}")
    print("  " + "-" * 72)

    for tgt in ALL_TARGETS:
        y_train = tgt_df.loc[train_mask, tgt].values
        y_test  = tgt_df.loc[test_mask,  tgt].values

        candidates = {
            "OLS":            LinearRegression(),
            f"Ridge(a={ar})": Ridge(alpha=ar),
            f"Lasso(a={al})": Lasso(alpha=al, max_iter=10000),
        }

        tgt_results = {}
        best_r2, best_name, best_mdl = -np.inf, None, None

        for name, mdl in candidates.items():
            mdl.fit(Xtr_s, y_train)
            y_pred = mdl.predict(Xte_s)
            rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
            r2     = r2_score(y_test, y_pred)
            tgt_results[name] = {"rmse": rmse, "r2": r2}
            if r2 > best_r2:
                best_r2, best_name, best_mdl = r2, name, mdl

        model_results[tgt] = tgt_results
        best_models[tgt]   = best_mdl
        best_names[tgt]    = best_name

        print(f"  {tgt:<25s}  {best_name:<22s}  "
              f"{tgt_results[best_name]['rmse']:>10.4f}  "
              f"{tgt_results[best_name]['r2']:>10.6f}")

    # Return one scaler entry per target (all point to the same object)
    scalers = {tgt: scaler for tgt in ALL_TARGETS}
    return model_results, best_models, scalers, best_names


print("\n--- CRACKING ---")
crack_results, crack_models, crack_scalers, crack_best = fit_phase_models(
    feat_crack, tgt_crack, "cracking_ss")

print("\n--- DECOKING ---")
decok_results, decok_models, decok_scalers, decok_best = fit_phase_models(
    feat_decok, tgt_decok, "decoking_ss")

# --- Save model performance summary -----------------------------------------
def build_results_table(model_results, best_names, phase_label):
    rows = []
    for tgt, model_dict in model_results.items():
        best = best_names[tgt]
        for mname, metrics in model_dict.items():
            rows.append({
                "phase":      phase_label,
                "target":     tgt,
                "model":      mname,
                "is_best":    mname == best,
                "rmse":       metrics["rmse"],
                "r2":         metrics["r2"],
            })
    return pd.DataFrame(rows)

results_table = pd.concat([
    build_results_table(crack_results, crack_best, "cracking_ss"),
    build_results_table(decok_results, decok_best, "decoking_ss"),
], ignore_index=True)

savecsv(results_table, "regression_results.csv")

# =============================================================================
# 5.  RESIDUAL ANALYSIS
# =============================================================================
# For each phase and each target, compute timestep-level residuals using the
# best model. Residuals are computed post-equilibration only (already handled
# in feature engineering — equil rows were cropped before building feat_df).
#
# Per-timestep: residual = actual − predicted
# Per-cycle summary:
#   - mean_resid       : systematic model bias this cycle
#   - max_abs_resid    : largest single deviation this cycle
#   - cusum_final      : cumulative sum of residuals at end of cycle
#                        (detects progressive drift)
#
# Degradation label: timestep is 'degraded' if its rolling 30-min mean
# residual exceeds (baseline_mean + 1σ), where baseline is computed from
# the first N_BASELINE_CYCLES training cycles.

print("\n" + "=" * 60)
print("STEP 3: RESIDUAL ANALYSIS")
print("=" * 60)


def compute_residuals(feat_df, tgt_df, best_models, scalers, phase_label):
    """
    Compute per-timestep residuals for every target using the best model for
    each. Also derives per-cycle summary statistics.

    Returns
    -------
    resid_df   : DataFrame — timestep-level residuals for all targets
    summary_df : DataFrame — per-cycle summary (mean, max_abs, cusum_final)
    """
    cycles     = sorted(feat_df["cycle_id"].unique())
    resid_frames = []

    for cyc in cycles:
        mask = feat_df["cycle_id"] == cyc
        X    = feat_df.loc[mask, FEAT_COLS].values
        ts   = feat_df.loc[mask, "Timestamp"].values
        spl  = feat_df.loc[mask, "split"].values[0]

        row = {
            "Timestamp": ts,
            "cycle_id":  cyc,
            "split":     spl,
        }

        for tgt in ALL_TARGETS:
            scaler = scalers[tgt]
            model  = best_models[tgt]
            Xs     = scaler.transform(X)
            y_true = tgt_df.loc[mask, tgt].values
            y_pred = model.predict(Xs)
            row[f"{tgt}_actual"]   = y_true
            row[f"{tgt}_pred"]     = y_pred
            row[f"{tgt}_resid"]    = y_true - y_pred

        resid_frames.append(pd.DataFrame(row))

    resid_df = pd.concat(resid_frames, ignore_index=True)

    # Rolling 30-min mean residual per target (within each cycle)
    roll_frames = []
    for cyc in cycles:
        mask = resid_df["cycle_id"] == cyc
        sub  = resid_df[mask].copy().reset_index(drop=True)
        for tgt in ALL_TARGETS:
            rcol = f"{tgt}_resid"
            sub[f"{tgt}_resid_roll_mean"] = (sub[rcol]
                .rolling(RESID_ROLL_WINDOW, min_periods=5).mean())
            sub[f"{tgt}_resid_cusum"] = sub[rcol].cumsum()
        roll_frames.append(sub)
    resid_df = pd.concat(roll_frames, ignore_index=True)

    # Per-cycle summary statistics
    summary_rows = []
    for cyc in cycles:
        mask = resid_df["cycle_id"] == cyc
        sub  = resid_df[mask]
        spl  = sub["split"].values[0]
        rec  = {"cycle_id": cyc, "phase": phase_label, "split": spl}
        for tgt in ALL_TARGETS:
            r = sub[f"{tgt}_resid"].values
            rec[f"{tgt}_mean_resid"]    = r.mean()
            rec[f"{tgt}_max_abs_resid"] = np.abs(r).max()
            rec[f"{tgt}_cusum_final"]   = r.cumsum()[-1]
        summary_rows.append(rec)

    summary_df = pd.DataFrame(summary_rows)

    print(f"  {phase_label}: residuals computed for {len(cycles)} cycles, "
          f"{len(ALL_TARGETS)} targets")
    return resid_df, summary_df


resid_crack, summary_crack = compute_residuals(
    feat_crack, tgt_crack, crack_models, crack_scalers, "cracking_ss")
resid_decok, summary_decok = compute_residuals(
    feat_decok, tgt_decok, decok_models, decok_scalers, "decoking_ss")

savecsv(summary_crack, "residual_summary_cracking.csv")
savecsv(summary_decok, "residual_summary_decoking.csv")

# =============================================================================
# 6.  CLASSIFICATION
# =============================================================================
# Label each timestep as 'nominal' (0) or 'degraded' (1).
#
# Labelling rule:
#   1. Compute the baseline distribution of rolling-mean residuals from the
#      first N_BASELINE_CYCLES training cycles (separately per phase and target).
#   2. A timestep is 'degraded' for target T if its 30-min rolling mean
#      residual for T exceeds (baseline_mean + 1σ).
#   3. A timestep is degraded overall if ANY of the 24 targets is degraded
#      (union rule — conservative for health monitoring).
#
# Classifier: Logistic Regression with leave-one-cycle-out (LOCO) CV.
# Features: rolling mean and std residuals for all 24 targets + setpoint.
# Evaluation: accuracy, precision, recall, F1 per holdout cycle.

print("\n" + "=" * 60)
print("STEP 4: CLASSIFICATION")
print("=" * 60)


def label_degraded(resid_df, phase_label):
    """
    Assign degradation labels to every timestep using the residual-based rule.
    Baseline computed from first N_BASELINE_CYCLES training cycles.

    Returns resid_df with added 'degraded' column (0/1).
    """
    resid_df = resid_df.copy()
    train_cycles = sorted(
        resid_df.loc[resid_df["split"] == "train", "cycle_id"].unique()
    )
    baseline_cycles = train_cycles[:N_BASELINE_CYCLES]

    # Compute per-target threshold from baseline cycles
    thresholds = {}
    baseline_mask = resid_df["cycle_id"].isin(baseline_cycles)
    for tgt in ALL_TARGETS:
        roll_col = f"{tgt}_resid_roll_mean"
        baseline_vals = resid_df.loc[baseline_mask, roll_col].dropna()
        mu  = baseline_vals.mean()
        sig = baseline_vals.std()
        thresholds[tgt] = mu + RESID_THRESHOLD_SIGMA * sig  # mean + N*sigma

    print(f"  {phase_label}: baseline from cycles "
          f"{baseline_cycles[0]}–{baseline_cycles[-1]}")

    # Label each timestep: degraded if ANY target exceeds its threshold
    degraded_flags = pd.DataFrame(index=resid_df.index)
    for tgt in ALL_TARGETS:
        roll_col  = f"{tgt}_resid_roll_mean"
        thresh    = thresholds[tgt]
        degraded_flags[tgt] = (resid_df[roll_col] > thresh).astype(int)

    resid_df["degraded"] = degraded_flags.max(axis=1)   # union rule

    n_deg  = resid_df["degraded"].sum()
    n_tot  = resid_df["degraded"].notna().sum()
    print(f"  {phase_label}: {n_deg:,} / {n_tot:,} timesteps labelled degraded "
          f"({100*n_deg/n_tot:.1f}%)")
    return resid_df, thresholds


resid_crack, thresholds_crack = label_degraded(resid_crack, "cracking_ss")
resid_decok, thresholds_decok = label_degraded(resid_decok, "decoking_ss")


def run_loco_classification(resid_df, phase_label):
    """
    Leave-one-cycle-out logistic regression classifier.

    Features used: rolling mean residuals for all 24 targets + setpoint.
    Each cycle is held out once; the classifier is trained on all other cycles.

    Returns
    -------
    loco_results : list of dicts — per-cycle accuracy, precision, recall, F1
    """
    clf_feat_cols = (
        [f"{tgt}_resid_roll_mean" for tgt in ALL_TARGETS]
        + [f"{tgt}_resid_roll_std" for tgt in ALL_TARGETS
           if f"{tgt}_resid_roll_std" in resid_df.columns]
        + [SP_COL]
    )
    # Some roll_std cols may not exist; keep only present ones
    clf_feat_cols = [c for c in clf_feat_cols if c in resid_df.columns]

    # Add roll_std if missing (compute now)
    for tgt in ALL_TARGETS:
        std_col = f"{tgt}_resid_roll_std"
        if std_col not in resid_df.columns:
            frames = []
            for cyc in sorted(resid_df["cycle_id"].unique()):
                mask = resid_df["cycle_id"] == cyc
                sub  = resid_df[mask].copy()
                sub[std_col] = (sub[f"{tgt}_resid"]
                                .rolling(RESID_ROLL_WINDOW, min_periods=5).std())
                frames.append(sub)
            resid_df = pd.concat(frames).sort_index()
            clf_feat_cols.append(std_col)

    cycles      = sorted(resid_df["cycle_id"].unique())
    loco_results = []

    for holdout_cyc in cycles:
        train_mask = (resid_df["cycle_id"] != holdout_cyc)
        test_mask  = (resid_df["cycle_id"] == holdout_cyc)

        train_data = resid_df[train_mask].dropna(subset=clf_feat_cols + ["degraded"])
        test_data  = resid_df[test_mask ].dropna(subset=clf_feat_cols + ["degraded"])

        if train_data["degraded"].nunique() < 2:
            continue   # can't train if only one class in training set

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                max_iter=1000,
                class_weight="balanced",  # handles class imbalance
                random_state=42
            ))
        ])
        pipe.fit(train_data[clf_feat_cols], train_data["degraded"])
        preds = pipe.predict(test_data[clf_feat_cols])
        probs = pipe.predict_proba(test_data[clf_feat_cols])[:, 1]
        true  = test_data["degraded"].values

        report = classification_report(true, preds, output_dict=True,
                                        zero_division=0)
        loco_results.append({
            "cycle_id":   holdout_cyc,
            "phase":      phase_label,
            "split":      test_data["split"].values[0],
            "accuracy":   report["accuracy"],
            "precision":  report.get("1", {}).get("precision", np.nan),
            "recall":     report.get("1", {}).get("recall",    np.nan),
            "f1":         report.get("1", {}).get("f1-score",  np.nan),
            "n_degraded": int(true.sum()),
            "n_total":    len(true),
            "preds":      preds,
            "probs":      probs,
            "true":       true,
            "timestamps": test_data["Timestamp"].values,
        })

    mean_acc = np.mean([r["accuracy"] for r in loco_results])
    mean_f1  = np.mean([r["f1"]       for r in loco_results
                        if not np.isnan(r["f1"])])
    print(f"  {phase_label}: LOCO-CV mean accuracy={mean_acc:.3f}, "
          f"mean F1={mean_f1:.3f}")
    return loco_results, resid_df


print()
loco_crack, resid_crack = run_loco_classification(resid_crack, "cracking_ss")
loco_decok, resid_decok = run_loco_classification(resid_decok, "decoking_ss")

# Save LOCO results
def loco_to_df(loco_results):
    return pd.DataFrame([
        {k: v for k, v in r.items()
         if k not in ("preds", "probs", "true", "timestamps")}
        for r in loco_results
    ])

savecsv(loco_to_df(loco_crack), "classification_loco_cracking.csv")
savecsv(loco_to_df(loco_decok), "classification_loco_decoking.csv")

# =============================================================================
# 7.  FIGURES
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: GENERATING FIGURES")
print("=" * 60)

CRACK_COLOUR = "#E53935"   # red family for cracking
DECOK_COLOUR = "#1E88E5"   # blue family for decoking
VIRIDIS      = plt.cm.viridis

# ── Figure 1: Model performance — RMSE and R² by target ─────────────────────
print("\n  Figure 1: Model performance by target")

best_rows = results_table[results_table["is_best"]]
crack_best_rows = best_rows[best_rows["phase"] == "cracking_ss"]
decok_best_rows = best_rows[best_rows["phase"] == "decoking_ss"]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Regression Model Performance — Best Model per Target",
             fontsize=13, fontweight="bold")

for row_idx, (phase_rows, phase_label, colour) in enumerate([
    (crack_best_rows, "Cracking", CRACK_COLOUR),
    (decok_best_rows, "Decoking", DECOK_COLOUR),
]):
    targets = phase_rows["target"].tolist()
    rmses   = phase_rows["rmse"].tolist()
    r2s     = phase_rows["r2"].tolist()

    # Colour bars differently for element targets vs zone targets
    bar_colours = [colour if t in ELEM_TARGETS else "#BDBDBD" for t in targets]

    ax_rmse = axes[row_idx, 0]
    ax_r2   = axes[row_idx, 1]

    ax_rmse.barh(targets, rmses, color=bar_colours)
    ax_rmse.set_xlabel("RMSE (°C)")
    ax_rmse.set_title(f"{phase_label} — Test RMSE")
    ax_rmse.axvline(np.mean(rmses), color="black", ls="--", lw=1,
                    label=f"Mean={np.mean(rmses):.3f}°C")
    ax_rmse.legend(fontsize=8)

    ax_r2.barh(targets, r2s, color=bar_colours)
    ax_r2.set_xlabel("R²")
    ax_r2.set_title(f"{phase_label} — Test R²")
    ax_r2.set_xlim(max(0, min(r2s) - 0.002), 1.001)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=colour,   label="Element avg target"),
                  Patch(facecolor="#BDBDBD", label="Zone max target")]
    ax_r2.legend(handles=legend_els, fontsize=7)

plt.tight_layout()
savefig("01_model_performance.png")

# ── Figure 2: Predicted vs Actual — one representative target per phase ──────
print("  Figure 2: Predicted vs actual (Elem3, both phases)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Predicted vs Actual — Element 3 Average Temperature (Test Set)",
             fontsize=12, fontweight="bold")

for ax, resid_df, phase_label, colour in [
    (axes[0], resid_crack, "Cracking", CRACK_COLOUR),
    (axes[1], resid_decok, "Decoking", DECOK_COLOUR),
]:
    tgt = "Elem3"
    test_mask = resid_df["split"] == "test"
    actual = resid_df.loc[test_mask, f"{tgt}_actual"].values
    pred   = resid_df.loc[test_mask, f"{tgt}_pred"].values
    ax.scatter(actual, pred, s=2, alpha=0.25, color=colour, rasterized=True)
    mn, mx = actual.min(), actual.max()
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.2, label="Perfect prediction")
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2   = r2_score(actual, pred)
    ax.set_xlabel("Actual (°C)")
    ax.set_ylabel("Predicted (°C)")
    ax.set_title(f"{phase_label}  RMSE={rmse:.3f}°C  R²={r2:.5f}")
    ax.legend(fontsize=8)

plt.tight_layout()
savefig("02_pred_vs_actual.png")

# ── Figure 3: Residual trends per cycle — element averages ───────────────────
print("  Figure 3: Mean residual trend across cycles (element targets)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Mean Residual per Cycle — Element Average Temperatures",
             fontsize=12, fontweight="bold")

for ax, summary_df, phase_label, colour in [
    (axes[0], summary_crack, "Cracking", CRACK_COLOUR),
    (axes[1], summary_decok, "Decoking", DECOK_COLOUR),
]:
    cycles = summary_df["cycle_id"].values
    for i, elem in enumerate(ELEM_TARGETS):
        col = f"{elem}_mean_resid"
        c   = VIRIDIS(i / len(ELEM_TARGETS))
        ax.plot(cycles, summary_df[col].values, "o-", lw=1.2, ms=4,
                color=c, label=elem)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Mean Residual (°C)")
    ax.set_title(f"{phase_label}")
    ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
savefig("03_mean_residual_trend.png")

# ── Figure 4: CUSUM residuals — element averages ─────────────────────────────
print("  Figure 4: CUSUM residual trajectories across cycles")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("CUSUM Residuals per Cycle — Element 3 (colour = cycle number)",
             fontsize=12, fontweight="bold")

for ax, resid_df, phase_label, cmap_name in [
    (axes[0], resid_crack, "Cracking", "viridis"),
    (axes[1], resid_decok, "Decoking", "plasma"),
]:
    tgt   = "Elem3"
    cmap  = plt.get_cmap(cmap_name)
    cycles = sorted(resid_df["cycle_id"].unique())
    for i, cyc in enumerate(cycles):
        sub = resid_df[resid_df["cycle_id"] == cyc].reset_index(drop=True)
        t   = np.arange(len(sub))
        ax.plot(t, sub[f"{tgt}_resid_cusum"].values,
                lw=0.8, color=cmap(i / len(cycles)), alpha=0.8)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Minutes from Cycle Start (post-equilibration)")
    ax.set_ylabel("Cumulative Residual (°C·min)")
    ax.set_title(f"{phase_label}")
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name),
                                norm=plt.Normalize(1, len(cycles)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Cycle number")

plt.tight_layout()
savefig("04_cusum_residuals.png")

# ── Figure 5: Zone-level max residual heatmap ─────────────────────────────────
print("  Figure 5: Zone max residual heatmap across cycles")

fig, axes = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle("Per-Zone Max Absolute Residual Across Cycles",
             fontsize=12, fontweight="bold")

for ax, summary_df, phase_label, cmap_name in [
    (axes[0], summary_crack, "Cracking", "hot"),
    (axes[1], summary_decok, "Decoking", "YlOrRd"),
]:
    cycles    = summary_df["cycle_id"].values
    zone_cols = [f"{t}_max_abs_resid" for t in ZONE_TARGETS]
    data      = summary_df[zone_cols].values   # (n_cycles, 6)

    im = ax.imshow(data.T, aspect="auto", cmap=cmap_name,
                   vmin=0, vmax=np.nanpercentile(data, 95))
    ax.set_yticks(range(len(ZONE_TARGETS)))
    ax.set_yticklabels([t.replace("_maxzone", "") for t in ZONE_TARGETS], fontsize=7)
    ax.set_xticks(range(len(cycles)))
    ax.set_xticklabels([str(c) for c in cycles], fontsize=7, rotation=45)
    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Zone")
    ax.set_title(f"{phase_label}")
    plt.colorbar(im, ax=ax, label="Max |Residual| (°C)", shrink=0.7)

plt.tight_layout()
savefig("05_zone_residual_heatmap.png")

# ── Figure 6: Degradation probability per cycle ───────────────────────────────
print("  Figure 6: Degradation probability trajectories")

for loco_results, phase_label, colour, fname in [
    (loco_crack, "Cracking", CRACK_COLOUR, "06a_degradation_probs_cracking.png"),
    (loco_decok, "Decoking", DECOK_COLOUR, "06b_degradation_probs_decoking.png"),
]:
    cycles = [r["cycle_id"] for r in loco_results]
    ncols  = 5
    nrows  = (len(cycles) + ncols - 1) // ncols
    fig, axes_flat = plt.subplots(nrows, ncols, figsize=(18, nrows * 2.5),
                                   sharey=True)
    axes_flat = axes_flat.flatten()
    fig.suptitle(f"{phase_label} — P(Degraded) per Cycle (LOCO-CV)",
                 fontsize=12, fontweight="bold")

    for i, res in enumerate(loco_results):
        ax = axes_flat[i]
        t  = np.arange(len(res["probs"]))
        ax.fill_between(t, res["probs"], alpha=0.25, color=colour)
        ax.plot(t, res["probs"], lw=0.8, color=colour)
        ax.axhline(0.5, color="red", ls="--", lw=0.8)
        ax.set_title(f"Cycle {res['cycle_id']}  acc={res['accuracy']:.2f}",
                     fontsize=7.5)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=6)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.text(0.5, 0.01, "Timestep (post-equilibration)", ha="center", fontsize=9)
    fig.text(0.005, 0.5, "P(Degraded)", va="center", rotation="vertical", fontsize=9)
    plt.tight_layout(rect=[0.015, 0.025, 1, 0.97])
    savefig(fname)

# ── Figure 7: Classifier accuracy across cycles ───────────────────────────────
print("  Figure 7: Classifier accuracy trend")

fig, ax = plt.subplots(figsize=(11, 4.5))
for loco_results, label, colour, marker in [
    (loco_crack, "Cracking", CRACK_COLOUR, "o"),
    (loco_decok, "Decoking", DECOK_COLOUR, "s"),
]:
    cx = [r["cycle_id"] for r in loco_results]
    ca = [r["accuracy"] for r in loco_results]
    cf = [r["f1"]       for r in loco_results]
    ax.plot(cx, ca, marker=marker, lw=1.5, ms=5, color=colour,
            label=f"{label} — accuracy")
    ax.plot(cx, cf, marker=marker, lw=1.0, ms=4, color=colour,
            alpha=0.5, ls="--", label=f"{label} — F1")

ax.axhline(0.5, color="black", ls=":", lw=0.8, label="Random chance")
ax.axvline(17, color="grey", ls="--", lw=1,
           label="Train / test boundary")
ax.set_xlabel("Cycle Number")
ax.set_ylabel("Score")
ax.set_title("Degradation Classifier: Accuracy and F1 per Cycle (LOCO-CV)")
ax.legend(fontsize=8, ncol=2)
ax.set_ylim(0, 1.05)
plt.tight_layout()
savefig("07_classifier_accuracy.png")

# ── Figure 8: Summary panel ───────────────────────────────────────────────────
print("  Figure 8: Summary panel")

fig = plt.figure(figsize=(18, 11))
gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38,
                         top=0.90, bottom=0.07, left=0.06, right=0.97)
fig.suptitle(
    "Steam Cracker Coil Health Monitoring — Results Summary\n"
    "CHEMENG 277, Winter 2026 | Rupin Vaidya",
    fontsize=13, fontweight="bold"
)

# Panel A: RMSE by target (cracking)
ax_a = fig.add_subplot(gs[0, 0])
cr_rmse = crack_best_rows["rmse"].values
cr_tgts = crack_best_rows["target"].values
bar_c   = [CRACK_COLOUR if t in ELEM_TARGETS else "#EF9A9A" for t in cr_tgts]
ax_a.barh(cr_tgts, cr_rmse, color=bar_c)
ax_a.set_xlabel("RMSE (°C)", fontsize=8)
ax_a.set_title("A.  Cracking — RMSE by Target", fontsize=9)
ax_a.tick_params(labelsize=6)

# Panel B: RMSE by target (decoking)
ax_b = fig.add_subplot(gs[0, 1])
dk_rmse = decok_best_rows["rmse"].values
dk_tgts = decok_best_rows["target"].values
bar_d   = [DECOK_COLOUR if t in ELEM_TARGETS else "#90CAF9" for t in dk_tgts]
ax_b.barh(dk_tgts, dk_rmse, color=bar_d)
ax_b.set_xlabel("RMSE (°C)", fontsize=8)
ax_b.set_title("B.  Decoking — RMSE by Target", fontsize=9)
ax_b.tick_params(labelsize=6)

# Panel C: Mean residual trend (Elem3 representative)
ax_c = fig.add_subplot(gs[0, 2])
for summary_df, label, colour in [
    (summary_crack, "Cracking", CRACK_COLOUR),
    (summary_decok, "Decoking", DECOK_COLOUR),
]:
    ax_c.plot(summary_df["cycle_id"], summary_df["Elem3_mean_resid"],
              "o-", lw=1.2, ms=4, color=colour, label=label)
ax_c.axhline(0, color="black", lw=0.8, ls="--")
ax_c.set_xlabel("Cycle", fontsize=8)
ax_c.set_ylabel("Mean Residual (°C)", fontsize=8)
ax_c.set_title("C.  Elem3 Mean Residual Trend", fontsize=9)
ax_c.legend(fontsize=7)

# Panel D: CUSUM (Elem3, cracking)
ax_d = fig.add_subplot(gs[1, 0])
cmap_v = plt.get_cmap("viridis")
cycles_c = sorted(resid_crack["cycle_id"].unique())
for i, cyc in enumerate(cycles_c):
    sub = resid_crack[resid_crack["cycle_id"] == cyc]
    ax_d.plot(np.arange(len(sub)), sub["Elem3_resid_cusum"].values,
              lw=0.7, color=cmap_v(i / len(cycles_c)), alpha=0.8)
ax_d.axhline(0, color="black", lw=0.7, ls="--")
ax_d.set_xlabel("Timestep", fontsize=8)
ax_d.set_ylabel("CUSUM Residual", fontsize=8)
ax_d.set_title("D.  Cracking CUSUM (Elem3)", fontsize=9)

# Panel E: Classifier accuracy
ax_e = fig.add_subplot(gs[1, 1])
for loco_results, label, colour, marker in [
    (loco_crack, "Cracking", CRACK_COLOUR, "o"),
    (loco_decok, "Decoking", DECOK_COLOUR, "s"),
]:
    cx = [r["cycle_id"] for r in loco_results]
    ca = [r["accuracy"] for r in loco_results]
    ax_e.plot(cx, ca, marker=marker, lw=1.2, ms=4, color=colour, label=label)
ax_e.axhline(0.5, color="black", ls=":", lw=0.8)
ax_e.set_xlabel("Cycle", fontsize=8)
ax_e.set_ylabel("Accuracy", fontsize=8)
ax_e.set_title("E.  Classifier Accuracy (LOCO-CV)", fontsize=9)
ax_e.legend(fontsize=7)
ax_e.set_ylim(0, 1.05)

# Panel F: Zone residual heatmap (cracking)
ax_f = fig.add_subplot(gs[1, 2])
zone_cols = [f"{t}_max_abs_resid" for t in ZONE_TARGETS]
data_c    = summary_crack[zone_cols].values
im = ax_f.imshow(data_c.T, aspect="auto", cmap="hot",
                  vmin=0, vmax=np.nanpercentile(data_c, 95))
ax_f.set_yticks(range(len(ZONE_TARGETS)))
ax_f.set_yticklabels([t.replace("_maxzone", "") for t in ZONE_TARGETS], fontsize=6)
ax_f.set_xlabel("Cycle", fontsize=8)
ax_f.set_title("F.  Cracking Zone Residual Heatmap", fontsize=9)
plt.colorbar(im, ax=ax_f, label="Max |Res| (°C)", shrink=0.7)

savefig("08_summary_panel.png")

# =============================================================================
# 8.  PRINTED SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("\n  Regression — Mean metrics across all 24 targets")
print(f"  {'Phase':<15s}  {'Mean RMSE (°C)':>15s}  {'Mean R²':>10s}")
print("  " + "-" * 45)
for phase_label, best_rows_df in [
    ("Cracking", crack_best_rows),
    ("Decoking",  decok_best_rows),
]:
    print(f"  {phase_label:<15s}  "
          f"{best_rows_df['rmse'].mean():>15.4f}  "
          f"{best_rows_df['r2'].mean():>10.6f}")

print("\n  Regression — Best model frequency")
for phase_label, results_df in [
    ("Cracking", results_table[results_table["phase"] == "cracking_ss"]),
    ("Decoking",  results_table[results_table["phase"] == "decoking_ss"]),
]:
    best_counts = (results_df[results_df["is_best"]]
                   .groupby("model")["target"].count()
                   .sort_values(ascending=False))
    print(f"  {phase_label}:")
    for mdl, cnt in best_counts.items():
        print(f"    {mdl:<25s}: {cnt:2d} targets")

print("\n  Classification — LOCO-CV mean scores")
print(f"  {'Phase':<15s}  {'Split':<8s}  {'N cycles':>9s}  {'Accuracy':>10s}  {'F1':>8s}  {'Precision':>10s}  {'Recall':>8s}")
print("  " + "-" * 75)
for loco_results, phase_label in [
    (loco_crack, "Cracking"),
    (loco_decok, "Decoking"),
]:
    for split_label, split_key in [("All", None), ("Train", "train"), ("Test", "test")]:
        subset = [r for r in loco_results
                  if split_key is None or r["split"] == split_key]
        if not subset:
            continue
        n         = len(subset)
        mean_acc  = np.mean([r["accuracy"]  for r in subset])
        mean_f1   = np.nanmean([r["f1"]       for r in subset])
        mean_prec = np.nanmean([r["precision"] for r in subset])
        mean_rec  = np.nanmean([r["recall"]    for r in subset])
        print(f"  {phase_label:<15s}  {split_label:<8s}  {n:>9d}  "
              f"{mean_acc:>10.3f}  {mean_f1:>8.3f}  "
              f"{mean_prec:>10.3f}  {mean_rec:>8.3f}")
    print()


print(f"\n  Figures saved to  : {FIGURES_DIR}")
print(f"  Results saved to  : {RESULTS_DIR}")
print("\n  PIPELINE COMPLETE")
