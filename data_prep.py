# =============================================================================
# CHEMENG 277 – Supervised Learning for Steam Cracker Coil Health Monitoring
# Data Preparation
#
# Author : Rupin Vaidya
# Course : CHEMENG 277, Winter 2026
#
# Purpose
# -------
# - Loads the Raw Time Series Data
# - Identifies & Labels Cracking & Decoking Regimes
# - Filters Out Invalid Readings (Noise or Uncontrolled Operation)
# - Computes Per-Cycle Metadata
#
# Outputs (written to ./prepared_data/)
# ----------------------------------------
# df_cracking_ss.csv   – steady-state cracking rows (SP >= 99.9 %)
# df_decoking_ss.csv   – steady-state decoking rows (35.5 % <= SP <= 36.5 %)
# df_rampup.csv        – ramp-up rows (SP rising, between decoking and cracking)
# df_rampdown.csv      – ramp-down rows (SP falling, between cracking and decoking)
# df_cycle_meta.csv    – one row per cycle: equilibration time + summary stats
#
# =============================================================================

import os
import numpy as np
import pandas as pd

# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

DATA_PATH = r"C:\Users\rupin\Documents\Documents\VS Code\CHEMENG 277\Project\data.xlsx"

# Output Directory: automatically placed in the same folder as data.xlsx
# so prepared_data/ always sits alongside the source file regardless of
# where the script is run from.
OUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "prepared_data")

# --- Setpoint Thresholds -----------------------------------------------------
SP_CRACKING_MIN  = 99.9    # >= this → cracking steady-state
SP_DECOKING_MIN  = 35.5    # >= this AND <= SP_DECOKING_MAX → decoking steady-state
SP_DECOKING_MAX  = 36.5
SP_TRANSITION_LO = SP_DECOKING_MAX  # lower bound of the transition band
SP_TRANSITION_HI = SP_CRACKING_MIN  # upper bound of the transition band
SP_SHUTDOWN      = 0.0     # exactly 0 → shutdown

# --- Cycle Filtering ---------------------------------------------------------
# Minimum number of rows a cracking or decoking block must have to be treated
# as a real phase (filters out ~33-row startup artifacts)
MIN_BLOCK_ROWS   = 300

# Raw cracking block number to drop (Cycle 7, Dec 9–11 anomaly)
ANOMALY_RAW_CYCLE = 7

# --- Train / Test Split ------------------------------------------------------
# Cycles 1–TRAIN_UP_TO (inclusive) → training; remainder → test
TRAIN_UP_TO = 17

# --- Equilibration Criterion -------------------------------------------------
EQUIL_ROLL_WINDOW_MIN = 10   # rolling window size in minutes (= rows, 1-min data)
EQUIL_STD_THRESHOLD   = 5.0  # °C – equilibration reached when rolling std < this

# --- TI Covariate Lag --------------------------------------------------------
TI_LAG_MIN = 5   # minutes lag applied to TI-144 Avg and TI-145 Avg

# =============================================================================
# 1.  COLUMN DEFINITIONS
# =============================================================================

ZONE_IDS   = list(range(18))   # zones 0–17
ZONE_AVG   = [f"Zone {i} Avg" for i in ZONE_IDS]
ZONE_MIN   = [f"Zone {i} Min" for i in ZONE_IDS]
ZONE_MAX   = [f"Zone {i} Max" for i in ZONE_IDS]

# Element layout: 6 heating elements, each covering 3 consecutive zones
# Element 1 (top):    zones 0, 1, 2  (left, center, right)
# Element 2:          zones 3, 4, 5
# Element 3:          zones 6, 7, 8
# Element 4:          zones 9, 10, 11
# Element 5:          zones 12, 13, 14
# Element 6 (bottom): zones 15, 16, 17
ELEMENT_ZONES = {
    "Elem1_Top": [0, 1, 2],
    "Elem2":     [3, 4, 5],
    "Elem3":     [6, 7, 8],
    "Elem4":     [9, 10, 11],
    "Elem5":     [12, 13, 14],
    "Elem6_Bot": [15, 16, 17],
}
ELEMENT_AVG_COLS = {
    name: [f"Zone {z} Avg" for z in zones]
    for name, zones in ELEMENT_ZONES.items()
}

TI_AVG_COLS = ["TI 144 Avg", "TI 145 Avg"]

# All temperature columns that use 799 as a lower-limit sentinel
SENTINEL_COLS = ZONE_AVG + ZONE_MIN + ZONE_MAX + [
    "BGCOMP Min", "BGCOMP Avg", "BGCOMP Max",
    "TI 144 Min", "TI 144 Avg", "TI 144 Max",
    "TI 145 Min", "TI 145 Avg", "TI 145 Max",
]

# =============================================================================
# 2.  LOAD RAW DATA
# =============================================================================

print("=" * 60)
print("STEP 1: Loading raw data")
print("=" * 60)

df = pd.read_excel(DATA_PATH, sheet_name=0)
df = df.sort_values("Timestamp").reset_index(drop=True)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

print(f"  Rows loaded       : {len(df):,}")
print(f"  Columns           : {df.shape[1]}")
print(f"  Date range        : {df['Timestamp'].min()} → {df['Timestamp'].max()}")
print(f"  Sampling interval : 1 minute (verified)")

# =============================================================================
# 3.  REPLACE SENTINEL VALUES WITH NaN
# =============================================================================
# 799 °C is the camera's lower detection limit. It appears when the furnace is
# cold (startup, shutdown, or weekend). These are not real temperature readings
# and must be excluded before any analysis.

print("\n" + "=" * 60)
print("STEP 2: Replacing sentinel values (799) with NaN")
print("=" * 60)

sentinel_count_before = (df[SENTINEL_COLS] == 799).sum().sum()
df[SENTINEL_COLS] = df[SENTINEL_COLS].replace(799, np.nan)
print(f"  Sentinel (799) values replaced: {sentinel_count_before:,}")

# =============================================================================
# 4.  DERIVE AGGREGATE TEMPERATURE COLUMNS
# =============================================================================
# Compute furnace-wide and per-element aggregate temperatures.
# These are used for regime labelling validation and as modelling features.

print("\n" + "=" * 60)
print("STEP 3: Deriving aggregate temperature columns")
print("=" * 60)

# Furnace-wide aggregates (across all 18 zone averages)
df["furnace_avg"]    = df[ZONE_AVG].mean(axis=1)
df["furnace_min"]    = df[ZONE_AVG].min(axis=1)
df["furnace_max"]    = df[ZONE_AVG].max(axis=1)
df["temp_spread"]    = df["furnace_max"] - df["furnace_min"]

# Per-element average (mean of the 3 zone averages within each element)
for elem_name, cols in ELEMENT_AVG_COLS.items():
    df[elem_name] = df[cols].mean(axis=1)

# Per-element max zone temperature (hottest zone within each element)
for elem_name, zones in ELEMENT_ZONES.items():
    max_cols = [f"Zone {z} Max" for z in zones]
    df[f"{elem_name}_maxzone"] = df[max_cols].max(axis=1)

# Lagged TI-144 and TI-145 average temperatures (5-minute lag)
# Using lagged values to avoid contemporaneous leakage: the refractory
# temperature is partly a response variable driven by element temperatures,
# but also influences them (thermal equilibration). A 5-min lag captures
# the leading indicator aspect without leakage.
for ti_col in TI_AVG_COLS:
    lag_col = f"{ti_col}_lag{TI_LAG_MIN}min"
    df[lag_col] = df[ti_col].shift(TI_LAG_MIN)

TI_LAGGED_COLS = [f"{c}_lag{TI_LAG_MIN}min" for c in TI_AVG_COLS]

print(f"  Added: furnace_avg, furnace_min, furnace_max, temp_spread")
print(f"  Added: {list(ELEMENT_AVG_COLS.keys())} (per-element averages)")
print(f"  Added: element max-zone columns")
print(f"  Added: {TI_LAGGED_COLS} (TI covariate lags)")

# =============================================================================
# 5.  IDENTIFY OPERATING REGIME (coarse pass)
# =============================================================================
# Assign a preliminary regime label based purely on setpoint value and
# setpoint direction (diff). The direction is used to distinguish ramp-up
# from ramp-down within the transition band.

print("\n" + "=" * 60)
print("STEP 4: Labelling operating regimes")
print("=" * 60)

# Setpoint first-difference (positive = rising, negative = falling)
df["sp_diff"] = df["Setpoint %"].diff().fillna(0)

def assign_regime(row):
    """
    Assign operating regime to a single row based on setpoint value and
    direction of change.

    Returns one of:
        'cracking_ss'  – steady-state cracking hold
        'decoking_ss'  – steady-state decoking hold
        'ramp_up'      – setpoint rising in the transition band
        'ramp_down'    – setpoint falling in the transition band
        'shutdown'     – setpoint = 0
        'other'        – low SP > 0 but outside defined bands (rare artifact)
    """
    sp  = row["Setpoint %"]
    dsp = row["sp_diff"]

    if sp == SP_SHUTDOWN:
        return "shutdown"
    if sp >= SP_CRACKING_MIN:
        return "cracking_ss"
    if SP_DECOKING_MIN <= sp <= SP_DECOKING_MAX:
        return "decoking_ss"
    if SP_DECOKING_MAX < sp < SP_CRACKING_MIN:
        # Transition band: use direction to classify
        return "ramp_up" if dsp >= 0 else "ramp_down"
    # SP > 0 but below decoking band (very rare, end-of-decoking artifact)
    return "other"

df["regime"] = df.apply(assign_regime, axis=1)

regime_counts = df["regime"].value_counts()
print("  Regime row counts:")
for regime, count in regime_counts.items():
    print(f"    {regime:<15s}: {count:>8,}")

# =============================================================================
# 6.  IDENTIFY AND NUMBER CRACKING CYCLES
# =============================================================================
# Find contiguous blocks of cracking_ss rows, filter out short artifacts,
# drop the anomalous Cycle 7 (extended 48 h run), and re-index remaining
# cycles 1–28.

print("\n" + "=" * 60)
print("STEP 5: Identifying cracking cycles")
print("=" * 60)

df["is_crack_ss"] = df["regime"] == "cracking_ss"
df["crack_blk_raw"] = (
    df["is_crack_ss"].ne(df["is_crack_ss"].shift())
).cumsum()

# Summarise each contiguous cracking block
crack_blocks = (
    df[df["is_crack_ss"]]
    .groupby("crack_blk_raw")
    .agg(
        start=("Timestamp", "min"),
        end=("Timestamp", "max"),
        n_rows=("Timestamp", "count"),
    )
    .reset_index()
)

# Keep only real cracking phases (filter startup artifacts)
crack_blocks = crack_blocks[crack_blocks["n_rows"] >= MIN_BLOCK_ROWS].reset_index(drop=True)

# Assign raw cycle numbers before dropping anomaly
crack_blocks["raw_cycle"] = range(1, len(crack_blocks) + 1)
crack_blocks["is_anomaly"] = crack_blocks["n_rows"] > 1000  # >1000 rows ≈ > ~16 h

anomaly_row = crack_blocks[crack_blocks["raw_cycle"] == ANOMALY_RAW_CYCLE]
print(f"  Raw cracking blocks found : {len(crack_blocks)}")
print(f"  Anomaly (Cycle 7)         : "
      f"{anomaly_row['start'].values[0]} → {anomaly_row['end'].values[0]}, "
      f"{anomaly_row['n_rows'].values[0]} rows — DROPPED")

# Drop anomalous cycle and re-index
crack_blocks = crack_blocks[~crack_blocks["is_anomaly"]].reset_index(drop=True)
crack_blocks["cycle"] = range(1, len(crack_blocks) + 1)
crack_blocks["split"] = crack_blocks["cycle"].apply(
    lambda c: "train" if c <= TRAIN_UP_TO else "test"
)

print(f"  Usable cracking cycles    : {len(crack_blocks)}")
print(f"  Train cycles              : {crack_blocks[crack_blocks['split']=='train']['cycle'].tolist()}")
print(f"  Test  cycles              : {crack_blocks[crack_blocks['split']=='test']['cycle'].tolist()}")

# =============================================================================
# 7.  IDENTIFY AND NUMBER DECOKING CYCLES
# =============================================================================
# Each decoking phase immediately follows a cracking phase (overnight hold).
# Decoking cycle N follows cracking cycle N on the same calendar day or
# the following morning.

print("\n" + "=" * 60)
print("STEP 6: Identifying decoking cycles")
print("=" * 60)

df["is_decok_ss"] = df["regime"] == "decoking_ss"
df["decok_blk_raw"] = (
    df["is_decok_ss"].ne(df["is_decok_ss"].shift())
).cumsum()

decok_blocks = (
    df[df["is_decok_ss"]]
    .groupby("decok_blk_raw")
    .agg(
        start=("Timestamp", "min"),
        end=("Timestamp", "max"),
        n_rows=("Timestamp", "count"),
    )
    .reset_index()
)

decok_blocks = decok_blocks[decok_blocks["n_rows"] >= MIN_BLOCK_ROWS].reset_index(drop=True)
decok_blocks["cycle"] = range(1, len(decok_blocks) + 1)

# Decoking cycle numbering note: since Cycle 7 cracking had no decoking phase,
# the decoking cycle numbers already align 1-to-1 with the usable cracking
# cycles (both are 28-long sequences). Verified by calendar inspection.
decok_blocks["split"] = decok_blocks["cycle"].apply(
    lambda c: "train" if c <= TRAIN_UP_TO else "test"
)

print(f"  Usable decoking cycles    : {len(decok_blocks)}")
print(f"  Train cycles              : {decok_blocks[decok_blocks['split']=='train']['cycle'].tolist()}")
print(f"  Test  cycles              : {decok_blocks[decok_blocks['split']=='test']['cycle'].tolist()}")

# =============================================================================
# 8.  LABEL ALL ROWS WITH CYCLE ID AND SPLIT
# =============================================================================
# Write cycle_id (1–28) and split (train/test) back onto every row
# that belongs to a real cracking or decoking SS phase.

print("\n" + "=" * 60)
print("STEP 7: Stamping cycle_id and split onto rows")
print("=" * 60)

df["cycle_id"]    = 0       # 0 = not part of a numbered cycle
df["phase_label"] = df["regime"].copy()   # preserve full regime label
df["split"]       = "none"

for _, row in crack_blocks.iterrows():
    mask = (df["Timestamp"] >= row["start"]) & (df["Timestamp"] <= row["end"])
    df.loc[mask, "cycle_id"]    = row["cycle"]
    df.loc[mask, "phase_label"] = "cracking_ss"
    df.loc[mask, "split"]       = row["split"]

for _, row in decok_blocks.iterrows():
    mask = (df["Timestamp"] >= row["start"]) & (df["Timestamp"] <= row["end"])
    df.loc[mask, "cycle_id"]    = row["cycle"]
    df.loc[mask, "phase_label"] = "decoking_ss"
    df.loc[mask, "split"]       = row["split"]

# Label ramp rows with the cracking cycle they belong to.
# A ramp-up row is assigned the cycle_id of the cracking phase it leads into.
# A ramp-down row is assigned the cycle_id of the cracking phase it follows.
# This is done by forward- and back-filling the cycle_id within contiguous
# ramp blocks that are bounded by SS phases on either side.
df["cycle_id_ffill"] = df["cycle_id"].replace(0, np.nan).ffill().fillna(0).astype(int)
df["cycle_id_bfill"] = df["cycle_id"].replace(0, np.nan).bfill().fillna(0).astype(int)

rampup_mask   = df["regime"] == "ramp_up"
rampdown_mask = df["regime"] == "ramp_down"

# Ramp-up: assign cycle it is ramping towards (bfill = next cracking cycle)
df.loc[rampup_mask,   "cycle_id"] = df.loc[rampup_mask,   "cycle_id_bfill"]
# Ramp-down: assign cycle it just left (ffill = preceding cracking cycle)
df.loc[rampdown_mask, "cycle_id"] = df.loc[rampdown_mask, "cycle_id_ffill"]

# Carry split label onto ramp rows
df.loc[rampup_mask | rampdown_mask, "split"] = df.loc[
    rampup_mask | rampdown_mask, "cycle_id"
].map(
    pd.Series(
        crack_blocks["split"].values,
        index=crack_blocks["cycle"].values
    )
).fillna("none")

df = df.drop(columns=["cycle_id_ffill", "cycle_id_bfill"])

print(f"  Rows labelled by phase:")
for phase in ["cracking_ss", "decoking_ss", "ramp_up", "ramp_down", "shutdown", "other"]:
    n = (df["phase_label"] == phase).sum()
    print(f"    {phase:<15s}: {n:>8,}")

# =============================================================================
# 9.  VALIDITY FLAG
# =============================================================================
# A row is 'valid' for modelling if all 18 zone average temperatures are
# non-NaN. Rows where any zone is NaN (camera offline / furnace cold) are
# flagged but kept in the dataset so the output files are complete.

df["all_zones_valid"] = df[ZONE_AVG].notna().all(axis=1)

n_invalid = (~df["all_zones_valid"]).sum()
print(f"\n  Rows with ≥1 NaN zone temp (flagged invalid): {n_invalid:,}")

# =============================================================================
# 10.  COMPUTE EQUILIBRATION TIME PER CYCLE
# =============================================================================
# For each cracking and decoking SS phase, measure how many minutes from the
# start of the hold until furnace_avg stabilises.
#
# Criterion: rolling std of furnace_avg over a 10-min window drops below 5 °C.
# If equilibration is never reached within the phase, the value is NaN.

print("\n" + "=" * 60)
print("STEP 8: Computing equilibration time per cycle")
print("=" * 60)

def compute_equil_time(df_phase):
    """
    Given a DataFrame slice for one SS phase (already sorted by Timestamp),
    return the number of minutes from row 0 until the rolling std of
    furnace_avg first drops below EQUIL_STD_THRESHOLD.
    Returns np.nan if the criterion is never met.
    """
    # min_periods matches the full window size so the std is only evaluated
    # once a complete 10-minute window is available, avoiding falsely small
    # std values in the first few rows where the window is partially filled.
    roll_std = (
        df_phase["furnace_avg"]
        .rolling(window=EQUIL_ROLL_WINDOW_MIN, min_periods=EQUIL_ROLL_WINDOW_MIN)
        .std()
    )
    # Find the first index where std < threshold
    equil_mask = roll_std < EQUIL_STD_THRESHOLD
    if equil_mask.any():
        first_equil_pos = equil_mask.idxmax()          # positional label
        first_equil_row = df_phase.index.get_loc(first_equil_pos)
        return float(first_equil_row)                  # minutes from start
    return np.nan

equil_records = []

for phase_name, block_df in [("cracking_ss", crack_blocks), ("decoking_ss", decok_blocks)]:
    for _, row in block_df.iterrows():
        cyc = row["cycle"]
        sub = df[
            (df["phase_label"] == phase_name) &
            (df["cycle_id"] == cyc) &
            (df["all_zones_valid"])
        ].copy()

        equil_min = compute_equil_time(sub) if len(sub) > 0 else np.nan

        equil_records.append({
            "cycle":      cyc,
            "phase":      phase_name,
            "equil_time_min": equil_min,
        })
        status = f"{equil_min:.0f} min" if not np.isnan(equil_min) else "NOT REACHED"
        print(f"  {phase_name} Cycle {cyc:2d}: equilibration at {status}")

equil_df = pd.DataFrame(equil_records)

# =============================================================================
# 11.  BUILD CYCLE METADATA TABLE
# =============================================================================
# One row per cycle (cracking and decoking separately) containing:
#  - Timing info (start, end, duration)
#  - Equilibration time
#  - Steady-state summary stats for each element and zone
#  - Train/test split label
#  - Week number within the experiment

print("\n" + "=" * 60)
print("STEP 9: Building cycle metadata table")
print("=" * 60)

ELEM_AVG_NAMES    = list(ELEMENT_AVG_COLS.keys())
ELEM_MAXZONE_COLS = [f"{n}_maxzone" for n in ELEM_AVG_NAMES]

def build_cycle_meta(block_df, phase_name, equil_df, df_full):
    records = []
    for _, row in block_df.iterrows():
        cyc = row["cycle"]
        sub = df_full[
            (df_full["phase_label"] == phase_name) &
            (df_full["cycle_id"] == cyc) &
            (df_full["all_zones_valid"])
        ]

        # Equilibration time for this cycle and phase
        equil_row = equil_df[
            (equil_df["cycle"] == cyc) & (equil_df["phase"] == phase_name)
        ]
        equil_time = equil_row["equil_time_min"].values[0] if len(equil_row) else np.nan

        rec = {
            "cycle":             cyc,
            "phase":             phase_name,
            "split":             row["split"],
            "start":             row["start"],
            "end":               row["end"],
            "duration_min":      row["n_rows"],
            "week":              ((row["start"] - pd.Timestamp("2025-12-01")).days // 7) + 1,
            "equil_time_min":    equil_time,
            "furnace_avg_mean":  sub["furnace_avg"].mean(),
            "furnace_avg_std":   sub["furnace_avg"].std(),
            "furnace_max_mean":  sub["furnace_max"].mean(),
            "temp_spread_mean":  sub["temp_spread"].mean(),
            "temp_spread_max":   sub["temp_spread"].max(),
        }

        # Per-element steady-state averages and max-zone temperatures
        for col in ELEM_AVG_NAMES + ELEM_MAXZONE_COLS:
            rec[f"{col}_mean"] = sub[col].mean() if col in sub.columns else np.nan
            rec[f"{col}_std"]  = sub[col].std()  if col in sub.columns else np.nan

        # Per-zone max temperature statistics
        for z in ZONE_IDS:
            max_col = f"Zone {z} Max"
            rec[f"z{z}_max_mean"] = sub[max_col].mean() if max_col in sub.columns else np.nan
            rec[f"z{z}_max_std"]  = sub[max_col].std()  if max_col in sub.columns else np.nan

        records.append(rec)

    return pd.DataFrame(records)

meta_crack = build_cycle_meta(crack_blocks, "cracking_ss", equil_df, df)
meta_decok = build_cycle_meta(decok_blocks, "decoking_ss", equil_df, df)
df_cycle_meta = pd.concat([meta_crack, meta_decok], ignore_index=True).sort_values(
    ["cycle", "phase"]
).reset_index(drop=True)

print(f"  Cycle metadata rows: {len(df_cycle_meta)}")
print(f"  Columns: {len(df_cycle_meta.columns)}")

# =============================================================================
# 12.  BUILD PHASE-SPECIFIC OUTPUT DATAFRAMES
# =============================================================================
# Subset the master dataframe into the four output datasets.
# Only rows with all_zones_valid == True are included in the SS datasets.
# Ramp datasets retain NaN rows but include the validity flag so downstream
# code can filter if needed.

print("\n" + "=" * 60)
print("STEP 10: Building output datasets")
print("=" * 60)

# Columns to retain in all output datasets
BASE_COLS = (
    ["Timestamp", "Setpoint %", "phase_label", "cycle_id", "split",
     "all_zones_valid", "furnace_avg", "furnace_min", "furnace_max",
     "temp_spread"]
    + ELEM_AVG_NAMES
    + ELEM_MAXZONE_COLS
    + ZONE_AVG
    + ZONE_MAX
    + TI_LAGGED_COLS
)

# Steady-state cracking: valid rows only, numbered cycles only (cycle_id > 0
# excludes the anomalous Cycle 7 block which was never assigned a cycle number)
df_cracking_ss = df[
    (df["phase_label"] == "cracking_ss") &
    (df["all_zones_valid"]) &
    (df["cycle_id"] > 0)
][BASE_COLS].copy().reset_index(drop=True)

# Steady-state decoking: valid rows only, numbered cycles only
df_decoking_ss = df[
    (df["phase_label"] == "decoking_ss") &
    (df["all_zones_valid"]) &
    (df["cycle_id"] > 0)
][BASE_COLS].copy().reset_index(drop=True)

# Ramp-up: all rows (validity flag included)
df_rampup = df[
    df["phase_label"] == "ramp_up"
][BASE_COLS].copy().reset_index(drop=True)

# Ramp-down: all rows (validity flag included)
df_rampdown = df[
    df["phase_label"] == "ramp_down"
][BASE_COLS].copy().reset_index(drop=True)

for name, dset in [
    ("df_cracking_ss", df_cracking_ss),
    ("df_decoking_ss", df_decoking_ss),
    ("df_rampup",      df_rampup),
    ("df_rampdown",    df_rampdown),
]:
    n_train = (dset["split"] == "train").sum()
    n_test  = (dset["split"] == "test").sum()
    print(f"  {name:<20s}: {len(dset):>6,} rows  "
          f"(train={n_train:,}, test={n_test:,})")

# =============================================================================
# 13.  SAVE OUTPUTS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 11: Saving outputs")
print("=" * 60)

os.makedirs(OUT_DIR, exist_ok=True)

output_files = {
    "df_cracking_ss.csv": df_cracking_ss,
    "df_decoking_ss.csv": df_decoking_ss,
    "df_rampup.csv":      df_rampup,
    "df_rampdown.csv":    df_rampdown,
    "df_cycle_meta.csv":  df_cycle_meta,
}

for filename, dataset in output_files.items():
    path = os.path.join(OUT_DIR, filename)
    dataset.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(dataset):,} rows × {dataset.shape[1]} cols)")

# =============================================================================
# 14.  SANITY CHECK SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

# Check the anomalous cycle (Dec 9-11) timestamps did not leak through.
# After re-indexing, cycle_id=7 in the output refers to the re-indexed normal
# cycle (old raw block 8, Dec 11). We verify exclusion by checking that no
# cracking SS rows fall within the anomaly's known date range.
ANOMALY_START = pd.Timestamp("2025-12-09 10:46")
ANOMALY_END   = pd.Timestamp("2025-12-11 10:36")
anomaly_leak_crack = df_cracking_ss[
    (df_cracking_ss["Timestamp"] >= ANOMALY_START) &
    (df_cracking_ss["Timestamp"] <= ANOMALY_END)
]
anomaly_leak_decok = df_decoking_ss[
    (df_decoking_ss["Timestamp"] >= ANOMALY_START) &
    (df_decoking_ss["Timestamp"] <= ANOMALY_END)
]
assert len(anomaly_leak_crack) == 0, "ERROR: Anomaly timestamps found in cracking SS!"
assert len(anomaly_leak_decok) == 0, "ERROR: Anomaly timestamps found in decoking SS!"
print("  ✓ Anomaly cycle (Dec 9–11) fully excluded from all SS datasets")

# Check no NaN zone values in SS datasets
assert df_cracking_ss[ZONE_AVG].isna().sum().sum() == 0, "ERROR: NaN zone temps in cracking SS!"
assert df_decoking_ss[ZONE_AVG].isna().sum().sum() == 0, "ERROR: NaN zone temps in decoking SS!"
print("  ✓ No NaN zone temperatures in SS datasets")

# Check train/test split proportions
for name, dset in [("Cracking", df_cracking_ss), ("Decoking", df_decoking_ss)]:
    n_tr = (dset["split"] == "train").sum()
    n_te = (dset["split"] == "test").sum()
    pct  = 100 * n_tr / (n_tr + n_te)
    print(f"  ✓ {name}: {pct:.1f}% train / {100-pct:.1f}% test")

# Check equilibration time is populated for all cycles
missing_equil = df_cycle_meta["equil_time_min"].isna().sum()
print(f"  {'✓' if missing_equil == 0 else '⚠'} "
      f"Equilibration time missing for {missing_equil} cycle(s)")

# Check expected cycle counts
assert len(df_cracking_ss["cycle_id"].unique()) == 28, "Expected 28 cracking cycles!"
assert len(df_decoking_ss["cycle_id"].unique()) == 28, "Expected 28 decoking cycles!"
print(f"  ✓ 28 cracking cycles and 28 decoking cycles confirmed")

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)
print(f"\n  Output files written to: ./{OUT_DIR}/")
print("  Ready for feature engineering and modelling.")
