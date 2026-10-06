#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements evidence-grade evaluation of learned health
# models for its clients: tests that separate real signal from noise on small
# trials. If your team needs expertise in validating predictive models
# against randomised data, you can procure our services by sending an email
# to info@swedishembedded.com.
"""Does the assigned arm carry predictive signal for short-term transitions?

For each readable fasting trial, separately (trials are never pooled: the
protocols differ), the script builds a tidy participant table and then, per
outcome with enough complete cases, compares by repeated cross-validation
the out-of-sample error of predicting the end-of-intervention value from

  (a) baseline covariates only,
  (b) baseline covariates plus the assigned arm,
  (c) (b) plus arm x baseline value of the outcome,
  (d) EXPLORATORY: (b) plus observed adherence/behaviour measures.

The gain of (b) over (a) is the incremental predictive information of the
randomised assignment; it is reported with a participant bootstrap interval
next to the unadjusted randomised effect (difference in means of the change
from baseline) as the reference.

What this is not. It evaluates learned signal in small trials. It is not
evidence of a treatment policy and not a claim about lifespan. Adherence
measures are observed after randomisation, so model (d) is associational and
is never read as an assignment effect. A trial without randomisation (the
single-arm pilot) gets a table and a within-person summary but no arm model.

Usage:
    python -I fasting_transitions.py --data-dir <health resources> --out-dir <dir>
    python -I fasting_transitions.py --self-test
"""
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261006
ALPHAS = np.logspace(-2, 3, 11)
OUTER_FOLDS = 5
OUTER_REPEATS = 20
INNER_FOLDS = 5
BOOTSTRAPS = 2000
MIN_COMPLETE = 20

ROLE_KEY = "key"
ROLE_ASSIGNMENT = "assignment"
ROLE_BASELINE = "baseline_covariate"
ROLE_ADHERENCE = "observational_post_randomisation"
ROLE_OUTCOME = "outcome_end_of_intervention"

MODEL_BASE = "a_baseline"
MODEL_ARM = "b_baseline_plus_arm"
MODEL_INTERACTION = "c_arm_x_baseline_value"
MODEL_ADHERENCE = "d_exploratory_adherence_not_causal"


class LeakageError(ValueError):
    """A feature list would let an outcome, or an unlabelled variable, in."""


@dataclass
class Outcome:
    name: str
    baseline_col: str
    end_col: str


@dataclass
class Trial:
    name: str
    table: pd.DataFrame
    roles: dict
    randomised: bool
    outcomes: list
    covariates: list
    adherence: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    arm_label: str = ""


# ----------------------------------------------------------------- features


def check_features(trial, outcome, features, allow_adherence):
    """Refuse any feature that is not a baseline covariate or the assignment
    (or, for the exploratory model only, an adherence measure)."""
    allowed = {ROLE_BASELINE, ROLE_ASSIGNMENT}
    if allow_adherence:
        allowed.add(ROLE_ADHERENCE)
    for col in features:
        role = trial.roles.get(col)
        if role not in allowed:
            raise LeakageError(f"{trial.name}: feature {col!r} has role {role!r}, not allowed")
        if col == outcome.end_col:
            raise LeakageError(f"{trial.name}: the outcome {col!r} is a feature")


def design_matrix(df, base_cols, extra_cols, interaction_with, bl_center):
    """Columns in a fixed order. The arm x baseline column uses the arm coded
    -0.5/+0.5 and the baseline centred on `bl_center`, so it is not
    mechanically collinear with the arm."""
    cols = [df[c].to_numpy(float) for c in base_cols + extra_cols]
    if interaction_with is not None:
        arm = df["arm"].to_numpy(float) - 0.5
        cols.append(arm * (df[interaction_with].to_numpy(float) - bl_center))
    return np.column_stack(cols) if cols else np.zeros((len(df), 0))


def _standardise(train_x):
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def ridge_fit(x, y, alpha):
    mu, sd = _standardise(x)
    xs = (x - mu) / sd
    y_mean = y.mean()
    gram = xs.T @ xs + alpha * np.eye(xs.shape[1])
    beta = np.linalg.solve(gram, xs.T @ (y - y_mean))
    return mu, sd, beta, y_mean


def ridge_predict(model, x):
    mu, sd, beta, y_mean = model
    return ((x - mu) / sd) @ beta + y_mean


def choose_alpha(x, y, seed):
    """Alpha by inner K-fold on the training rows only."""
    n = len(y)
    if x.shape[1] == 0:
        return ALPHAS[0]
    order = np.random.RandomState(seed).permutation(n)
    folds = np.array_split(order, min(INNER_FOLDS, n))
    sse = np.zeros(len(ALPHAS))
    for fold in folds:
        keep = np.setdiff1d(order, fold)
        for i, alpha in enumerate(ALPHAS):
            pred = ridge_predict(ridge_fit(x[keep], y[keep], alpha), x[fold])
            sse[i] += np.sum((y[fold] - pred) ** 2)
    return ALPHAS[int(np.argmin(sse))]


def fit_predict(make_x, df, y, train, test, seed):
    """Fit on `train` rows (alpha from those rows only) and predict `test`.
    `make_x(rows_df)` builds the design from covariates and the arm only."""
    x_train = make_x(df.iloc[train])
    x_test = make_x(df.iloc[test])
    if x_train.shape[1] == 0:
        return np.full(len(test), y[train].mean())
    alpha = choose_alpha(x_train, y[train], seed)
    return ridge_predict(ridge_fit(x_train, y[train], alpha), x_test)


def cross_validate(models, df, y, seed=SEED, repeats=OUTER_REPEATS, folds=OUTER_FOLDS):
    """Per participant mean squared error over repeats, for every model and
    for the training-mean reference. `models` maps a name to make_x."""
    n = len(y)
    sums = {name: np.zeros(n) for name in list(models) + ["null"]}
    counts = np.zeros(n)
    for rep in range(repeats):
        order = np.random.RandomState(seed + rep).permutation(n)
        for k, test in enumerate(np.array_split(order, folds)):
            train = np.setdiff1d(order, test)
            inner_seed = seed + 1000 * rep + k
            for name, make_x in models.items():
                pred = fit_predict(make_x, df, y, train, test, inner_seed)
                sums[name][test] += (y[test] - pred) ** 2
            sums["null"][test] += (y[test] - y[train].mean()) ** 2
            counts[test] += 1
    return {name: s / counts for name, s in sums.items()}


# ---------------------------------------------------------------- statistics


def bootstrap_interval(stat, n, seed, draws=BOOTSTRAPS):
    rng = np.random.RandomState(seed)
    values = np.array([stat(rng.randint(0, n, n)) for _ in range(draws)])
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def paired_gain(se_ref, se_new, se_null, seed):
    """MSE reduction and R2 gain of `se_new` over `se_ref`, with a participant
    bootstrap interval. R2 is relative to the training-mean predictor."""
    n = len(se_ref)

    def d_mse(idx):
        return se_ref[idx].mean() - se_new[idx].mean()

    def d_r2(idx):
        return (se_ref[idx].sum() - se_new[idx].sum()) / se_null[idx].sum()

    return dict(
        mse_ref=float(se_ref.mean()),
        mse_new=float(se_new.mean()),
        mse_reduction=float(d_mse(np.arange(n))),
        mse_reduction_ci95=bootstrap_interval(d_mse, n, seed),
        r2_ref=float(1 - se_ref.sum() / se_null.sum()),
        r2_new=float(1 - se_new.sum() / se_null.sum()),
        r2_gain=float(d_r2(np.arange(n))),
        r2_gain_ci95=bootstrap_interval(d_r2, n, seed + 1),
    )


def unadjusted_effect(change, arm):
    """Welch difference in means of the change (arm 1 minus arm 0) with the
    95% interval, and the smallest difference detectable with 80% power at
    the observed spread (a power limit, not an estimate)."""
    from scipy import stats

    a, b = change[arm == 1], change[arm == 0]
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    dof = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    half = float(stats.t.ppf(0.975, dof)) * se
    diff = float(a.mean() - b.mean())
    mde = float((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se)
    return dict(
        n_arm1=int(len(a)),
        n_arm0=int(len(b)),
        diff_in_mean_change=diff,
        ci95=[diff - half, diff + half],
        sd_change=float(np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                                / (len(a) + len(b) - 2))),
        min_detectable_difference_80pct_power=mde,
    )


# ------------------------------------------------------------ per-trial runs


def analyse_outcome(trial, outcome, seed):
    df = trial.table
    covs = list(dict.fromkeys(trial.covariates + [outcome.baseline_col]))
    needed = ["arm", outcome.end_col, *covs, *trial.adherence]
    rows = df.dropna(subset=needed).reset_index(drop=True)
    result = dict(outcome=outcome.name, n_rows_in_table=int(len(df)), n_complete=int(len(rows)))
    if len(rows) < MIN_COMPLETE:
        result["skipped"] = f"fewer than {MIN_COMPLETE} complete cases"
        return result
    arm = rows["arm"].to_numpy()
    if len(np.unique(arm)) < 2:
        result["skipped"] = "one arm only among complete cases"
        return result

    check_features(trial, outcome, covs, allow_adherence=False)
    check_features(trial, outcome, ["arm"], allow_adherence=False)
    check_features(trial, outcome, trial.adherence, allow_adherence=True)
    y = rows[outcome.end_col].to_numpy(float)
    bl = outcome.baseline_col

    centre = rows[bl].mean()  # a covariate-only constant: no outcome information

    def make(extra, interaction):
        return lambda part: design_matrix(part, covs, extra, interaction, centre)

    models = {MODEL_BASE: make([], None), MODEL_ARM: make(["arm"], None),
              MODEL_INTERACTION: make(["arm"], bl)}
    if trial.adherence:
        models[MODEL_ADHERENCE] = make(["arm"] + trial.adherence, None)
    return finish_outcome(result, trial, outcome, rows, y, arm, models, seed)


def finish_outcome(result, trial, outcome, rows, y, arm, models, seed):
    se = cross_validate(models, rows, y, seed)
    result["models"] = {
        name: dict(mse=float(se[name].mean()), r2=float(1 - se[name].sum() / se["null"].sum()))
        for name in models
    }
    result["reference_mean_predictor_mse"] = float(se["null"].mean())
    result["arm_gain"] = paired_gain(se[MODEL_BASE], se[MODEL_ARM], se["null"], seed + 11)
    result["interaction_gain_over_arm"] = paired_gain(
        se[MODEL_ARM], se[MODEL_INTERACTION], se["null"], seed + 12)
    if MODEL_ADHERENCE in se:
        result["exploratory_adherence_gain_over_arm"] = paired_gain(
            se[MODEL_ARM], se[MODEL_ADHERENCE], se["null"], seed + 13)
    change = rows[outcome.end_col].to_numpy(float) - rows[outcome.baseline_col].to_numpy(float)
    result["unadjusted_effect"] = unadjusted_effect(change, arm)

    # Negative control: the same pipeline on a shuffled outcome must show no
    # skill and no arm information.
    shuffled = y[np.random.RandomState(seed + 99).permutation(len(y))]
    control = cross_validate({MODEL_BASE: models[MODEL_BASE], MODEL_ARM: models[MODEL_ARM]},
                             rows, shuffled, seed, repeats=max(2, OUTER_REPEATS // 4))
    result["permuted_outcome_control"] = dict(
        r2_baseline=float(1 - control[MODEL_BASE].sum() / control["null"].sum()),
        r2_gain_of_arm=float((control[MODEL_BASE].sum() - control[MODEL_ARM].sum())
                             / control["null"].sum()))
    result["detectable"] = dict(
        predictive_arm_signal=bool(result["arm_gain"]["mse_reduction_ci95"][0] > 0),
        randomised_effect=bool(not (result["unadjusted_effect"]["ci95"][0] <= 0
                                    <= result["unadjusted_effect"]["ci95"][1])),
    )
    return result


def analyse_trial(trial, seed=SEED):
    out = dict(trial=trial.name, randomised=trial.randomised, notes=trial.notes,
               n_participants=int(len(trial.table)), arm_definition=trial.arm_label,
               covariates=trial.covariates, adherence_columns=trial.adherence)
    if trial.randomised:
        out["arm_counts"] = {str(k): int(v) for k, v in trial.table["arm"].value_counts().items()}
        out["outcomes"] = [analyse_outcome(trial, o, seed) for o in trial.outcomes]
    else:
        out["outcomes"] = [within_person_change(trial, o) for o in trial.outcomes]
        out["arm_models"] = "not run: no assigned comparison arm"
    return out


def within_person_change(trial, outcome):
    """Single-arm summary: mean change with its interval. There is no control,
    so this is not an effect of the intervention."""
    from scipy import stats

    d = trial.table[[outcome.baseline_col, outcome.end_col]].dropna()
    result = dict(outcome=outcome.name, n_complete=int(len(d)),
                  n_rows_in_table=int(len(trial.table)))
    if len(d) < MIN_COMPLETE:
        result["skipped"] = f"fewer than {MIN_COMPLETE} complete cases"
        return result
    change = (d[outcome.end_col] - d[outcome.baseline_col]).to_numpy(float)
    half = float(stats.t.ppf(0.975, len(d) - 1) * change.std(ddof=1) / np.sqrt(len(d)))
    result["mean_change_within_person"] = float(change.mean())
    result["ci95"] = [float(change.mean() - half), float(change.mean() + half)]
    result["interpretation"] = "uncontrolled; completers only; not an intervention effect"
    return result


# ------------------------------------------------------------------- loaders


def _bath_lookup(raw, timepoint, name):
    top = [None if pd.isna(v) else str(v).strip() for v in raw.iloc[0]]
    sub = [None if pd.isna(v) else str(v).strip() for v in raw.iloc[1]]
    hits = [i for i in range(raw.shape[1]) if top[i] == timepoint and sub[i] == name]
    if not hits:
        raise KeyError(f"bath column not found: {timepoint!r} {name!r}")
    return pd.to_numeric(raw.iloc[2:, hits[0]], errors="coerce").reset_index(drop=True)


BATH_OUTCOMES = [
    ("mass_kg", "Mass (kg)", "Mass (kg)"),
    ("bmi", "BMI (kg/m2)", "BMI (kg/m2)"),
    ("waist_cm", "Waist Circ (cm)", "Waist Circ (cm)"),
    ("fat_mass_kg", "Fat Mass (kg)", "Fat Mass (kg)"),
    ("fat_free_mass_kg", "Fat-Free Mass (kg)", "Fat-Free Mass (kg)"),
    ("total_cholesterol_mmol", "Total Cholesterol (mmol/l)", "Total Cholesterol (mmol/l)"),
    ("hdl_mmol", "HDL Cholesterol (mmol/l)", "HDL Cholesterol (mmol/l)"),
    ("ldl_mmol", "LDL Cholesterol (mmol/l)", "LDL Cholesterol (mmol/l)"),
    ("triacylglycerol_mmol", "Triacylglycerol (mmol/l)", "Triacylglycerol (mmol/l)"),
    ("glucose_mmol", "Glucose (mmol/l)", "Glucose (mmol/l)"),
    ("insulin_pmol", "Insulin (pmol/l)", "Insulin (pmol/l)"),
    ("homa_ir", "HOMA2 (IR)", "HOMA-IR"),
    ("leptin_ug_l", "Leptin (ug/L)", "Leptin (ug/L)"),
    ("adiponectin_mg_l", "Adiponectin (mg/l)", "Adiponectin (mg/l)"),
    ("nefa_mmol", "Non-Esterified Fatty Acids (mmol/l)", "Non-Esterified Fatty Acids (mmol/l)"),
]
BATH_COVARIATES = ["age_y", "sex_female", "height_m", "b_mass_kg", "b_glucose_mmol",
                   "b_triacylglycerol_mmol", "b_total_cholesterol_mmol"]


def load_bath(path):
    """Bath 3-week trial: calorie restriction (CR) versus two intermittent
    fasting arms (ADF+CR, ADF-CR), 12 each. The arm is IF (either) versus CR;
    the raw group is kept as an assignment column. 'Pre' is the measurement at
    the start of the intervention and is the baseline value; the sparse
    screening 'Baseline' column is not used."""
    raw = pd.read_excel(path, header=None)
    ids = raw.iloc[2:, 1]
    keep = ids.notna().to_numpy()
    group = raw.iloc[2:, 0].reset_index(drop=True)
    t = pd.DataFrame({ROLE_KEY: [f"bath-{i:03d}" for i in range(len(group))]})
    t["group_raw"] = group
    t["arm"] = (group != "CR").astype(int)
    t["age_y"] = _bath_lookup(raw, None, "Age (y)")
    t["sex_female"] = (_bath_lookup(raw, None, "Sex (M1; F2)") == 2).astype(int)
    t["height_m"] = _bath_lookup(raw, None, "Height (m)")
    roles = {"arm": ROLE_ASSIGNMENT, "group_raw": ROLE_ASSIGNMENT, "age_y": ROLE_BASELINE,
             "sex_female": ROLE_BASELINE, "height_m": ROLE_BASELINE, ROLE_KEY: ROLE_KEY}
    outcomes = []
    for short, pre, post in BATH_OUTCOMES:
        t[f"b_{short}"] = _bath_lookup(raw, "Pre", pre)
        t[f"e_{short}"] = _bath_lookup(raw, "Post", post)
        roles[f"b_{short}"], roles[f"e_{short}"] = ROLE_BASELINE, ROLE_OUTCOME
        outcomes.append(Outcome(short, f"b_{short}", f"e_{short}"))
    t["obs_energy_intake_kcal_d"] = _bath_lookup(raw, "Intervention", "Total Energy Intake (kcal/d)")
    roles["obs_energy_intake_kcal_d"] = ROLE_ADHERENCE
    t = t[keep].reset_index(drop=True)
    return Trial(
        "bath_if_rct", t, roles, True, outcomes, BATH_COVARIATES,
        ["obs_energy_intake_kcal_d"],
        ["arm 1 = ADF+CR or ADF-CR pooled; the two fasting protocols differ",
         "baseline value = the Pre measurement",
         "energy intake during the intervention is observed, not assigned"],
        "1 = intermittent fasting (ADF+CR or ADF-CR), 0 = calorie restriction (CR)")


TIMET_OUTCOMES = [
    ("hba1c_pct", "Baseline_GlycatedHemoglobin, %", "End_Intervention_GlycatedHemoglobin, %"),
    ("glucose_mg_dl", "Baseline_Glucose, mgdL", "End_Intervention_Glucose, mgdL"),
    ("fasting_insulin", "Baseline_Fasting_Insulin, mlU/L", "End_Intervention_FastingInsulin, mIU/L"),
    ("homa_ir", "Baseline_HOMA_IR, A.U.", "End_Intervention_HOMA_IR, A.U."),
    ("cgm_mean_glucose", "Baseline_MeanGlucose,mgdL", "End_Intervention_MeanGlucose,mgdL"),
    ("cgm_conga", "Baseline_CONGA, A.U.", "End_Intervention_CONGA, A.U."),
    ("cgm_modd", "Baseline_MODD, mg/dL", "End_Interventionn_MODD, mg/dL"),
    ("weight_kg", "Baseline_Weight, kg", "End_Intervention_Weight,kg"),
    ("bmi", "Baseline_BMI, kg/m2", "End_Intervention_BMI, kg/m2"),
    ("total_fat_pct", "Baseline_Total_Fat_percent", "End_Intervention_Total_Fat_percent"),
    ("trunk_fat_pct", "Baseline_Trunk_Fat_percent", "End_Intervention_Trunk_Fat_percent"),
    ("fat_mass_g", "Baseline_Total_FatMass, g", "End_Intervention_Total_FatMass, g"),
    ("lean_mass_g", "Baseline_Total_LeanMass, g", "End_Intervention_Total_LeanMass, g"),
    ("bmc_g", "Baseline_Total_BMC, g", "End_Intervention_Total_BMC, g"),
    ("ldl_direct", "Baseline_LDLCholesterolDirect, mg/dL", "End_Intervention_LDLCholesterolDirect, mg/dL"),
    ("ldl_particles", "Baseline_LDLParticleNumberbyNMR, nmol/L", "End_Intervention_LDLParticleNumberbyNMR, nmol/L"),
    ("hdl", "Baseline_HDLCholesterol, mg/dL", "End_Intervention_HDLCholesterol, mg/dL"),
    ("triglycerides", "Baseline_Triglycerides, mg/dL", "End_Intervention_Triglycerides, mg/dL"),
    ("hs_crp", "Baseline_hs-CRP, mg/L", "End_Intervention_hs-CRP, mg/L"),
    ("sbp", "Baseline_SystolicBloodPressure, mm/Hg", "End_Intervention_SystolicBloodPressure, mm/Hg"),
    ("dbp", "Baseline_DiastolicBloodPressure, mm/Hg", "End_Intervention_DiastolicBloodPressure, mm/Hg"),
]
# Baseline values that are observed for every completer, used as covariates
# for every outcome (an outcome's own baseline is always added).
TIMET_PANEL = {
    "b_weight_kg": "weight_kg", "b_hba1c_pct": "hba1c_pct", "b_glucose_mg_dl": "glucose_mg_dl",
    "b_homa_ir": "homa_ir", "b_sbp": "sbp", "b_ldl_direct": "ldl_direct",
    "b_triglycerides": "triglycerides",
}
TIMET_ADHERENCE = {
    "obs_eating_window_h_full": "mCC_data_Full_Intervention_95% eating_window, hours",
    "obs_eating_window_change_h_full": "mCC_data_delta_95% eating_window_fullintervention, hours",
    "obs_days_outside_window_full": "mCC_data_days_outside_eating_window_Full_intervention, days",
    "obs_logged_entries_full": "mCC_data_Full_Intervention_numberof_caloric_entried",
}


def load_timet(path):
    """TIMET: time-restricted eating (TRE) versus standard of care (SOC). The
    analysed rows are the completers; the 14 randomised participants who did
    not complete have no end values and are counted, not imputed."""
    raw = pd.read_excel(path)
    randomised = raw[raw["RandomizationGroup"].isin(["TRE", "SOC"])]
    done = randomised[randomised["StudyPhase"] == "Completed"].reset_index(drop=True)
    t = pd.DataFrame({ROLE_KEY: [f"timet-{i:03d}" for i in range(len(done))]})
    t["arm"] = (done["RandomizationGroup"] == "TRE").astype(int)
    t["age_y"] = done["Age"].astype(float)
    t["sex_female"] = (done["Gender"] == "F").astype(int)
    for flag in ("one_plus_med", "Statin", "Antihypertensive"):
        t[f"med_{flag.lower()}"] = done[flag].astype(float)
    roles = {"arm": ROLE_ASSIGNMENT, ROLE_KEY: ROLE_KEY, "age_y": ROLE_BASELINE,
             "sex_female": ROLE_BASELINE, "med_one_plus_med": ROLE_BASELINE,
             "med_statin": ROLE_BASELINE, "med_antihypertensive": ROLE_BASELINE}
    outcomes = []
    for short, base, end in TIMET_OUTCOMES:
        t[f"b_{short}"], t[f"e_{short}"] = done[base].astype(float), done[end].astype(float)
        roles[f"b_{short}"], roles[f"e_{short}"] = ROLE_BASELINE, ROLE_OUTCOME
        outcomes.append(Outcome(short, f"b_{short}", f"e_{short}"))
    for short, col in TIMET_ADHERENCE.items():
        t[short], roles[short] = done[col].astype(float), ROLE_ADHERENCE
    covs = ["age_y", "sex_female", "med_one_plus_med", "med_statin", "med_antihypertensive",
            *TIMET_PANEL]
    return Trial(
        "timet", t, roles, True, outcomes, covs, list(TIMET_ADHERENCE),
        [f"randomised to TRE or SOC: {len(randomised)}; completers analysed: {len(done)}; "
         f"{len(randomised) - len(done)} randomised non-completers have no end values "
         "(completers-only analysis: attrition can bias the comparison)",
         "eating-window columns come from an app logged by both arms and are observed, not assigned"],
        "1 = time-restricted eating (TRE), 0 = standard of care (SOC)")


QM_OUTCOMES = [
    ("weight_kg", "BL_WGHT", "WK12_WGHT"),
    ("sbp", "BP_BL_SYS", "BP_12_SYS"),
    ("dbp", "BP_BL_DIA", "BP_12_DIA"),
    ("total_cholesterol_mmol", "BL_CHO", "WK12_CHO"),
    ("hdl_mmol", "BL_HDL", "WK12_HDL"),
    ("ldl_mmol", "BL_LDL", "WK12_LDL"),
    ("triglycerides_mmol", "BL_TRI", "WK12_TRI"),
]


def load_queen_mary_tre(path):
    """Queen Mary TRE pilot: every participant was asked to follow TRE (single
    arm; the protocol describes a pilot, not a randomised comparison). There is
    no assigned comparison, so the table is built but no arm model is run."""
    import pyreadstat

    df, _ = pyreadstat.read_sav(str(path))
    t = pd.DataFrame({ROLE_KEY: [f"qmtre-{i:03d}" for i in range(len(df))]})
    t["arm"] = 1
    t["age_y"] = df["AGE"].astype(float)
    t["sex_female"] = (df["GENDER"] == 2).astype(float)
    t["height_cm"] = df["BL_HGT"].astype(float)
    roles = {"arm": ROLE_ASSIGNMENT, ROLE_KEY: ROLE_KEY, "age_y": ROLE_BASELINE,
             "sex_female": ROLE_BASELINE, "height_cm": ROLE_BASELINE}
    outcomes = []
    for short, base, end in QM_OUTCOMES:
        t[f"b_{short}"], t[f"e_{short}"] = df[base].astype(float), df[end].astype(float)
        roles[f"b_{short}"], roles[f"e_{short}"] = ROLE_BASELINE, ROLE_OUTCOME
        outcomes.append(Outcome(short, f"b_{short}", f"e_{short}"))
    t["obs_days_completed_wk12"] = df["WK12_DAYS_COM"].astype(float)
    t["obs_adherent_wk12"] = (df["WK12_ADH"] == 1).astype(float)
    roles["obs_days_completed_wk12"] = roles["obs_adherent_wk12"] = ROLE_ADHERENCE
    return Trial(
        "queen_mary_tre", t, roles, False, outcomes, ["age_y", "sex_female", "height_cm"],
        ["obs_days_completed_wk12", "obs_adherent_wk12"],
        ["single-arm pilot: arm column is constant and is NOT a randomised assignment",
         "end values exist only for those who attended week 12"],
        "everyone asked to follow TRE (no comparison arm)")


# -------------------------------------------------------------------- output


def column_dictionary(trial):
    return {c: trial.roles.get(c, "unlabelled") for c in trial.table.columns}


def fmt_ci(ci):
    return f"[{ci[0]:.3g}, {ci[1]:.3g}]"


def render_report(results):
    lines = ["# Fasting trials: does the assigned arm carry predictive signal?", "",
             "Each trial is analysed alone. This evaluates learned signal in small trials. It is "
             "not evidence of a treatment policy and not a claim that fasting changes lifespan.",
             "",
             f"Repeated {OUTER_FOLDS}-fold cross-validation, {OUTER_REPEATS} repeats, fixed seed "
             f"{SEED}; ridge alpha by inner {INNER_FOLDS}-fold CV on training rows only; "
             f"participant bootstrap ({BOOTSTRAPS} draws) for the intervals (level 0.95). "
             "R2 is relative to the training-mean predictor. A positive gain means the arm "
             "improved out-of-sample prediction. Intervals are not adjusted for the number of "
             "outcomes: with many outcomes a few intervals exclude zero by chance.", ""]
    for r in results["trials"]:
        lines += [f"## {r['trial']}", "", f"Randomised assignment: {r['randomised']}. "
                  f"Participants: {r['n_participants']}. Arm: {r['arm_definition']}.", ""]
        lines += [f"- {n}" for n in r["notes"]] + [""]
        if not r["randomised"]:
            lines += ["No arm model was run. Within-person change among those with both values "
                      "(uncontrolled, not an intervention effect):", "",
                      "| outcome | n | mean change | CI |", "|---|---|---|---|"]
            for o in r["outcomes"]:
                if "skipped" in o:
                    lines.append(f"| {o['outcome']} | {o['n_complete']} | skipped | {o['skipped']} |")
                else:
                    lines.append(f"| {o['outcome']} | {o['n_complete']} | "
                                 f"{o['mean_change_within_person']:.3g} | {fmt_ci(o['ci95'])} |")
            lines.append("")
            continue
        lines += [f"Arm counts: {r['arm_counts']}.", "",
                  "| outcome | n | R2 baseline | arm R2 gain [CI] | arm MSE reduction [CI] "
                  "| arm x baseline gain over arm [CI] | unadjusted change diff [CI] "
                  "| min detectable diff (power 0.8) | signal |", "|" + "---|" * 9]
        for o in r["outcomes"]:
            if "skipped" in o:
                lines.append(f"| {o['outcome']} | {o['n_complete']} | skipped: {o['skipped']} |"
                             + " |" * 6)
                continue
            g, ia, u = o["arm_gain"], o["interaction_gain_over_arm"], o["unadjusted_effect"]
            d = o["detectable"]
            label = ("predictive+randomised" if d["predictive_arm_signal"] and d["randomised_effect"]
                     else "predictive only" if d["predictive_arm_signal"]
                     else "randomised only" if d["randomised_effect"] else "none detected")
            lines.append(
                f"| {o['outcome']} | {o['n_complete']} | {g['r2_ref']:.3g} | {g['r2_gain']:.3g} "
                f"{fmt_ci(g['r2_gain_ci95'])} | {g['mse_reduction']:.3g} "
                f"{fmt_ci(g['mse_reduction_ci95'])} | {ia['r2_gain']:.3g} "
                f"{fmt_ci(ia['r2_gain_ci95'])} | {u['diff_in_mean_change']:.3g} {fmt_ci(u['ci95'])} "
                f"| {u['min_detectable_difference_80pct_power']:.3g} | {label} |")
        lines += ["", "Exploratory adherence model (observed after randomisation; associational, "
                  "never an assignment effect), R2 gain over the arm model:", "",
                  "| outcome | R2 gain [CI] |", "|---|---|"]
        for o in r["outcomes"]:
            if "exploratory_adherence_gain_over_arm" in o:
                g = o["exploratory_adherence_gain_over_arm"]
                lines.append(f"| {o['outcome']} | {g['r2_gain']:.3g} {fmt_ci(g['r2_gain_ci95'])} |")
        done = [o for o in r["outcomes"] if "arm_gain" in o]
        worst = max(o["permuted_outcome_control"]["r2_gain_of_arm"] for o in done)
        lines += ["", f"Negative control (shuffled outcomes): largest arm R2 gain {worst:.3g}; "
                  "values near zero or below mean the pipeline does not manufacture signal.", ""]
    lines += ["## Reading", "",
              "A detected predictive gain says the arm label improves out-of-sample prediction of "
              "that outcome in that trial. It does not establish a policy, does not transfer "
              "across protocols, and says nothing about lifespan. Samples are small, intervals "
              "are wide, and the minimum detectable difference column states what each trial "
              "could not have seen. Outcomes within a trial are correlated and tested many times, so "
              "one isolated detected outcome is weak evidence.", ""]
    return "\n".join(lines)


def run(data_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(data_dir)
    trials = [
        load_bath(data / "bath_if_rct" / "Lean Archive IFP.xlsx"),
        load_timet(data / "timet" / "TIMET_AnnalsofInternalMed_Table_SourceData_2024.xlsx"),
        load_queen_mary_tre(data / "queen_mary_tre" / "pone.0246186.s002.sav"),
    ]
    results = dict(seed=SEED, cv=dict(folds=OUTER_FOLDS, repeats=OUTER_REPEATS,
                                      inner_folds=INNER_FOLDS, alphas=ALPHAS.tolist(),
                                      bootstraps=BOOTSTRAPS, min_complete=MIN_COMPLETE),
                   trials=[])
    for trial in trials:
        trial.table.to_csv(out / f"{trial.name}_table.csv", index=False)
        (out / f"{trial.name}_columns.json").write_text(
            json.dumps(column_dictionary(trial), indent=2) + "\n")
        results["trials"].append(analyse_trial(trial))
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "report.md").write_text(render_report(results))
    return results


# ----------------------------------------------------------------- self-test


def _synthetic(n, arm_effect, seed):
    rng = np.random.RandomState(seed)
    t = pd.DataFrame({ROLE_KEY: [f"s{i}" for i in range(n)], "arm": rng.randint(0, 2, n)})
    t["b_x"] = rng.normal(size=n)
    t["age_y"] = rng.normal(50, 10, n)
    t["e_x"] = 0.7 * t["b_x"] + arm_effect * t["arm"] + rng.normal(scale=0.5, size=n)
    t["obs_adh"] = t["e_x"] + rng.normal(size=n)  # would leak the outcome if used as assigned
    roles = {ROLE_KEY: ROLE_KEY, "arm": ROLE_ASSIGNMENT, "b_x": ROLE_BASELINE,
             "age_y": ROLE_BASELINE, "e_x": ROLE_OUTCOME, "obs_adh": ROLE_ADHERENCE}
    return Trial("synthetic", t, roles, True, [Outcome("x", "b_x", "e_x")], ["age_y"], ["obs_adh"])


def self_test():
    effect = analyse_outcome(_synthetic(80, 1.0, 1), Outcome("x", "b_x", "e_x"), SEED)
    assert effect["arm_gain"]["mse_reduction_ci95"][0] > 0, "a real arm effect must be detected"
    assert effect["detectable"]["randomised_effect"]
    null = analyse_outcome(_synthetic(80, 0.0, 2), Outcome("x", "b_x", "e_x"), SEED)
    assert null["arm_gain"]["mse_reduction_ci95"][0] <= 0 <= null["arm_gain"]["mse_reduction_ci95"][1], \
        "no arm effect: the interval must cover zero"
    assert effect["permuted_outcome_control"]["r2_baseline"] < 0.1

    small = analyse_outcome(_synthetic(MIN_COMPLETE - 1, 1.0, 3), Outcome("x", "b_x", "e_x"), SEED)
    assert "skipped" in small, "fewer than the minimum complete cases must be skipped"

    gap = _synthetic(60, 1.0, 4)
    gap.table.loc[:9, "e_x"] = np.nan
    counted = analyse_outcome(gap, Outcome("x", "b_x", "e_x"), SEED)
    assert counted["n_complete"] == 50, "missing outcomes are dropped and counted, never imputed"

    leaky = _synthetic(60, 1.0, 5)
    for bad in (["e_x"], ["obs_adh"]):
        try:
            check_features(leaky, leaky.outcomes[0], bad, allow_adherence=False)
        except LeakageError:
            continue
        raise AssertionError(f"{bad} must be refused as an assignment-conditioned feature")
    check_features(leaky, leaky.outcomes[0], ["obs_adh"], allow_adherence=True)

    # Held-out outcomes never influence the fitted model or its alpha.
    base = _synthetic(60, 1.0, 6)
    rows = base.table
    y = rows["e_x"].to_numpy(float)
    make = lambda part: design_matrix(part, ["age_y", "b_x"], ["arm"], None, 0.0)  # noqa: E731
    train, test = np.arange(0, 45), np.arange(45, 60)
    first = fit_predict(make, rows, y, train, test, 7)
    y2 = y.copy()
    y2[test] += 100.0
    assert np.array_equal(first, fit_predict(make, rows, y2, train, test, 7)), "test outcomes leaked"

    single = analyse_trial(Trial("single", rows, base.roles, False, base.outcomes, ["age_y"]))
    assert single["arm_models"].startswith("not run"), "no arm model without randomisation"
    print("fasting_transitions self-test: OK")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", help="directory holding bath_if_rct, timet, queen_mary_tre")
    parser.add_argument("--out-dir", help="directory for tables, results.json, report.md")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not (args.data_dir and args.out_dir):
        parser.error("--data-dir and --out-dir are required")
    run(args.data_dir, args.out_dir)
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
