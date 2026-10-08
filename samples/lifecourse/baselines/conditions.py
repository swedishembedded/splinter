# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""Models for prevalent conditions and undiagnosed disease (T4 and T5).

Labels come from `conditions.jsonl` (`lifecourse labels`). For each label and
each cross-validation fold, models are fitted on the training subjects whose
label is known and predict the probability for every test subject; the Rust
scorer (`lifecourse conditions`) keeps those with a known label. A label
defined by a measurement must not be predicted from that measurement, so the
inputs of a model are one of

* `nondef`: demographics, body measures, recalled weights, smoking, alcohol
  and eating times, the same for every label;
* `full`: every input except those that define the label (`EXCLUDED`);
* `conv`: the conventional risk factors except those that define the label;
* `agesex`, `agebmi`: the reference models.

Predictions: `<out>/<model>/r<repeat>-k<fold>.jsonl`, one line per test
subject, `{"subject_id": ..., "p": {label: probability}}`, and `.meta.json`
with, per undiagnosed label, the score at which a call is positive: the
threshold giving 90% specificity among training subjects, estimated on
cross-fitted predictions so that a flexible model's training fit does not
flatter it.

    python -I conditions.py --data <lifecourse dir> --out <dir> [--model NAME ...] [--jobs N]
"""
import argparse
import json
import os
import sys
import warnings

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402
import run  # noqa: E402

PREVALENT = ("diabetes", "hypertension", "kidney_markers", "anaemia", "high_cholesterol",
             "osteoporosis", "depression", "sleep_problem", "diabetes_fasting")
UNDIAGNOSED = ("undiagnosed_diabetes", "undiagnosed_hypertension", "undiagnosed_kidney_markers",
               "undiagnosed_high_cholesterol")
LABELS = PREVALENT + UNDIAGNOSED
SCREEN_SPECIFICITY = 0.9

NONDEF_NUMERIC = (
    "age", "bmi", "height_cm", "weight_kg", "waist_cm", "hist:weight_1y_ago", "hist:weight_10y_ago",
    "hist:weight_age25", "hist:weight_heaviest", "drinks_per_day", "income_poverty_ratio",
    "diet:eating_events", "diet:eating_window_h", "diet:energy_kcal", "diet:first_meal_h",
    "diet:last_meal_h", "diet:late_energy_share", "diet:recall_days",
)
NONDEF_CATEGORICAL = ("sex", "race_ethnicity", "education", "marital", "smoking")

# What defines each label (or a drug that treats it), per group of labels.
_DIABETES = {"hba1c_pct", "glucose_mgdl", "told_diabetes", "hist:years_since_dx:diabetes", "prescriptions"}
_HYPERTENSION = {"sbp", "dbp", "told_hypertension", "hist:years_since_dx:hypertension", "prescriptions"}
_KIDNEY = {"creatinine_mgdl", "urine_albumin_ugml", "urine_creatinine_mgdl", "told_weak_kidneys"}
_CHOLESTEROL = {"total_chol_mgdl", "told_high_cholesterol", "prescriptions"}
EXCLUDED = {
    "diabetes": _DIABETES, "diabetes_fasting": _DIABETES, "undiagnosed_diabetes": _DIABETES,
    "hypertension": _HYPERTENSION, "undiagnosed_hypertension": _HYPERTENSION,
    "kidney_markers": _KIDNEY, "undiagnosed_kidney_markers": _KIDNEY,
    "anaemia": {"hemoglobin_gdl"},
    "high_cholesterol": _CHOLESTEROL, "undiagnosed_high_cholesterol": _CHOLESTEROL,
    "osteoporosis": set(),
    "depression": {"phq9"},
    "sleep_problem": set(),
}
MODELS = ("agesex", "agebmi", "conv-logit", "nondef-logit", "full-logit", "nondef-hgb", "full-hgb")


def names(data):
    return [str(x) for x in data["num_names"]], [str(x) for x in data["cat_names"]]


def inputs_for(data, label, kind):
    """(numeric names, categorical names) a `kind` of model may use for `label`."""
    num, cat = names(data)
    drop = EXCLUDED[label]
    if kind == "agesex":
        return ("age",), ("sex",)
    if kind == "agebmi":
        return ("age", "bmi"), ()
    if kind == "nondef":
        return NONDEF_NUMERIC, NONDEF_CATEGORICAL
    if kind == "conv":
        return (tuple(n for n in features.STANDARD_NUMERIC if n not in drop),
                tuple(c for c in features.STANDARD_CATEGORICAL if c not in drop))
    if kind == "full":
        return tuple(n for n in num if n not in drop), tuple(c for c in cat if c not in drop)
    raise ValueError(kind)


def load_labels(data_dir, ids):
    """Per label an int8 array (1, 0, or -1 unknown) aligned with `ids`."""
    row = {str(s): i for i, s in enumerate(ids)}
    out = {label: np.full(len(ids), -1, dtype=np.int8) for label in LABELS}
    with open(os.path.join(data_dir, "conditions.jsonl")) as f:
        for line in f:
            c = json.loads(line)
            i = row.get(c["subject_id"])
            if i is None:
                continue
            for label in LABELS:
                v = c["labels"].get(label)
                if v is not None:
                    out[label][i] = int(v)
    return out


def threshold_for_specificity(score, label, specificity):
    """Lowest score at which a call is positive and at least `specificity` of the negatives are not."""
    neg = np.sort(score[label == 0])
    k = int(np.ceil(specificity * len(neg)))
    k = min(max(k, 1), len(neg))
    # Calling positive from the score above the k-th lowest negative leaves k negatives correct.
    return float(np.nextafter(neg[k - 1], np.inf))


def make_estimator(kind, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    if kind.endswith("hgb"):
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True,
                                              validation_fraction=0.15, random_state=seed)
    if kind in ("agesex", "agebmi"):
        return LogisticRegression(max_iter=1000)
    return LogisticRegressionCV(Cs=6, cv=3, max_iter=2000, scoring="neg_log_loss")


def fit_predict(data, train, test, label, model, y, seed, want_threshold):
    """(test probabilities, threshold or None) of `model` for `label`."""
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    kind = model.split("-")[0]  # which inputs: agesex, agebmi, conv, nondef or full
    known = train[y[train] >= 0]
    prep = features.Preprocessor(data, inputs_for(data, label, kind)).fit(data, known)
    x_tr, x_te = prep.transform(data, known), prep.transform(data, test)
    y_tr = y[known]
    with warnings.catch_warnings():  # scikit-learn announces defaults it will change
        warnings.simplefilter("ignore")
        est = make_estimator(model, seed).fit(x_tr, y_tr)
        p = est.predict_proba(x_te)[:, 1]
        threshold = None
        if want_threshold:
            cv = StratifiedKFold(3, shuffle=True, random_state=seed)
            oof = cross_val_predict(make_estimator(model, seed), x_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
            threshold = threshold_for_specificity(oof, y_tr, SCREEN_SPECIFICITY)
    return p, threshold


def run_one(args):
    data_dir, out_dir, model, repeat, fold = args
    target = os.path.join(out_dir, model, f"r{repeat}-k{fold}.jsonl")
    if os.path.exists(target):
        return f"{model} r{repeat} k{fold}: done already"
    timelines, partition, _ = run.paths(data_dir)
    data = features.load(timelines, partition, os.path.join(data_dir, "baselines", "_features.npz"))
    labels = load_labels(data_dir, data["ids"])
    repeats, _ = run.load_partition(partition)
    train, test = run.split(data, repeats, repeat, fold)
    seed = run.fold_seed(repeat, fold)
    probs, thresholds = {}, {}
    for label in LABELS:
        y = labels[label]
        if (y[train] == 1).sum() < 20:
            continue
        p, t = fit_predict(data, train, test, label, model, y, seed, label in UNDIAGNOSED)
        probs[label] = p
        if t is not None:
            thresholds[label] = t
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = target + ".tmp"
    with open(tmp, "w") as f:
        for i, sid in enumerate(data["ids"][test]):
            f.write(json.dumps({"subject_id": str(sid),
                                "p": {k: round(float(v[i]), 9) for k, v in probs.items()}}) + "\n")
    os.replace(tmp, target)
    with open(target[:-6] + ".meta.json", "w") as f:
        json.dump(dict(model=model, repeat=repeat, fold=fold, seed=seed, thresholds=thresholds,
                       specificity=SCREEN_SPECIFICITY, n_train=len(train), n_test=len(test)),
                  f, indent=1, sort_keys=True)
    return f"{model} r{repeat} k{fold}: done"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="directory for the model directories")
    ap.add_argument("--model", action="append", choices=MODELS)
    ap.add_argument("--repeat", action="append", type=int)
    ap.add_argument("--fold", action="append", type=int)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    if not 1 <= a.jobs <= run.MAX_JOBS:
        ap.error(f"--jobs must be between 1 and {run.MAX_JOBS}")
    _, partition, _ = run.paths(a.data)
    repeats, spec = run.load_partition(partition)
    tasks = [(a.data, a.out, m, r, k) for m in (a.model or MODELS)
             for r in (a.repeat if a.repeat is not None else range(len(repeats)))
             for k in (a.fold if a.fold is not None else range(spec["folds"]))]
    if a.jobs == 1:
        for line in map(run_one, tasks):
            print(line, flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            for line in pool.map(run_one, tasks):
                print(line, flush=True)


if __name__ == "__main__":
    main()
