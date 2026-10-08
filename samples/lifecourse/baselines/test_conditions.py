# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""Specification of the condition models: python -I test_conditions.py"""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conditions  # noqa: E402
import features  # noqa: E402

NUM = ["age", "bmi", "hba1c_pct", "glucose_mgdl", "sbp", "dbp", "prescriptions", "total_chol_mgdl",
       "creatinine_mgdl", "hemoglobin_gdl", "phq9", "waist_cm", "hist:years_since_dx:diabetes",
       "hist:weight_heaviest", "diet:energy_kcal", "hdl_mgdl", "height_cm", "weight_kg",
       "drinks_per_day", "income_poverty_ratio", "urine_albumin_ugml", "urine_creatinine_mgdl",
       "hist:years_since_dx:hypertension"]
CAT = ["sex", "smoking", "told_diabetes", "told_hypertension", "told_weak_kidneys",
       "told_high_cholesterol", "race_ethnicity", "education", "marital", "told_heart_attack",
       "told_chd", "told_stroke", "told_heart_failure"]


# Every name a model set can ask for must exist: an unknown name is an error, not a silent omission.
NUM = list(dict.fromkeys(NUM + list(conditions.NONDEF_NUMERIC) + list(features.STANDARD_NUMERIC)))
CAT = list(dict.fromkeys(CAT + list(conditions.NONDEF_CATEGORICAL) + list(features.STANDARD_CATEGORICAL)))


def world(n=3000, seed=4):
    rng = np.random.RandomState(seed)
    num = rng.normal(size=(n, len(NUM))).astype(np.float32)
    cat = rng.randint(0, 2, size=(n, len(CAT))).astype(np.int8)
    levels = {c: ["a", "b"] for c in CAT}
    return dict(num=num, cat=cat, num_names=np.array(NUM), cat_names=np.array(CAT),
                cat_levels=np.array(json.dumps(levels)), ids=np.array([f"s{i}" for i in range(n)]))


class Conditions(unittest.TestCase):
    def test_no_column_that_defines_a_label_reaches_a_model_of_it(self):
        data = world()
        rows = np.arange(len(data["num"]))
        for label, drop in conditions.EXCLUDED.items():
            for kind in ("full", "conv", "nondef"):
                names = conditions.inputs_for(data, label, kind)
                seen = set(names[0]) | set(names[1])
                self.assertFalse(seen & drop, (label, kind, seen & drop))
                prep = features.Preprocessor(data, names).fit(data, rows)
                design = " ".join(prep.names(data))
                for column in drop:
                    self.assertNotIn(column, design, (label, kind))

    def test_the_full_set_keeps_everything_else(self):
        data = world()
        num, cat = conditions.inputs_for(data, "diabetes", "full")
        self.assertIn("sbp", num)
        self.assertNotIn("hba1c_pct", num)
        self.assertIn("told_hypertension", cat)
        self.assertNotIn("told_diabetes", cat)

    def test_a_label_shuffled_among_the_subjects_cannot_be_predicted(self):
        from sklearn.metrics import roc_auc_score
        data = world()
        rng = np.random.RandomState(0)
        y = (data["num"][:, 1] + rng.normal(size=len(data["num"])) > 0.5).astype(np.int8)  # depends on bmi
        train, test = np.arange(0, 2200), np.arange(2200, 3000)
        p, _ = conditions.fit_predict(data, train, test, "osteoporosis", "full-logit", y, 1, False)
        self.assertGreater(roc_auc_score(y[test], p), 0.7)
        # Shuffled among all subjects, train and test alike, nothing is left to find. (Shuffling
        # only the training labels would leave a random direction scored against real labels.)
        shuffled = rng.permutation(y)
        p, _ = conditions.fit_predict(data, train, test, "osteoporosis", "full-logit", shuffled, 1, False)
        self.assertLess(abs(roc_auc_score(shuffled[test], p) - 0.5), 0.06)

    def test_unknown_labels_are_not_trained_on(self):
        from sklearn.metrics import roc_auc_score
        data = world()
        rng = np.random.RandomState(1)
        y = (data["num"][:, 1] + rng.normal(size=len(data["num"])) > 0).astype(np.int8)
        train, test = np.arange(0, 2200), np.arange(2200, 3000)
        y[train[::2]] = -1  # half of the training labels unknown: not read as "no"
        p, _ = conditions.fit_predict(data, train, test, "osteoporosis", "agebmi", y, 1, False)
        self.assertGreater(roc_auc_score(y[test], p), 0.65)

    def test_the_screening_threshold_gives_the_stated_specificity_on_the_data_it_came_from(self):
        rng = np.random.RandomState(3)
        score = rng.uniform(size=5000)
        label = (rng.uniform(size=5000) < score).astype(int)
        t = conditions.threshold_for_specificity(score, label, 0.9)
        self.assertGreaterEqual(np.mean(score[label == 0] < t), 0.9)
        self.assertLess(np.mean(score[label == 0] < t), 0.905)


if __name__ == "__main__":
    unittest.main()
