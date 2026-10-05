// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The variables a timeline is built from: one concept per measured
//! quantity, with the NHANES variable names it goes by across cycles (in
//! priority order), the factor to its canonical unit, and the values that are
//! real measurements rather than "refused" or "don't know" codes.
//!
//! The harmonisation is written down here rather than inferred: every rename
//! (HDL is `LBDHDL`, `LBXHDD` and `LBDHDD` in different cycles; creatinine
//! `LBDSCR` in 2001-2002), every unit change (high-sensitivity CRP from 2015
//! is reported in mg/L, the earlier assay in mg/dL) and every special code
//! (`7`, `9`, `77`, `99`, `777`, `999`, `7777`, `9999` mean refused or unknown
//! in questionnaire items) is a line a reviewer can check against the
//! codebooks.

/// A numeric concept.
pub struct Numeric {
    /// Token name.
    pub name: &'static str,
    /// `(variable, factor to the canonical unit)`, first match wins.
    pub vars: &'static [(&'static str, f64)],
    /// Valid range in the canonical unit; anything outside is a code, not a value.
    pub valid: (f64, f64),
}

/// A categorical concept: numeric codes to levels; other codes are missing.
pub struct Categorical {
    /// Token name.
    pub name: &'static str,
    /// Variable names, first match wins.
    pub vars: &'static [&'static str],
    /// Code to level.
    pub levels: &'static [(f64, &'static str)],
}

const YES_NO: &[(f64, &str)] = &[(1.0, "yes"), (2.0, "no")];

/// Measured at the examination: anthropometry, blood pressure, laboratory.
pub const EXAM: &[Numeric] = &[
    Numeric {
        name: "bmi",
        vars: &[("BMXBMI", 1.0)],
        valid: (10.0, 100.0),
    },
    Numeric {
        name: "waist_cm",
        vars: &[("BMXWAIST", 1.0)],
        valid: (40.0, 250.0),
    },
    Numeric {
        name: "height_cm",
        vars: &[("BMXHT", 1.0)],
        valid: (100.0, 230.0),
    },
    Numeric {
        name: "weight_kg",
        vars: &[("BMXWT", 1.0)],
        valid: (20.0, 350.0),
    },
    Numeric {
        name: "hba1c_pct",
        vars: &[("LBXGH", 1.0)],
        valid: (2.0, 20.0),
    },
    Numeric {
        name: "glucose_mgdl",
        vars: &[("LBXGLU", 1.0)],
        valid: (20.0, 800.0),
    },
    Numeric {
        name: "total_chol_mgdl",
        vars: &[("LBXTC", 1.0)],
        valid: (50.0, 800.0),
    },
    Numeric {
        name: "hdl_mgdl",
        vars: &[("LBDHDD", 1.0), ("LBXHDD", 1.0), ("LBDHDL", 1.0)],
        valid: (5.0, 250.0),
    },
    // hs-CRP (2015-2018) is in mg/L; the earlier CRP in mg/dL.
    Numeric {
        name: "crp_mgdl",
        vars: &[("LBXCRP", 1.0), ("LBXHSCRP", 0.1)],
        valid: (0.0, 50.0),
    },
    Numeric {
        name: "creatinine_mgdl",
        vars: &[("LBXSCR", 1.0), ("LBDSCR", 1.0)],
        valid: (0.1, 25.0),
    },
    Numeric {
        name: "albumin_gdl",
        vars: &[("LBXSAL", 1.0)],
        valid: (1.0, 7.0),
    },
    Numeric {
        name: "alt_ul",
        vars: &[("LBXSATSI", 1.0)],
        valid: (1.0, 3000.0),
    },
    Numeric {
        name: "ast_ul",
        vars: &[("LBXSASSI", 1.0)],
        valid: (1.0, 3000.0),
    },
    Numeric {
        name: "ggt_ul",
        vars: &[("LBXSGTSI", 1.0)],
        valid: (1.0, 3000.0),
    },
    Numeric {
        name: "uric_acid_mgdl",
        vars: &[("LBXSUA", 1.0)],
        valid: (0.5, 20.0),
    },
    Numeric {
        name: "bun_mgdl",
        vars: &[("LBXSBU", 1.0)],
        valid: (1.0, 200.0),
    },
    Numeric {
        name: "wbc_k",
        vars: &[("LBXWBCSI", 1.0)],
        valid: (0.5, 200.0),
    },
    Numeric {
        name: "hemoglobin_gdl",
        vars: &[("LBXHGB", 1.0)],
        valid: (3.0, 25.0),
    },
    Numeric {
        name: "platelets_k",
        vars: &[("LBXPLTSI", 1.0)],
        valid: (5.0, 2000.0),
    },
    Numeric {
        name: "rdw_pct",
        vars: &[("LBXRDW", 1.0)],
        valid: (8.0, 40.0),
    },
    Numeric {
        name: "lymphocyte_pct",
        vars: &[("LBXLYPCT", 1.0)],
        valid: (0.0, 100.0),
    },
    Numeric {
        name: "mcv_fl",
        vars: &[("LBXMCVSI", 1.0)],
        valid: (40.0, 150.0),
    },
    Numeric {
        name: "urine_albumin_ugml",
        vars: &[("URXUMA", 1.0)],
        valid: (0.0, 20000.0),
    },
    Numeric {
        name: "urine_creatinine_mgdl",
        vars: &[("URXUCR", 1.0)],
        valid: (1.0, 1000.0),
    },
    Numeric {
        name: "income_poverty_ratio",
        vars: &[("INDFMPIR", 1.0)],
        valid: (0.0, 5.0),
    },
    Numeric {
        name: "self_rated_health",
        vars: &[("HSD010", 1.0)],
        valid: (1.0, 5.0),
    },
    Numeric {
        name: "drinks_per_day",
        vars: &[("ALQ130", 1.0)],
        valid: (0.0, 30.0),
    },
    Numeric {
        name: "sleep_hours",
        vars: &[("SLD012", 1.0), ("SLD010H", 1.0)],
        valid: (1.0, 24.0),
    },
];

/// The number of prescription medicines taken in the past month: `RXDCOUNT`
/// in the multi-row prescriptions file (`RXQ_RX`), the same on each of a
/// participant's rows (zero for a participant who answered that they took
/// none, `RXDUSE` = 2); valid range.
pub const PRESCRIPTIONS: (&str, &str, (f64, f64)) = ("RXQ_RX", "RXDCOUNT", (0.0, 60.0));

/// Categorical answers at the interview and examination.
pub const CATEGORIES: &[Categorical] = &[
    Categorical {
        name: "sex",
        vars: &["RIAGENDR"],
        levels: &[(1.0, "male"), (2.0, "female")],
    },
    Categorical {
        name: "race_ethnicity",
        vars: &["RIDRETH1"],
        levels: &[
            (1.0, "mexican_american"),
            (2.0, "other_hispanic"),
            (3.0, "nh_white"),
            (4.0, "nh_black"),
            (5.0, "other"),
        ],
    },
    Categorical {
        name: "education",
        vars: &["DMDEDUC2"],
        levels: &[
            (1.0, "less_than_9th"),
            (2.0, "9_11th"),
            (3.0, "high_school"),
            (4.0, "some_college"),
            (5.0, "college"),
        ],
    },
    Categorical {
        name: "marital",
        vars: &["DMDMARTL"],
        levels: &[
            (1.0, "married"),
            (2.0, "widowed"),
            (3.0, "divorced"),
            (4.0, "separated"),
            (5.0, "never_married"),
            (6.0, "partner"),
        ],
    },
    Categorical {
        name: "told_diabetes",
        vars: &["DIQ010"],
        levels: &[(1.0, "yes"), (2.0, "no"), (3.0, "borderline")],
    },
    Categorical {
        name: "told_hypertension",
        vars: &["BPQ020"],
        levels: YES_NO,
    },
    Categorical {
        name: "told_high_cholesterol",
        vars: &["BPQ080"],
        levels: YES_NO,
    },
    Categorical {
        name: "told_heart_failure",
        vars: &["MCQ160B"],
        levels: YES_NO,
    },
    Categorical {
        name: "told_chd",
        vars: &["MCQ160C"],
        levels: YES_NO,
    },
    Categorical {
        name: "told_heart_attack",
        vars: &["MCQ160E"],
        levels: YES_NO,
    },
    Categorical {
        name: "told_stroke",
        vars: &["MCQ160F"],
        levels: YES_NO,
    },
    Categorical {
        name: "told_emphysema",
        vars: &["MCQ160G"],
        levels: YES_NO,
    },
    Categorical {
        name: "told_bronchitis",
        vars: &["MCQ160K"],
        levels: YES_NO,
    },
    Categorical {
        name: "told_cancer",
        vars: &["MCQ220"],
        levels: YES_NO,
    },
    Categorical {
        name: "told_weak_kidneys",
        vars: &["KIQ022"],
        levels: YES_NO,
    },
];

/// Ages at which a condition was first diagnosed: history events.
pub const ONSETS: &[(&str, &[&str])] = &[
    ("dx:diabetes", &["DID040", "DID040Q", "DIQ040Q"]),
    ("dx:hypertension", &["BPD035"]),
    ("dx:heart_failure", &["MCQ180B", "MCD180B"]),
    ("dx:chd", &["MCQ180C", "MCD180C"]),
    ("dx:heart_attack", &["MCQ180E", "MCD180E"]),
    ("dx:stroke", &["MCQ180F", "MCD180F"]),
    ("smoking:start", &["SMD030"]),
];

/// Plausible ages for a recalled event; top codes and refusal codes fall outside.
pub const AGE_RANGE: (f64, f64) = (1.0, 85.0);

/// Recalled body weights (pounds), as `(variable, years before the exam or
/// None for an age variable)`. `WHD120` is the weight at age 25 and
/// `WHD140` the greatest weight, at the age `WHQ150`.
pub const POUNDS_TO_KG: f64 = 0.453_592_37;
/// Valid recalled weight in pounds.
pub const POUNDS_RANGE: (f64, f64) = (50.0, 900.0);

/// The concepts of the conventional risk-score baseline: age, sex, blood
/// pressure, cholesterol, glycaemia, body mass, smoking and cardiovascular and
/// diabetes history.
pub const STANDARD_RISK_FACTORS: &[&str] = &[
    "age",
    "sex",
    "sbp",
    "dbp",
    "total_chol_mgdl",
    "hdl_mgdl",
    "hba1c_pct",
    "bmi",
    "smoking",
    "told_diabetes",
    "told_heart_attack",
    "told_chd",
    "told_stroke",
    "told_heart_failure",
];

/// The concepts of the demographic baseline.
pub const AGE_SEX: &[&str] = &["age", "sex"];
