// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the linked mortality file says about a death beyond the three cause
//! groups of the timelines, kept in a sidecar (`causes.jsonl`) keyed by
//! subject so the frozen `timelines.jsonl` stays byte for byte as it was.
//!
//! Swedish Embedded AB implements validation of risk models whose outcomes
//! arrive years later, censored and competing, for its clients. If your team
//! needs expertise in reading cause-of-death linkages without overclaiming,
//! you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Per decedent: the leading underlying-cause recode (1-10) and the two
//! multiple-cause flags the file carries, diabetes and hypertension listed
//! anywhere on the certificate. A flag is a mention, not a diagnosis. Where
//! the file has no multiple-cause data the flags are absent and the death is
//! not evaluable for a flag outcome ([`Cause::mcod_available`] is false), which
//! is different from a flag that is false.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::nhanes::Mortality;

/// One subject's row in `causes.jsonl`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cause {
    pub subject_id: String,
    /// Leading underlying cause of death recode (1-10); `None` for a survivor
    /// or a death without cause data.
    pub ucod: Option<u16>,
    pub diabetes_mcod: Option<bool>,
    pub hypertension_mcod: Option<bool>,
    /// Both multiple-cause flags are present (true only for decedents).
    pub mcod_available: bool,
}

/// The sidecar row for a subject, refusing a record that cannot be right: a
/// survivor with a cause or a flag means the columns were misread.
pub fn cause(subject_id: &str, m: &Mortality) -> Result<Cause> {
    if !m.died && (m.cause.is_some() || m.diabetes_mcod.is_some() || m.hypertension_mcod.is_some())
    {
        bail!("{subject_id}: a survivor carries a cause of death or a multiple-cause flag");
    }
    Ok(Cause {
        subject_id: subject_id.to_string(),
        ucod: m.cause,
        diabetes_mcod: m.diabetes_mcod,
        hypertension_mcod: m.hypertension_mcod,
        mcod_available: m.diabetes_mcod.is_some() && m.hypertension_mcod.is_some(),
    })
}

/// Counts of flagged deaths for the build report.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct FlagCounts {
    pub diabetes: usize,
    pub hypertension: usize,
    /// Deaths with no multiple-cause data.
    pub unavailable: usize,
}

impl FlagCounts {
    pub fn add(&mut self, c: &Cause, died: bool) {
        if !died {
            return;
        }
        self.diabetes += usize::from(c.diabetes_mcod == Some(true));
        self.hypertension += usize::from(c.hypertension_mcod == Some(true));
        self.unavailable += usize::from(!c.mcod_available);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mortality(died: bool, d: Option<bool>, h: Option<bool>) -> Mortality {
        Mortality {
            eligible: true,
            died,
            cause: died.then_some(7),
            months_from_exam: Some(40.0),
            diabetes_mcod: d,
            hypertension_mcod: h,
        }
    }

    #[test]
    fn a_death_keeps_its_flags_and_a_missing_flag_is_unavailable_not_false() {
        let flagged = cause("a", &mortality(true, Some(true), Some(false))).unwrap();
        assert_eq!(
            (flagged.ucod, flagged.diabetes_mcod, flagged.mcod_available),
            (Some(7), Some(true), true)
        );
        let unknown = cause("b", &mortality(true, None, None)).unwrap();
        assert!(!unknown.mcod_available && unknown.diabetes_mcod.is_none());
    }

    #[test]
    fn a_survivor_with_a_flag_or_a_cause_is_refused() {
        let mut flagged = mortality(false, Some(true), Some(false));
        assert!(cause("c", &flagged).is_err());
        flagged.diabetes_mcod = None;
        flagged.hypertension_mcod = None;
        assert!(cause("c", &flagged).is_ok());
        flagged.cause = Some(1);
        assert!(cause("c", &flagged).is_err());
    }

    #[test]
    fn counts_cover_decedents_only_and_separate_the_unavailable() {
        let mut n = FlagCounts::default();
        for (died, d, h) in [
            (true, Some(true), Some(true)),
            (true, Some(false), Some(true)),
            (true, None, None),
            (false, None, None),
        ] {
            n.add(&cause("x", &mortality(died, d, h)).unwrap(), died);
        }
        assert_eq!(
            n,
            FlagCounts {
                diabetes: 1,
                hypertension: 2,
                unavailable: 1
            }
        );
    }
}
