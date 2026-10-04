// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source-grounded persona and expertise
// transfer for language models for its clients. If your team needs expertise
// in keeping a historical record's authorship honest before a model learns
// from it, you can procure our services by sending an email to
// info@swedishembedded.com.

//! One document of the corpus, and the questions asked of it before any model
//! sees it: who wrote it, when, and is it the right Samuel Adams.

/// How sure the record is that Adams wrote the words, strongest first.
///
/// A committee text he sat on is not his private letter, and a newspaper
/// piece under a pseudonym is not his signature; a model that learns them as
/// one voice learns a composite.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Authorship {
    DirectAutograph,
    SignedScribal,
    DraftInHand,
    CommitteeCoauthored,
    PseudonymousAttributed,
    EditorAttributed,
    SecondaryQuoted,
    ContextOnly,
}

impl Authorship {
    /// May the text be taught as what Adams said, and be quoted as his.
    pub fn is_voice(self) -> bool {
        self <= Authorship::CommitteeCoauthored
    }

    /// Is there a settled answer to "what kind of document is this", so that
    /// a question about its kind can be graded: his own text or a newspaper
    /// piece under a pseudonym, but not a text only an editor ascribes to him.
    pub fn has_a_settled_kind(self) -> bool {
        self <= Authorship::PseudonymousAttributed
    }

    /// May the text support a hypothesis about how he reasoned.
    pub fn supports_principles(self) -> bool {
        self <= Authorship::EditorAttributed
    }
}

/// The span of a document's date; month and day are often unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Date {
    pub year: u16,
    pub month: Option<u8>,
    pub day: Option<u8>,
}

/// Where in his life a document falls.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Period {
    EarlyCareer,
    PreRevolution,
    Revolutionary,
    Confederation,
    MassachusettsExecutive,
    LateLife,
}

impl Period {
    pub fn of(year: u16) -> Period {
        match year {
            ..=1764 => Period::EarlyCareer,
            1765..=1773 => Period::PreRevolution,
            1774..=1783 => Period::Revolutionary,
            1784..=1788 => Period::Confederation,
            1789..=1796 => Period::MassachusettsExecutive,
            1797.. => Period::LateLife,
        }
    }
}

/// The first year withheld from training, so that a model tested on his last
/// years has never seen them.
pub const TEMPORAL_HOLDOUT_FROM: u16 = 1790;

pub fn in_temporal_holdout(year: u16) -> bool {
    year >= TEMPORAL_HOLDOUT_FROM
}

/// The years Adams wrote publicly or in office: before 1740 no surviving text
/// is his, and he died in 1803.
const WORKING_YEARS: std::ops::RangeInclusive<u16> = 1740..=1803;

/// His father, Deacon Samuel Adams, died in 1748; until then a "junior" on a
/// Boston document may be either of them.
const FATHERS_DEATH: u16 = 1748;

/// Author lines that name someone else called Samuel Adams, with who they are.
const NAMESAKES: [(&str, &str); 6] = [
    ("physician", "a physician of the same name"),
    ("dr. samuel adams", "a physician of the same name"),
    ("samuel adams drake", "Samuel Adams Drake, a later writer"),
    ("hopkins adams", "Samuel Hopkins Adams, a later writer"),
    ("surgeon", "a surgeon of the same name"),
    ("deacon", "his father, Deacon Samuel Adams"),
];

/// Whether a document is by the Samuel Adams being modelled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Identity {
    Target,
    Excluded(&'static str),
    NeedsReview(&'static str),
}

/// Decide from the author line and the date alone; a document the rules cannot
/// settle is sent for review rather than guessed.
pub fn identity(author: &str, date: Date) -> Identity {
    let author = author.to_lowercase();
    if let Some((_, who)) = NAMESAKES.iter().find(|(marker, _)| author.contains(marker)) {
        return Identity::Excluded(who);
    }
    if !WORKING_YEARS.contains(&date.year) {
        return Identity::Excluded("dated outside the years Samuel Adams wrote");
    }
    let junior = author.contains("jr") || author.contains("junr") || author.contains("jun.");
    if junior && date.year < FATHERS_DEATH {
        return Identity::NeedsReview(
            "a junior before his father's death may be either Samuel Adams",
        );
    }
    Identity::Target
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dated(year: u16) -> Date {
        Date {
            year,
            month: None,
            day: None,
        }
    }

    #[test]
    fn only_the_first_four_are_taught_as_his_voice() {
        use Authorship::*;
        let voice = [
            DirectAutograph,
            SignedScribal,
            DraftInHand,
            CommitteeCoauthored,
        ];
        let not_voice = [
            PseudonymousAttributed,
            EditorAttributed,
            SecondaryQuoted,
            ContextOnly,
        ];
        assert!(voice.iter().all(|a| a.is_voice()));
        assert!(not_voice.iter().all(|a| !a.is_voice()));
    }

    #[test]
    fn attributed_writing_may_support_a_principle_but_never_be_quoted_as_his() {
        use Authorship::*;
        for a in [PseudonymousAttributed, EditorAttributed] {
            assert!(a.supports_principles() && !a.is_voice(), "{a:?}");
        }
        for a in [SecondaryQuoted, ContextOnly] {
            assert!(!a.supports_principles(), "{a:?}");
        }
    }

    #[test]
    fn a_kind_is_settled_for_his_own_text_and_for_newspaper_pieces_but_not_for_ascriptions() {
        use Authorship::*;
        for a in [
            DirectAutograph,
            SignedScribal,
            DraftInHand,
            CommitteeCoauthored,
            PseudonymousAttributed,
        ] {
            assert!(a.has_a_settled_kind(), "{a:?}");
        }
        for a in [EditorAttributed, SecondaryQuoted, ContextOnly] {
            assert!(!a.has_a_settled_kind(), "{a:?}");
        }
    }

    #[test]
    fn authorship_orders_strongest_first() {
        assert!(Authorship::DirectAutograph < Authorship::CommitteeCoauthored);
        assert!(Authorship::CommitteeCoauthored < Authorship::ContextOnly);
    }

    #[test]
    fn period_follows_the_phases_of_his_career() {
        assert_eq!(Period::of(1760), Period::EarlyCareer);
        assert_eq!(Period::of(1770), Period::PreRevolution);
        assert_eq!(Period::of(1776), Period::Revolutionary);
        assert_eq!(Period::of(1786), Period::Confederation);
        assert_eq!(Period::of(1794), Period::MassachusettsExecutive);
        assert_eq!(Period::of(1800), Period::LateLife);
    }

    #[test]
    fn the_last_years_are_withheld_from_training() {
        assert!(!in_temporal_holdout(1789));
        assert!(in_temporal_holdout(TEMPORAL_HOLDOUT_FROM));
        assert!(in_temporal_holdout(1803));
    }

    #[test]
    fn a_namesake_is_excluded_by_the_author_line() {
        for author in [
            "Samuel Adams, physician",
            "Dr. Samuel Adams",
            "Samuel Adams Drake",
            "Samuel Hopkins Adams",
        ] {
            assert!(
                matches!(identity(author, dated(1775)), Identity::Excluded(_)),
                "{author}"
            );
        }
    }

    #[test]
    fn a_date_outside_his_working_life_is_excluded() {
        assert!(matches!(
            identity("Samuel Adams", dated(1730)),
            Identity::Excluded(_)
        ));
        assert!(matches!(
            identity("Samuel Adams", dated(1861)),
            Identity::Excluded(_)
        ));
        assert_eq!(identity("Samuel Adams", dated(1774)), Identity::Target);
        assert_eq!(identity("Samuel Adams", dated(1803)), Identity::Target);
    }

    #[test]
    fn junior_before_the_fathers_death_is_a_question_not_a_guess() {
        assert!(matches!(
            identity("Samuel Adams, Jr.", dated(1745)),
            Identity::NeedsReview(_)
        ));
        assert_eq!(identity("Samuel Adams, Jr.", dated(1770)), Identity::Target);
    }
}
