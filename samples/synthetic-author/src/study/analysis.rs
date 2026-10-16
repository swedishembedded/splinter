// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in sizing and
// analysing a paired evaluation with honest statistics, you can procure our
// services by sending an email to info@swedishembedded.com.

//! What the replication study concludes from the outcomes of its arms, as a
//! pure function of them: per-seed paired tests, the gain of each comparison
//! per author with an item-clustered bootstrap interval, the spread over
//! training seeds, the sign test across authors with Holm's correction over
//! the protocol's family, survivorship and the measures of the reasons.
//!
//! Nothing here reads a file. What was not measured is absent from every table:
//! an arm without outcomes for all of the examination's items has no row, no
//! test uses it, and the report lists what is missing and why.

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;
use splinter_sdk::model::stats::{bootstrap_interval, sign_test, Interval};

use super::arm_key::ArmKey;
use super::layout::CorpusRecord;
use super::protocol::{Comparison, Multiplicity, Protocol};
use super::stats::{holm, sign_test_p, spread, Spread};
use crate::arms::Outcome;
use crate::exam::Item;
use crate::motives;
use crate::persona::Persona;
use crate::report::{accuracy, baselines};
use crate::sampled::SampledOutcome;
use crate::survivors::Survivors;

/// How many resamples of the items each bootstrap interval is made from.
pub const RESAMPLES: usize = 2000;
/// The share of resampled means an interval holds.
pub const LEVEL: f64 = 0.95;

/// The outcomes of one arm, in one slot: the arm named `arm`, and for an
/// adapter the training seed that made it. Every item of the examination has
/// an outcome: an arm answered in part is not measured.
#[derive(Clone, Debug)]
pub struct Measured {
    arm: String,
    training_seed: Option<u64>,
    outcomes: Vec<Outcome>,
}

impl Measured {
    /// The outcomes of `arm`, if there is one for every one of `items` (and
    /// none for any other).
    #[must_use]
    pub fn covering(
        arm: &str,
        training_seed: Option<u64>,
        outcomes: Vec<Outcome>,
        items: &[Item],
    ) -> Option<Self> {
        let ids: std::collections::BTreeSet<&str> = items.iter().map(|i| i.id.as_str()).collect();
        let answered: std::collections::BTreeSet<&str> =
            outcomes.iter().map(|o| o.id.as_str()).collect();
        (ids == answered && outcomes.len() == items.len()).then(|| Self {
            arm: arm.to_string(),
            training_seed,
            outcomes,
        })
    }

    /// The arm's name.
    #[must_use]
    pub fn arm(&self) -> &str {
        &self.arm
    }

    /// The training seed of an adapter; `None` for any other arm.
    #[must_use]
    pub fn training_seed(&self) -> Option<u64> {
        self.training_seed
    }
}

/// The sampled outcomes of one arm in one slot, for every item.
#[derive(Clone, Debug)]
pub struct MeasuredSampled {
    arm: String,
    training_seed: Option<u64>,
    outcomes: Vec<SampledOutcome>,
}

impl MeasuredSampled {
    /// The sampled outcomes of `arm`, if there is one for every one of
    /// `items`.
    #[must_use]
    pub fn covering(
        arm: &str,
        training_seed: Option<u64>,
        outcomes: Vec<SampledOutcome>,
        items: &[Item],
    ) -> Option<Self> {
        let ids: std::collections::BTreeSet<&str> = items.iter().map(|i| i.id.as_str()).collect();
        let answered: std::collections::BTreeSet<&str> =
            outcomes.iter().map(|o| o.id.as_str()).collect();
        (ids == answered && outcomes.len() == items.len()).then(|| Self {
            arm: arm.to_string(),
            training_seed,
            outcomes,
        })
    }
}

/// Something the study asked for that has no outcomes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Unmeasured {
    /// The author, if the gap is one author's.
    pub author: Option<u64>,
    /// What is missing: an arm and its slot, or the author's corpus.
    pub what: String,
    /// Why it is missing.
    pub why: String,
}

/// Everything the analysis of one author reads.
pub struct AuthorInput {
    /// The author's seed.
    pub seed: u64,
    /// The author's name.
    pub name: String,
    /// What the corpus came to.
    pub corpus: CorpusRecord,
    /// What the gates took out of it.
    pub survivors: Survivors,
    /// The frozen examination.
    pub items: Vec<Item>,
    /// The author, who says what the reasons should name.
    pub persona: Persona,
    /// The arms with outcomes for every item.
    pub measured: Vec<Measured>,
    /// The arms with sampled outcomes for every item.
    pub sampled: Vec<MeasuredSampled>,
}

/// An arm's accuracy in one slot.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AccuracyRow {
    /// The arm.
    pub arm: String,
    /// The training seed of an adapter.
    pub training_seed: Option<u64>,
    /// Items right.
    pub right: usize,
    /// Items asked.
    pub n: usize,
    /// The share right.
    pub rate: f64,
    /// The 95% Wilson interval's lower end.
    pub low: f64,
    /// The interval's upper end.
    pub high: f64,
}

/// An arm's accuracy on one kind of item.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CellRow {
    /// The arm.
    pub arm: String,
    /// The training seed of an adapter.
    pub training_seed: Option<u64>,
    /// The kind of item: `topic_seen`, `topic_new`, `combo_seen`, `combo_new`,
    /// `tool_seen` or `tool_new`.
    pub cell: String,
    /// Items right.
    pub right: usize,
    /// Items of the kind; a kind without items has no row.
    pub n: usize,
}

/// The paired sign test of one comparison at one training seed.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PairedRow {
    /// The comparison.
    pub candidate: String,
    /// What it is tested against.
    pub reference: String,
    /// The candidate's training seed, if it is an adapter.
    pub training_seed: Option<u64>,
    /// Items both were asked.
    pub n: usize,
    /// Items only the candidate got right.
    pub only_candidate: usize,
    /// Items only the reference got right.
    pub only_reference: usize,
    /// The one-sided exact p-value that the candidate is better.
    pub p: f64,
}

/// The gain of one comparison for one author, over the training seeds.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GainRow {
    /// The candidate.
    pub candidate: String,
    /// The reference.
    pub reference: String,
    /// The training seeds the gain is the mean over; empty for a candidate
    /// without an adapter.
    pub training_seeds: Vec<u64>,
    /// Items.
    pub n: usize,
    /// The mean over items of the seed-mean difference in correctness: the
    /// candidate's accuracy minus the reference's, averaged over seeds.
    pub gain: f64,
    /// The item-clustered bootstrap interval of the gain.
    pub interval: Option<Interval>,
    /// Items on which the candidate, averaged over seeds, beats the reference.
    pub items_won: usize,
    /// Items on which it loses.
    pub items_lost: usize,
    /// The one-sided exact p-value of the sign test over those items.
    pub p_items: f64,
}

/// How an adapter's accuracy varies over its training seeds.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SpreadRow {
    /// The adapter arm.
    pub arm: String,
    /// Mean accuracy over the seeds.
    pub mean: f64,
    /// Min, max and standard deviation of the accuracy over the seeds.
    #[serde(flatten)]
    pub spread: Spread,
}

/// An arm's accuracy on the items whose right answer is one option, beside the
/// share of that option the corpus kept.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SurvivorshipRow {
    /// The arm.
    pub arm: String,
    /// The option the beliefs determine for these items.
    pub option: String,
    /// Items of that option in the examination.
    pub items: usize,
    /// The arm's accuracy on them, averaged over its training seeds.
    pub accuracy: f64,
    /// Decisions of that option the corpus planned.
    pub planned: usize,
    /// Of those, the ones an article kept tells.
    pub kept: usize,
    /// The share kept; absent when none was planned.
    pub kept_share: Option<f64>,
}

/// What the reasons an arm gave come to.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReasonRow {
    /// The arm.
    pub arm: String,
    /// The training seed of an adapter.
    pub training_seed: Option<u64>,
    /// Answers.
    pub answers: usize,
    /// Answers that gave a reason.
    pub reasons: usize,
    /// Of the reasons, the share that name the fact the deciding belief turns
    /// on; absent when no reason was given.
    pub naming_deciding_fact: Option<f64>,
    /// Of the reasons, the share that repeat a phrase of the brief.
    pub copying_brief: Option<f64>,
    /// How many different reasons there are.
    pub distinct: usize,
}

/// An arm's results under the sampled decoding.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SampledRow {
    /// The arm.
    pub arm: String,
    /// The training seed of an adapter.
    pub training_seed: Option<u64>,
    /// Items asked.
    pub items: usize,
    /// Draws per item.
    pub draws: usize,
    /// The share of an item's draws that were right, averaged over items.
    pub share_right: f64,
}

/// The analysis of one author.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AuthorReport {
    /// The author's seed.
    pub seed: u64,
    /// The author's name.
    pub name: String,
    /// What the corpus came to.
    pub corpus: CorpusRecord,
    /// Articles planned and not kept, by why.
    pub given_up: BTreeMap<String, usize>,
    /// Accuracy of every arm, adapters once per training seed, and of the two
    /// constant policies.
    pub accuracy: Vec<AccuracyRow>,
    /// Accuracy by kind of item.
    pub cells: Vec<CellRow>,
    /// The paired sign tests per comparison and training seed.
    pub paired: Vec<PairedRow>,
    /// The gain per comparison over the training seeds.
    pub gains: Vec<GainRow>,
    /// The spread of each adapter's accuracy over its training seeds.
    pub spread: Vec<SpreadRow>,
    /// Accuracy by option beside what the corpus kept of the option.
    pub survivorship: Vec<SurvivorshipRow>,
    /// The reasons each arm gave.
    pub reasons: Vec<ReasonRow>,
    /// Results under the sampled decoding.
    pub sampled: Vec<SampledRow>,
}

/// One comparison across authors.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AcrossAuthors {
    /// The comparison.
    pub comparison: Comparison,
    /// Whether it is the protocol's primary comparison.
    pub primary: bool,
    /// Whether it is in the confirmatory family: an exploratory comparison has
    /// a raw p-value and no adjusted one.
    pub confirmatory: bool,
    /// Authors the comparison was measured for.
    pub authors_measured: usize,
    /// Authors in the protocol.
    pub authors_total: usize,
    /// Authors whose mean gain is positive.
    pub authors_improved: usize,
    /// Authors whose mean gain is negative.
    pub authors_worse: usize,
    /// Authors whose mean gain is exactly zero; they carry no evidence.
    pub authors_tied: usize,
    /// The one-sided exact sign test over the authors that improved or got
    /// worse; absent when no author was measured.
    pub p: Option<f64>,
    /// `p` adjusted by Holm's step-down over the protocol's whole family.
    pub p_holm: Option<f64>,
    /// Whether the adjusted p-value is at or below the family-wise level.
    pub significant: Option<bool>,
    /// The mean over authors of the per-author gain.
    pub mean_gain: Option<f64>,
    /// The smallest per-author gain.
    pub min_gain: Option<f64>,
    /// The largest per-author gain.
    pub max_gain: Option<f64>,
}

/// The whole report.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Report {
    /// The protocol the study was run under.
    pub protocol_digest: String,
    /// The seed of the bootstrap intervals.
    pub bootstrap_seed: u64,
    /// Resamples per interval.
    pub resamples: usize,
    /// The level of the intervals.
    pub level: f64,
    /// The family-wise error rate of the Holm correction.
    pub alpha: f64,
    /// The authors that were measured.
    pub authors: Vec<AuthorReport>,
    /// The comparisons across authors, in the protocol's order.
    pub across: Vec<AcrossAuthors>,
    /// What was asked for and is missing.
    pub unmeasured: Vec<Unmeasured>,
}

fn is_adapter(arm: &str) -> bool {
    ArmKey::parse(arm).is_some_and(ArmKey::is_adapter)
}

/// Whether an arm was right, by item id.
type Answers<'a> = HashMap<&'a str, bool>;

fn correct_by_item(m: &Measured) -> Answers<'_> {
    m.outcomes
        .iter()
        .map(|o| (o.id.as_str(), o.correct))
        .collect()
}

/// The measured arm of `comparison`'s reference that goes with the candidate
/// measured at `seed`: the one trained at the same seed when the reference is
/// an adapter, else the only one.
fn reference_for<'a>(
    measured: &'a [Measured],
    reference: &str,
    seed: Option<u64>,
) -> Option<&'a Measured> {
    measured
        .iter()
        .find(|m| m.arm == reference && (!is_adapter(reference) || m.training_seed == seed))
}

fn paired_row(candidate: &Measured, reference: &Measured, names: &Comparison) -> PairedRow {
    let theirs = correct_by_item(reference);
    let pairs: Vec<(bool, bool)> = candidate
        .outcomes
        .iter()
        .filter_map(|o| theirs.get(o.id.as_str()).map(|&r| (o.correct, r)))
        .collect();
    let test = sign_test(&pairs);
    PairedRow {
        candidate: names.candidate.clone(),
        reference: names.reference.clone(),
        training_seed: candidate.training_seed,
        n: pairs.len(),
        only_candidate: pairs.iter().filter(|(c, r)| *c && !*r).count(),
        only_reference: pairs.iter().filter(|(c, r)| !*c && *r).count(),
        p: test.p_value,
    }
}

/// The gain of `names` for one author over the training seeds that both
/// sides have, or `None` when there is no pair of measured arms.
fn gain_row(
    measured: &[Measured],
    names: &Comparison,
    items: &[Item],
    bootstrap_seed: u64,
) -> Option<GainRow> {
    let pairs: Vec<(&Measured, &Measured)> = measured
        .iter()
        .filter(|m| m.arm == names.candidate)
        .filter_map(|m| reference_for(measured, &names.reference, m.training_seed).map(|r| (m, r)))
        .collect();
    if pairs.is_empty() {
        return None;
    }
    let maps: Vec<(Answers<'_>, Answers<'_>)> = pairs
        .iter()
        .map(|(c, r)| (correct_by_item(c), correct_by_item(r)))
        .collect();
    let score = |right: bool| f64::from(u8::from(right));
    let diffs: Vec<f64> = items
        .iter()
        .map(|item| {
            let total: f64 = maps
                .iter()
                .map(|(c, r)| score(c[item.id.as_str()]) - score(r[item.id.as_str()]))
                .sum();
            total / maps.len() as f64
        })
        .collect();
    let won = diffs.iter().filter(|&&d| d > 1e-12).count();
    let lost = diffs.iter().filter(|&&d| d < -1e-12).count();
    Some(GainRow {
        candidate: names.candidate.clone(),
        reference: names.reference.clone(),
        training_seeds: pairs.iter().filter_map(|(c, _)| c.training_seed).collect(),
        n: diffs.len(),
        gain: diffs.iter().sum::<f64>() / diffs.len() as f64,
        interval: bootstrap_interval(&diffs, RESAMPLES, LEVEL, bootstrap_seed),
        items_won: won,
        items_lost: lost,
        p_items: sign_test_p(won, won + lost),
    })
}

/// A kind of item the examination balances, by the flag each outcome keeps.
struct Cell {
    name: &'static str,
    holds: fn(&Outcome) -> bool,
}

const CELLS: [Cell; 6] = [
    Cell {
        name: "topic_seen",
        holds: |o| o.topic_seen,
    },
    Cell {
        name: "topic_new",
        holds: |o| !o.topic_seen,
    },
    Cell {
        name: "combo_seen",
        holds: |o| o.combo_seen,
    },
    Cell {
        name: "combo_new",
        holds: |o| !o.combo_seen,
    },
    Cell {
        name: "tool_seen",
        holds: |o| o.tool_seen,
    },
    Cell {
        name: "tool_new",
        holds: |o| !o.tool_seen,
    },
];

fn cell_rows(measured: &[Measured]) -> Vec<CellRow> {
    measured
        .iter()
        .flat_map(|m| {
            CELLS.iter().filter_map(move |cell| {
                let of_kind: Vec<&Outcome> =
                    m.outcomes.iter().filter(|o| (cell.holds)(o)).collect();
                (!of_kind.is_empty()).then(|| CellRow {
                    arm: m.arm.clone(),
                    training_seed: m.training_seed,
                    cell: cell.name.to_string(),
                    right: of_kind.iter().filter(|o| o.correct).count(),
                    n: of_kind.len(),
                })
            })
        })
        .collect()
}

fn spread_rows(accuracy: &[AccuracyRow]) -> Vec<SpreadRow> {
    let mut by_arm: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for row in accuracy.iter().filter(|r| r.training_seed.is_some()) {
        by_arm.entry(&row.arm).or_default().push(row.rate);
    }
    by_arm
        .into_iter()
        .filter_map(|(arm, rates)| {
            Some(SpreadRow {
                arm: arm.to_string(),
                mean: rates.iter().sum::<f64>() / rates.len() as f64,
                spread: spread(&rates)?,
            })
        })
        .collect()
}

fn option_name(item: &Item) -> String {
    serde_json::to_value(item.expected)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

fn survivorship_rows(
    measured: &[Measured],
    items: &[Item],
    survivors: &Survivors,
) -> Vec<SurvivorshipRow> {
    let arms: std::collections::BTreeSet<&str> = measured.iter().map(|m| m.arm.as_str()).collect();
    let mut options: Vec<String> = items.iter().map(option_name).collect();
    options.sort();
    options.dedup();
    let mut rows = Vec::new();
    for arm in arms {
        for option in &options {
            let of_option: Vec<&Item> =
                items.iter().filter(|i| &option_name(i) == option).collect();
            let slots: Vec<&Measured> = measured.iter().filter(|m| m.arm == arm).collect();
            let accuracy = slots
                .iter()
                .map(|m| {
                    let ours = correct_by_item(m);
                    of_option.iter().filter(|i| ours[i.id.as_str()]).count() as f64
                        / of_option.len() as f64
                })
                .sum::<f64>()
                / slots.len() as f64;
            let tally = survivors.by_option.get(option).copied().unwrap_or_default();
            rows.push(SurvivorshipRow {
                arm: arm.to_string(),
                option: option.clone(),
                items: of_option.len(),
                accuracy,
                planned: tally.planned,
                kept: tally.kept,
                kept_share: (tally.planned > 0).then(|| tally.kept as f64 / tally.planned as f64),
            });
        }
    }
    rows
}

fn reason_rows(measured: &[Measured], input: &AuthorInput) -> Vec<ReasonRow> {
    measured
        .iter()
        .filter(|m| m.outcomes.iter().any(|o| o.call.is_some()))
        .map(|m| {
            let graded = motives::grade(&input.items, &m.outcomes, &input.persona);
            let share =
                |part: usize| (graded.reasons > 0).then(|| part as f64 / graded.reasons as f64);
            ReasonRow {
                arm: m.arm.clone(),
                training_seed: m.training_seed,
                answers: m.outcomes.len(),
                reasons: graded.reasons,
                naming_deciding_fact: share(graded.naming_deciding_fact),
                copying_brief: share(graded.copying_brief),
                distinct: graded.distinct,
            }
        })
        .collect()
}

fn sampled_rows(sampled: &[MeasuredSampled]) -> Vec<SampledRow> {
    sampled
        .iter()
        .filter_map(|m| {
            let shares: Vec<f64> = m
                .outcomes
                .iter()
                .filter_map(SampledOutcome::share_right)
                .collect();
            (!shares.is_empty()).then(|| SampledRow {
                arm: m.arm.clone(),
                training_seed: m.training_seed,
                items: shares.len(),
                draws: m.outcomes.first().map_or(0, |o| o.draws.len()),
                share_right: shares.iter().sum::<f64>() / shares.len() as f64,
            })
        })
        .collect()
}

/// The analysis of one author: every table, from the outcomes alone.
/// `bootstrap_seed` seeds each interval.
#[must_use]
pub fn analyse_author(
    protocol: &Protocol,
    input: &AuthorInput,
    bootstrap_seed: u64,
) -> AuthorReport {
    let mut measured = input.measured.clone();
    measured.extend(
        baselines(&input.items)
            .into_iter()
            .filter_map(|(name, outcomes)| Measured::covering(name, None, outcomes, &input.items)),
    );
    let accuracy_rows: Vec<AccuracyRow> = measured
        .iter()
        .map(|m| {
            let r = accuracy(&m.outcomes);
            AccuracyRow {
                arm: m.arm.clone(),
                training_seed: m.training_seed,
                right: r.right,
                n: r.n,
                rate: r.rate,
                low: r.low,
                high: r.high,
            }
        })
        .collect();
    let mut paired = Vec::new();
    let mut gains = Vec::new();
    for names in protocol.comparisons.iter().chain(&protocol.exploratory) {
        for candidate in measured.iter().filter(|m| m.arm == names.candidate) {
            if let Some(reference) =
                reference_for(&measured, &names.reference, candidate.training_seed)
            {
                paired.push(paired_row(candidate, reference, names));
            }
        }
        gains.extend(gain_row(&measured, names, &input.items, bootstrap_seed));
    }
    AuthorReport {
        seed: input.seed,
        name: input.name.clone(),
        corpus: input.corpus.clone(),
        given_up: input
            .survivors
            .given_up
            .iter()
            .map(|(why, articles)| (why.clone(), articles.len()))
            .collect(),
        spread: spread_rows(&accuracy_rows),
        cells: cell_rows(&measured),
        accuracy: accuracy_rows,
        paired,
        gains,
        survivorship: survivorship_rows(&measured, &input.items, &input.survivors),
        reasons: reason_rows(&measured, input),
        sampled: sampled_rows(&input.sampled),
    }
}

/// Each comparison of the protocol across `authors`: the sign test of the
/// authors' mean gains, and Holm's correction over the whole family.
#[must_use]
pub fn across_authors(protocol: &Protocol, authors: &[AuthorReport]) -> Vec<AcrossAuthors> {
    let Multiplicity::HolmOverComparisons { alpha } = protocol.multiplicity;
    let family = protocol.comparisons.len();
    let mut rows: Vec<AcrossAuthors> = protocol
        .comparisons
        .iter()
        .chain(&protocol.exploratory)
        .enumerate()
        .map(|(place, names)| {
            let gains: Vec<f64> = authors
                .iter()
                .filter_map(|a| {
                    a.gains
                        .iter()
                        .find(|g| g.candidate == names.candidate && g.reference == names.reference)
                })
                .map(|g| g.gain)
                .collect();
            let improved = gains.iter().filter(|&&g| g > 1e-12).count();
            let worse = gains.iter().filter(|&&g| g < -1e-12).count();
            AcrossAuthors {
                comparison: names.clone(),
                primary: *names == protocol.primary,
                confirmatory: place < family,
                authors_measured: gains.len(),
                authors_total: protocol.authors.len(),
                authors_improved: improved,
                authors_worse: worse,
                authors_tied: gains.len() - improved - worse,
                p: (!gains.is_empty()).then(|| sign_test_p(improved, improved + worse)),
                p_holm: None,
                significant: None,
                mean_gain: (!gains.is_empty())
                    .then(|| gains.iter().sum::<f64>() / gains.len() as f64),
                min_gain: gains.iter().copied().reduce(f64::min),
                max_gain: gains.iter().copied().reduce(f64::max),
            }
        })
        .collect();
    let raw: Vec<Option<f64>> = rows.iter().take(family).map(|r| r.p).collect();
    for (row, adjusted) in rows.iter_mut().zip(holm(&raw, family)) {
        row.p_holm = adjusted;
        row.significant = adjusted.map(|p| p <= alpha);
    }
    rows
}
