// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in sizing and
// analysing a paired evaluation with honest statistics, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the study's report is made of its manifests and outcomes alone. It
//! tests tuned against every other arm per author and training seed, gives the
//! spread over seeds, tests the per-author gain across authors by an exact
//! sign test and corrects over the protocol's family, and leaves out what was
//! not measured instead of writing zero.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod study_support;

use std::sync::Arc;
use std::time::Duration;

use splinter_sdk::agent::sven::model::ModelProvider;
use study_support::{frozen, responding, small_protocol, FirstOption, Oracle, Scripted};
use synthetic_author::arms::Outcome;
use synthetic_author::cache::Usage;
use synthetic_author::exam::Item;
use synthetic_author::lab::exam_items;
use synthetic_author::persona::Persona;
use synthetic_author::study::analysis::{
    across_authors, analyse_author, AuthorInput, AuthorReport, Measured,
};
use synthetic_author::study::corpus::{self, Staff};
use synthetic_author::study::examine::{self, Examinee, Loader};
use synthetic_author::study::layout::CorpusRecord;
use synthetic_author::study::protocol::{AuthorPlan, Protocol};
use synthetic_author::study::{render, report, train};
use synthetic_author::survivors::{Survivors, Tally};
use synthetic_author::transit::Transit;

/// A fixed number of items per kind keeps the in-memory specs small.
const PER_CELL: usize = 10;

fn items_of(persona: &Persona, seed: u64) -> Vec<Item> {
    exam_items(persona, seed, 6, PER_CELL)
}

/// The outcomes of a model that is right on the items `right` names.
fn outcomes(items: &[Item], right: impl Fn(usize) -> bool) -> Vec<Outcome> {
    items
        .iter()
        .enumerate()
        .map(|(n, item)| Outcome {
            id: item.id.clone(),
            correct: right(n),
            topic_seen: item.topic_seen,
            combo_seen: item.combo_seen,
            combo_trained: item.combo_trained,
            tool_seen: item.tool_seen,
            call: None,
            item: String::new(),
            model: String::new(),
        })
        .collect()
}

fn measured(
    arm: &str,
    seed: Option<u64>,
    items: &[Item],
    right: impl Fn(usize) -> bool,
) -> Measured {
    Measured::covering(arm, seed, outcomes(items, right), items).unwrap()
}

struct Fixture {
    protocol: Protocol,
    persona: Persona,
    items: Vec<Item>,
    seed: u64,
}

fn fixture(authors: usize) -> Fixture {
    let mut protocol = small_protocol(|_| {});
    protocol.authors = (0..authors)
        .map(|n| AuthorPlan {
            seed: n as u64 + 1,
            training_seeds: vec![101, 102],
        })
        .collect();
    let seed = splinter_sdk::model::seed::draw().value();
    let persona = Persona::generate(seed);
    let items = items_of(&persona, seed);
    Fixture {
        protocol,
        persona,
        items,
        seed,
    }
}

impl Fixture {
    fn input(&self, author: u64, measured: Vec<Measured>, survivors: Survivors) -> AuthorInput {
        AuthorInput {
            seed: author,
            name: format!("Author {author}"),
            corpus: CorpusRecord {
                articles_planned: 6,
                articles_kept: 6,
                cases_read: 12,
                decisions_kept: 12,
                usage: Usage::default(),
            },
            survivors,
            items: self.items.clone(),
            persona: self.persona.clone(),
            measured,
            sampled: Vec::new(),
        }
    }

    /// An author whose tuned adapter beats retrieval (when `improves`) and
    /// whose retrieval arm beats the base.
    fn author(&self, author: u64, improves: bool) -> AuthorReport {
        let n = self.items.len();
        let arms = vec![
            measured("base", None, &self.items, |i| i % 5 == 0),
            measured("retrieval", None, &self.items, |i| i % 2 == 0),
            // Right on everything (but the last two items at the second seed)
            // when the author improves; on a quarter of the items otherwise.
            measured("tuned", Some(101), &self.items, move |i| {
                if improves {
                    i < n
                } else {
                    i % 4 == 1
                }
            }),
            measured("tuned", Some(102), &self.items, move |i| {
                if improves {
                    i < n - 2
                } else {
                    i % 4 == 1
                }
            }),
        ];
        analyse_author(
            &self.protocol,
            &self.input(author, arms, Survivors::default()),
            self.seed,
        )
    }
}

fn row<'a>(
    rows: &'a [synthetic_author::study::analysis::AcrossAuthors],
    candidate: &str,
    reference: &str,
) -> &'a synthetic_author::study::analysis::AcrossAuthors {
    rows.iter()
        .find(|r| r.comparison.candidate == candidate && r.comparison.reference == reference)
        .unwrap()
}

#[test]
fn eight_authors_out_of_eight_give_the_exact_sign_test_p_and_seven_give_the_next() {
    let f = fixture(8);
    let eight: Vec<AuthorReport> = (1..=8).map(|a| f.author(a, true)).collect();
    let rows = across_authors(&f.protocol, &eight);
    let primary = row(&rows, "tuned", "retrieval");
    assert!(primary.primary && primary.confirmatory);
    assert_eq!((primary.authors_measured, primary.authors_improved), (8, 8));
    assert!((primary.p.unwrap() - 0.5_f64.powi(8)).abs() < 1e-12);
    assert!(
        (primary.p.unwrap() - 0.0039).abs() < 5e-5,
        "{:?}",
        primary.p
    );
    // Holm over the whole family of the protocol, not over what was measured.
    let family = f.protocol.comparisons.len() as f64;
    assert!((primary.p_holm.unwrap() - family * 0.5_f64.powi(8)).abs() < 1e-12);
    assert_eq!(primary.significant, Some(true));

    let mut seven = eight.clone();
    seven[7] = f.author(8, false);
    let rows = across_authors(&f.protocol, &seven);
    let primary = row(&rows, "tuned", "retrieval");
    assert_eq!((primary.authors_improved, primary.authors_worse), (7, 1));
    assert!((primary.p.unwrap() - 9.0 / 256.0).abs() < 1e-12, "0.035");
    assert_eq!(
        primary.significant,
        Some(false),
        "seven of eight survives no correction over ten"
    );
}

#[test]
fn a_comparison_with_an_arm_that_was_not_measured_has_no_p_and_no_row_anywhere() {
    let f = fixture(3);
    let reports: Vec<AuthorReport> = (1..=3).map(|a| f.author(a, true)).collect();
    let rows = across_authors(&f.protocol, &reports);
    let named = row(&rows, "tuned", "named");
    assert_eq!(named.authors_measured, 0);
    assert!(named.p.is_none() && named.p_holm.is_none() && named.significant.is_none());
    assert!(named.mean_gain.is_none());
    for report in &reports {
        assert!(report.accuracy.iter().all(|r| r.arm != "named"));
        assert!(report.paired.iter().all(|r| r.reference != "named"));
        assert!(report.gains.iter().all(|r| r.reference != "named"));
    }
    // The constant policies are measured from the items alone.
    assert!(reports[0].accuracy.iter().any(|r| r.arm == "first-listed"));
    assert!(reports[0]
        .accuracy
        .iter()
        .any(|r| r.arm == "majority-option"));
}

#[test]
fn exploratory_comparisons_are_reported_raw_and_never_adjusted() {
    let f = fixture(3);
    let reports: Vec<AuthorReport> = (1..=3).map(|a| f.author(a, true)).collect();
    let rows = across_authors(&f.protocol, &reports);
    let sweep = row(&rows, "retrieval", "base");
    assert!(!sweep.confirmatory);
    assert_eq!(sweep.authors_improved, 3);
    assert!(sweep.p.is_some() && sweep.p_holm.is_none() && sweep.significant.is_none());
}

#[test]
fn every_training_seed_is_tested_alone_and_the_gain_is_their_mean_with_an_item_clustered_interval()
{
    let f = fixture(1);
    let report = f.author(1, true);
    let n = f.items.len();
    let per_seed: Vec<_> = report
        .paired
        .iter()
        .filter(|p| p.candidate == "tuned" && p.reference == "retrieval")
        .collect();
    assert_eq!(per_seed.len(), 2, "one test per training seed");
    let first = per_seed
        .iter()
        .find(|p| p.training_seed == Some(101))
        .unwrap();
    // Seed 101 is right on everything; retrieval on every other item.
    assert_eq!(
        (first.n, first.only_candidate, first.only_reference),
        (n, n / 2, 0)
    );
    let reference = splinter_sdk::model::stats::sign_test(
        &(0..n).map(|i| (true, i % 2 == 0)).collect::<Vec<_>>(),
    );
    assert!((first.p - reference.p_value).abs() < 1e-12);

    let gain = report
        .gains
        .iter()
        .find(|g| g.candidate == "tuned" && g.reference == "retrieval")
        .unwrap();
    assert_eq!(gain.training_seeds, [101, 102]);
    let seed_gain = |right: usize| right as f64 / n as f64 - 0.5;
    let expected = (seed_gain(n) + seed_gain(n - 2)) / 2.0;
    assert!(
        (gain.gain - expected).abs() < 1e-9,
        "{} {expected}",
        gain.gain
    );
    let interval = gain.interval.unwrap();
    assert!(interval.low <= gain.gain && gain.gain <= interval.high);
    assert!(interval.above_zero());

    // The interval is a function of the bootstrap seed that is recorded.
    let again = f.author(1, true);
    assert_eq!(again.gains, report.gains);
}

#[test]
fn the_spread_over_seeds_has_min_max_and_a_standard_deviation_only_from_two_seeds() {
    let f = fixture(1);
    let n = f.items.len();
    let arms = vec![
        measured("tuned", Some(101), &f.items, |i| i < n / 2),
        measured("tuned", Some(102), &f.items, |i| i < n * 3 / 4),
        measured("tuned-reader-only", Some(101), &f.items, |i| i < n / 4),
    ];
    let report = analyse_author(&f.protocol, &f.input(1, arms, Survivors::default()), f.seed);
    let tuned = report.spread.iter().find(|s| s.arm == "tuned").unwrap();
    assert_eq!(tuned.spread.seeds, 2);
    assert!((tuned.spread.min - 0.5).abs() < 1e-9 && (tuned.spread.max - 0.75).abs() < 1e-9);
    assert!((tuned.spread.sd.unwrap() - 0.25 / 2.0_f64.sqrt()).abs() < 1e-9);
    let single = report
        .spread
        .iter()
        .find(|s| s.arm == "tuned-reader-only")
        .unwrap();
    assert_eq!(single.spread.sd, None, "one seed has no standard deviation");
    assert!(
        report.spread.iter().all(|s| s.arm != "retrieval"),
        "an arm without seeds has no spread"
    );
}

#[test]
fn accuracy_by_option_sits_beside_the_share_of_the_option_the_corpus_kept() {
    let f = fixture(1);
    let mut survivors = Survivors::default();
    for item in &f.items {
        let name = serde_json::to_value(item.expected)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        survivors.by_option.insert(
            name,
            Tally {
                planned: 10,
                kept: 4,
            },
        );
    }
    survivors.by_option.insert(
        "hold_and_wait".into(),
        Tally {
            planned: 0,
            kept: 0,
        },
    );
    let arms = vec![measured("tuned", Some(101), &f.items, |_| true)];
    let report = analyse_author(&f.protocol, &f.input(1, arms, survivors), f.seed);
    let rows: Vec<_> = report
        .survivorship
        .iter()
        .filter(|r| r.arm == "tuned")
        .collect();
    assert!(!rows.is_empty());
    for r in &rows {
        assert!((r.accuracy - 1.0).abs() < 1e-12);
        if r.option == "hold_and_wait" {
            assert_eq!(
                r.kept_share, None,
                "nothing was planned, so there is no share"
            );
        } else {
            assert_eq!((r.planned, r.kept), (10, 4));
            assert!((r.kept_share.unwrap() - 0.4).abs() < 1e-12);
        }
    }
    assert_eq!(rows.iter().map(|r| r.items).sum::<usize>(), f.items.len());
}

#[test]
fn an_arm_that_gave_no_reason_has_no_share_of_reasons() {
    let f = fixture(1);
    let mut silent = outcomes(&f.items, |_| true);
    for o in &mut silent {
        o.call = Some(serde_json::json!({"name": "t", "arguments": {}}));
    }
    let arms = vec![Measured::covering("tuned", Some(101), silent, &f.items).unwrap()];
    let report = analyse_author(&f.protocol, &f.input(1, arms, Survivors::default()), f.seed);
    let r = report.reasons.iter().find(|r| r.arm == "tuned").unwrap();
    assert_eq!((r.answers, r.reasons), (f.items.len(), 0));
    assert!(r.naming_deciding_fact.is_none() && r.copying_brief.is_none());
    assert!(
        report.reasons.iter().all(|r| r.arm != "first-listed"),
        "a constant policy gives no reasons to grade"
    );
}

#[test]
fn an_arm_answered_in_part_is_not_measured() {
    let f = fixture(1);
    let mut partial = outcomes(&f.items, |_| true);
    partial.pop();
    assert!(Measured::covering("tuned", Some(101), partial, &f.items).is_none());
}

// ---- the report of a study on disk ----

struct Plain(Arc<dyn ModelProvider>);
struct Tuned(Arc<dyn ModelProvider>);

impl Examinee for Plain {
    fn greedy(&self) -> Arc<dyn ModelProvider> {
        self.0.clone()
    }
    fn sampled(&self, _: f32, _: u64) -> Arc<dyn ModelProvider> {
        self.0.clone()
    }
}
impl Examinee for Tuned {
    fn greedy(&self) -> Arc<dyn ModelProvider> {
        self.0.clone()
    }
    fn sampled(&self, _: f32, _: u64) -> Arc<dyn ModelProvider> {
        self.0.clone()
    }
}

/// Plain models take the first option; an adapter takes the right one.
struct Loading {
    oracle: Vec<Item>,
}

impl Loader for Loading {
    fn load(&self, adapter: Option<&std::path::Path>) -> anyhow::Result<Box<dyn Examinee>> {
        Ok(match adapter {
            None => Box::new(Plain(Arc::new(FirstOption {
                requests: Arc::default(),
            }))),
            Some(_) => Box::new(Tuned(Arc::new(Oracle::of(&self.oracle)))),
        })
    }
}

fn pace() -> Transit {
    Transit::new(1, Duration::ZERO)
}

/// A study of `authors` authors run through every stage with scripted models.
fn run_study(tag: &str, authors: usize) -> synthetic_author::study::layout::Study {
    let study = frozen(
        tag,
        small_protocol(|p| p.decoding.secondary = None),
        authors,
    );
    let base = study.root().join("base.bin");
    std::fs::write(&base, "base weights").unwrap();
    for plan in &study.protocol().authors {
        let (model, _) = responding(plan.seed, study.protocol());
        corpus::run(
            &study,
            plan.seed,
            &Staff {
                writer: &model,
                reader: &model,
                miner: &model,
            },
            &pace(),
        )
        .unwrap();
        train::run(&study, plan.seed, &base, &Scripted::default()).unwrap();
        let cache = study.cache(plan.seed).unwrap();
        let author = synthetic_author::manifest::author(cache.dir(), Some(plan.seed)).unwrap();
        let items = synthetic_author::lab::exam_items_of(
            &cache,
            &author.persona,
            author.seed,
            study.protocol().corpus.articles,
            study.protocol().exam.per_cell,
            study.protocol().exam.version,
        )
        .unwrap();
        examine::run(&study, plan.seed, &base, &Loading { oracle: items }, false).unwrap();
    }
    study
}

fn csv(study: &synthetic_author::study::layout::Study, name: &str) -> Vec<Vec<String>> {
    std::fs::read_to_string(study.root().join("report").join(name))
        .unwrap()
        .lines()
        .map(|l| l.split(',').map(str::to_string).collect())
        .collect()
}

#[test]
fn the_report_of_a_study_on_disk_is_written_as_csv_json_and_a_table() {
    let study = run_study("report-disk", 3);
    let report = report::run(&study, Some(41)).unwrap();
    let dir = render::write(study.root(), &report).unwrap();
    assert_eq!(report.bootstrap_seed, 41);
    assert!(report.unmeasured.is_empty(), "{:?}", report.unmeasured);
    assert_eq!(report.authors.len(), 3);

    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(json["bootstrap_seed"], 41);
    assert_eq!(json["protocol_digest"], study.digest().as_str());

    // The adapters (right on every item) beat the base (first option) for every author.
    let across = csv(&study, "across_authors.csv");
    let header = &across[0];
    let at = |name: &str| header.iter().position(|h| h == name).unwrap();
    let base = across
        .iter()
        .find(|r| r[at("candidate")] == "tuned" && r[at("reference")] == "base")
        .unwrap();
    assert_eq!(base[at("authors_improved")], "3");
    assert_eq!(base[at("authors_measured")], "3");
    assert_eq!(base[at("confirmatory")], "true");
    assert!((base[at("p")].parse::<f64>().unwrap() - 0.125).abs() < 1e-12);

    // One accuracy row per author, arm and training seed, with n.
    let accuracy = csv(&study, "accuracy.csv");
    let n = study.protocol().exam.per_cell * 4;
    let seeds = &study.protocol().authors[0].training_seeds;
    let rows: Vec<_> = accuracy
        .iter()
        .filter(|r| r[0] == study.protocol().authors[0].seed.to_string() && r[1] == "tuned")
        .collect();
    assert_eq!(rows.len(), seeds.len());
    assert!(rows
        .iter()
        .all(|r| r[4] == n.to_string() && r[3] == n.to_string()));

    let text = std::fs::read_to_string(dir.join("report.txt")).unwrap();
    assert!(text.contains("tuned over retrieval (primary)"), "{text}");
    assert!(
        text.contains("first listed") && text.contains("Holm"),
        "{text}"
    );
}

#[test]
fn what_was_not_examined_is_absent_from_every_table_and_named_as_missing() {
    let study = run_study("report-missing", 2);
    let victim = study.protocol().authors[0].seed;
    let exam = study.author_dir(victim).join("exam").join("shared");
    std::fs::remove_file(exam.join("named.jsonl")).unwrap();
    // And a half-answered arm.
    let retrieval = exam.join("retrieval.jsonl");
    let text = std::fs::read_to_string(&retrieval).unwrap();
    let kept: String = text.lines().take(3).map(|l| format!("{l}\n")).collect();
    std::fs::write(&retrieval, kept).unwrap();

    let report = report::run(&study, Some(7)).unwrap();
    render::write(study.root(), &report).unwrap();
    let missing: Vec<&str> = report.unmeasured.iter().map(|u| u.what.as_str()).collect();
    assert!(
        missing.contains(&"shared/named") && missing.contains(&"shared/retrieval"),
        "{missing:?}"
    );
    let author = report.authors.iter().find(|a| a.seed == victim).unwrap();
    assert!(author
        .accuracy
        .iter()
        .all(|r| r.arm != "named" && r.arm != "retrieval"));
    let other = report.authors.iter().find(|a| a.seed != victim).unwrap();
    assert!(other.accuracy.iter().any(|r| r.arm == "named"));
    // The comparison with the missing arm is measured for the author that has it.
    let named = report
        .across
        .iter()
        .find(|r| r.comparison.candidate == "tuned" && r.comparison.reference == "named")
        .unwrap();
    assert_eq!(named.authors_measured, 1);
    assert!(csv(&study, "unmeasured.csv")
        .iter()
        .any(|r| r[1] == "shared/named"));
    assert!(render::table(&report).contains("not measured"));
}

#[test]
fn an_author_whose_corpus_stage_has_not_finished_is_listed_and_the_rest_are_reported() {
    let study = frozen("report-early", small_protocol(|_| {}), 2);
    let report = report::run(&study, Some(7)).unwrap();
    assert!(report.authors.is_empty());
    assert_eq!(report.unmeasured.len(), 2);
    assert!(report
        .across
        .iter()
        .all(|r| r.p.is_none() && r.mean_gain.is_none()));
    let written = render::write(study.root(), &report).unwrap();
    let across = std::fs::read_to_string(written.join("across_authors.csv")).unwrap();
    assert!(
        across.lines().skip(1).all(|l| l.contains(",0,2,0,0,0,,,,")),
        "{across}"
    );
}

#[test]
fn outcomes_that_contradict_their_manifest_stop_the_report() {
    let study = run_study("report-tampered", 1);
    let seed = study.protocol().authors[0].seed;
    let path = study
        .author_dir(seed)
        .join("exam")
        .join("shared")
        .join("base.jsonl");
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let model = lines[0]["model"].as_str().unwrap().to_string();
    std::fs::write(&path, text.replace(&model, "blake3:another-model")).unwrap();
    let message = format!("{:#}", report::run(&study, Some(7)).unwrap_err());
    assert!(
        message.contains("another-model") && message.contains("base"),
        "{message}"
    );

    // Outcomes of an adapter that is not the one its training run recorded.
    let study = run_study("report-swapped", 1);
    let seed = study.protocol().authors[0].seed;
    let training = study.protocol().authors[0].training_seeds[0];
    let manifest = study
        .author_dir(seed)
        .join("exam")
        .join(format!("tuned-s{training}"))
        .join("manifest.json");
    let text = std::fs::read_to_string(&manifest).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    let adapter = doc["arms"]["tuned"]["model"]["adapter_digest"]
        .as_str()
        .unwrap()
        .to_string();
    std::fs::write(&manifest, text.replace(&adapter, "sha256:other")).unwrap();
    let message = format!("{:#}", report::run(&study, Some(7)).unwrap_err());
    assert!(message.contains("tuned"), "{message}");
}

#[test]
fn the_command_writes_the_report_and_prints_the_table_with_the_seed_it_drew() {
    let study = run_study("report-cli", 2);
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_synthetic-author"))
        .args(["study", "report", "--study"])
        .arg(study.root())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stdout.contains("tuned over retrieval (primary)"),
        "{stdout}"
    );
    assert!(
        stderr.contains("no --bootstrap-seed given, using seed"),
        "{stderr}"
    );
    let drawn: u64 = stderr
        .split("using seed ")
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(study.root().join("report/report.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(json["bootstrap_seed"], drawn, "the drawn seed is recorded");
}

#[test]
fn the_paper_script_turns_the_report_into_the_tables_the_build_reads() {
    let study = run_study("report-paper", 2);
    let report = report::run(&study, Some(5)).unwrap();
    render::write(study.root(), &report).unwrap();
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../papers/persona-training/scripts/study_data.py");
    let run = |out: &std::path::Path, extra: &[&str]| {
        std::process::Command::new("python3")
            .arg(&script)
            .arg(study.root().join("report"))
            .arg(out)
            .args(extra)
            .output()
            .unwrap()
    };
    let out = study.root().join("paper");
    let made = run(&out, &[]);
    assert!(
        made.status.success(),
        "{}",
        String::from_utf8_lossy(&made.stderr)
    );

    let read = |name: &str| -> Vec<Vec<String>> {
        std::fs::read_to_string(out.join(name))
            .unwrap()
            .lines()
            .map(|l| l.split(',').map(str::to_string).collect())
            .collect()
    };
    let arms = read("arms.csv");
    assert_eq!(arms[0], ["seed", "arm", "right", "n"]);
    assert_eq!(arms.len() - 1, 2 * 5, "five arms for each of two authors");
    let first = &report.authors[0];
    let tuned = first
        .accuracy
        .iter()
        .find(|r| {
            r.arm == "tuned"
                && r.training_seed == Some(study.protocol().authors[0].training_seeds[0])
        })
        .unwrap();
    let row = arms
        .iter()
        .find(|r| r[0] == first.seed.to_string() && r[1] == "tuned")
        .unwrap();
    assert_eq!(row[2], tuned.right.to_string());
    assert_eq!(read("paired.csv").len() - 1, 2 * 6);
    assert!(read("cells.csv").len() > 1);

    // The second training seed is the one asked for with the flag, and a seed
    // that was not measured is an error, not a row.
    let second = study.root().join("paper-second");
    assert!(run(&second, &["--training-seed-index", "1"])
        .status
        .success());
    let other = std::fs::read_to_string(second.join("arms.csv")).unwrap();
    assert!(other.contains(&first.seed.to_string()));
    let refused = run(
        &study.root().join("paper-none"),
        &["--training-seed-index", "9"],
    );
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("training seeds"));
}
