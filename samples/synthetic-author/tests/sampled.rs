// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in measuring how stable a model's
// answers are under sampling, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: an arm can be asked each item K times under sampled decoding. Every
//! draw has a seed that is recorded, the draws repeat when the run seed is
//! passed back, the manifest says how the model decoded, and arms are
//! compared by a bootstrap over items beside the greedy result.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::json;
use splinter_sdk::agent::sven::model::{
    CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
};
use synthetic_author::arms::Outcome;
use synthetic_author::binding::{recorded, ModelIdentity};
use synthetic_author::cache::Cache;
use synthetic_author::exam::{Item, Version};
use synthetic_author::lab::{exam_items, Arm, ExamRun};
use synthetic_author::persona::Persona;
use synthetic_author::sampled::{
    draw_seed, read_sampled, resolve, run_arm_sampled, Draw, Requested, SampledDecoding,
    SampledOutcome,
};
use synthetic_author::sampled_report::render;

/// Takes the first listed option when the seed it was made with is even and
/// the last when it is odd, so that what it answers is a function of its seed.
struct ByParity(u64);

#[async_trait::async_trait]
impl ModelProvider for ByParity {
    fn name(&self) -> &str {
        "parity"
    }
    fn model_name(&self) -> &str {
        "parity-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let tool = &req.tools[0];
        let choice = tool.parameters["required"][0].as_str().unwrap().to_string();
        let labels = tool.parameters["properties"][&choice]["enum"]
            .as_array()
            .unwrap();
        let label = if self.0.is_multiple_of(2) {
            labels.first()
        } else {
            labels.last()
        }
        .unwrap()
        .clone();
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::ToolCall {
                index: 0,
                id: "c".into(),
                name: tool.name.clone(),
                arguments: json!({ choice: label }).to_string(),
            }),
            Ok(ResponseEvent::Done),
        ])))
    }
}

const PER_CELL: usize = 2;
const ARTICLES: usize = 20;
const DRAWS: usize = 5;
const TEMPERATURE: f32 = 0.5;

struct Study {
    cache: Cache,
    persona: Persona,
    items: Vec<Item>,
    built_with: Arc<Mutex<Vec<u64>>>,
}

fn study(tag: &str) -> Study {
    let dir: PathBuf = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("synthetic-author-sampled-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    let seed = splinter_sdk::model::seed::draw().value();
    let persona = Persona::generate(seed);
    Study {
        cache: Cache::open(dir).unwrap(),
        items: exam_items(&persona, seed, ARTICLES, PER_CELL),
        persona,
        built_with: Arc::default(),
    }
}

fn run() -> ExamRun {
    ExamRun {
        articles: ARTICLES,
        per_cell: PER_CELL,
        max_tokens: 64,
        model: ModelIdentity {
            base_digest: "sha256:base".into(),
            adapter_digest: None,
            decoding: "greedy".into(),
            decoding_seed: None,
        },
        exam: Version::V1,
        adopt: false,
        slots: 1,
    }
}

fn sampled(run_seed: u64, draws: usize) -> SampledDecoding {
    SampledDecoding {
        temperature: TEMPERATURE,
        draws,
        run_seed,
    }
}

impl Study {
    fn ask(&self, how: &SampledDecoding) -> anyhow::Result<Vec<SampledOutcome>> {
        let built_with = Arc::clone(&self.built_with);
        let provider_for = move |seed: u64| -> anyhow::Result<Arc<dyn ModelProvider>> {
            built_with.lock().unwrap().push(seed);
            Ok(Arc::new(ByParity(seed)))
        };
        run_arm_sampled(
            &self.cache,
            &self.persona,
            Arm::Tuned,
            &provider_for,
            &self.items,
            &run(),
            how,
        )
    }
}

#[test]
fn every_item_is_asked_k_times_each_draw_with_a_seed_that_is_recorded_and_used() {
    let s = study("draws");
    let run_seed = splinter_sdk::model::seed::draw().value();
    let outcomes = s.ask(&sampled(run_seed, DRAWS)).unwrap();

    assert_eq!(outcomes.len(), s.items.len());
    let recorded_seeds: Vec<u64> = outcomes
        .iter()
        .flat_map(|o| o.draws.iter().map(|d| d.seed))
        .collect();
    assert_eq!(recorded_seeds.len(), s.items.len() * DRAWS);
    assert_eq!(
        recorded_seeds.iter().collect::<BTreeSet<_>>().len(),
        recorded_seeds.len(),
        "no two draws share a seed"
    );
    let mut built = s.built_with.lock().unwrap().clone();
    let mut recorded_sorted = recorded_seeds.clone();
    built.sort_unstable();
    recorded_sorted.sort_unstable();
    assert_eq!(
        built, recorded_sorted,
        "the model was made with the recorded seeds"
    );

    for (item, outcome) in s.items.iter().zip(&outcomes) {
        assert_eq!(outcome.id, item.id);
        for draw in &outcome.draws {
            let picked = if draw.seed.is_multiple_of(2) {
                0
            } else {
                item.options.len() - 1
            };
            assert_eq!(
                draw.correct,
                item.options[picked] == item.expected,
                "{}",
                item.id
            );
        }
    }
    assert_eq!(read_sampled(&s.cache, Arm::Tuned).unwrap(), outcomes);
}

#[test]
fn the_draws_repeat_when_the_run_seed_is_passed_back_and_differ_under_another() {
    let run_seed = splinter_sdk::model::seed::draw().value();
    let first = study("repeat-a").ask(&sampled(run_seed, DRAWS)).unwrap();
    let again = study("repeat-b").ask(&sampled(run_seed, DRAWS)).unwrap();
    assert_eq!(first.len(), again.len());
    // The studies draw different authors, so compare the seeds, which depend
    // on the run seed and the place of the item alone.
    let seeds = |o: &[SampledOutcome]| -> Vec<Vec<u64>> {
        o.iter()
            .map(|o| o.draws.iter().map(|d| d.seed).collect())
            .collect()
    };
    let expected: Vec<Vec<u64>> = (0..first.len())
        .map(|i| (0..DRAWS).map(|k| draw_seed(run_seed, i, k)).collect())
        .collect();
    assert_eq!(seeds(&first), expected);
    assert_eq!(seeds(&again), expected);
    assert_ne!(
        seeds(&first),
        (0..first.len())
            .map(|i| (0..DRAWS)
                .map(|k| draw_seed(!run_seed, i, k))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_finished_item_is_not_asked_again_and_the_manifest_says_how_the_model_decoded() {
    let s = study("manifest");
    let run_seed = splinter_sdk::model::seed::draw().value();
    let how = sampled(run_seed, DRAWS);
    s.ask(&how).unwrap();
    let asked = s.built_with.lock().unwrap().len();
    s.ask(&how).unwrap();
    assert_eq!(
        s.built_with.lock().unwrap().len(),
        asked,
        "nothing was asked again"
    );

    let record = recorded(&s.cache.dir().join("exam"), "tuned-sampled")
        .unwrap()
        .unwrap();
    assert_eq!(record.model.decoding, format!("sample:{TEMPERATURE}"));
    assert_eq!(record.model.decoding_seed, Some(run_seed));
    assert_eq!(record.draws, Some(DRAWS));
    assert!(
        recorded(&s.cache.dir().join("exam"), "tuned")
            .unwrap()
            .is_none(),
        "the greedy arm keeps its own record"
    );
}

#[test]
fn outcomes_are_never_appended_to_under_another_temperature_seed_or_draw_count() {
    let s = study("refused");
    let run_seed = splinter_sdk::model::seed::draw().value();
    s.ask(&sampled(run_seed, DRAWS)).unwrap();
    for (other, named) in [
        (sampled(run_seed, DRAWS + 1), "draws"),
        (sampled(!run_seed, DRAWS), "decoding seed"),
        (
            SampledDecoding {
                temperature: TEMPERATURE * 2.0,
                ..sampled(run_seed, DRAWS)
            },
            "decoding",
        ),
    ] {
        let message = format!("{:#}", s.ask(&other).unwrap_err());
        assert!(message.contains(named), "{named}: {message}");
    }
}

fn outcome(id: usize, shares: &[bool]) -> SampledOutcome {
    SampledOutcome {
        id: format!("i{id}"),
        item: String::new(),
        model: String::new(),
        draws: shares
            .iter()
            .enumerate()
            .map(|(k, &correct)| Draw {
                seed: k as u64,
                correct,
                call: None,
            })
            .collect(),
    }
}

fn greedy(id: usize, correct: bool) -> Outcome {
    Outcome {
        id: format!("i{id}"),
        correct,
        topic_seen: true,
        combo_seen: true,
        combo_trained: None,
        tool_seen: true,
        call: None,
        item: String::new(),
        model: String::new(),
    }
}

#[test]
fn arms_are_compared_by_item_with_a_bootstrap_whose_seed_is_in_the_report_beside_greedy() {
    let always = |n| (0..n).map(|i| outcome(i, &[true; 4])).collect::<Vec<_>>();
    let never = |n| (0..n).map(|i| outcome(i, &[false; 4])).collect::<Vec<_>>();
    let greedy_a: Vec<Outcome> = (0..10).map(|i| greedy(i, true)).collect();
    let greedy_b: Vec<Outcome> = (0..10).map(|i| greedy(i, false)).collect();
    let bootstrap_seed = splinter_sdk::model::seed::draw().value();

    let text = render(
        &[("a", greedy_a.clone()), ("b", greedy_b.clone())],
        &[("a", always(10)), ("b", never(10))],
        &[("a", "b")],
        bootstrap_seed,
    );
    assert!(
        text.contains(&format!("bootstrap seed {bootstrap_seed}")),
        "{text}"
    );
    assert!(
        text.contains("| a | 1.000 (10/10) | 1.000 [1.000, 1.000] (4 draws) |"),
        "{text}"
    );
    assert!(
        text.contains("| b | 0.000 (0/10) | 0.000 [0.000, 0.000] (4 draws) |"),
        "{text}"
    );
    assert!(
        text.contains(
            "a over b: mean difference in share right 1.000 [1.000, 1.000] over 10 items"
        ),
        "{text}"
    );

    // Half the items won by a fixed margin: the mean is exact and the interval
    // is a function of the seed alone.
    let mixed = |n: usize| -> Vec<SampledOutcome> {
        (0..n)
            .map(|i| outcome(i, &[i % 2 == 0, i % 2 == 0, i % 2 == 0, i % 2 == 0]))
            .collect()
    };
    let compared = |seed| {
        render(
            &[("a", greedy_a.clone()), ("b", greedy_b.clone())],
            &[("a", mixed(20)), ("b", never(20))],
            &[("a", "b")],
            seed,
        )
    };
    assert!(compared(bootstrap_seed).contains("mean difference in share right 0.500"));
    assert_eq!(compared(bootstrap_seed), compared(bootstrap_seed));
}

#[test]
fn a_comparison_with_an_arm_that_has_no_draws_says_it_was_not_measured() {
    let text = render(
        &[("a", vec![greedy(0, true)])],
        &[("a", vec![outcome(0, &[true, false])])],
        &[("a", "b")],
        splinter_sdk::model::seed::draw().value(),
    );
    assert!(text.contains("a over b: not measured"), "{text}");
}

fn asked(draws: Option<usize>) -> Requested {
    Requested {
        temperature: TEMPERATURE,
        draws,
        run_seed: None,
    }
}

#[test]
fn a_sampled_run_continues_with_the_seed_and_draw_count_its_record_holds() {
    let s = study("resolve");
    let first = resolve(&s.cache, Arm::Tuned, asked(Some(DRAWS))).unwrap();
    let other = resolve(&s.cache, Arm::Tuned, asked(Some(DRAWS))).unwrap();
    assert_ne!(
        first.run_seed, other.run_seed,
        "nothing is recorded yet, so each call draws"
    );
    s.ask(&first).unwrap();
    let continued = resolve(&s.cache, Arm::Tuned, asked(None)).unwrap();
    assert_eq!(
        (continued.run_seed, continued.draws),
        (first.run_seed, DRAWS)
    );
}

#[test]
fn the_report_command_lists_the_sampled_arm_with_the_bootstrap_seed_it_was_given() {
    let s = study("report");
    s.cache.write_persona(&s.persona).unwrap();
    let run_seed = splinter_sdk::model::seed::draw().value();
    s.ask(&sampled(run_seed, DRAWS)).unwrap();
    let bootstrap_seed = splinter_sdk::model::seed::draw().value();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_synthetic-author"))
        .args(["report", "--arms", "tuned", "--bootstrap-seed"])
        .arg(bootstrap_seed.to_string())
        .arg("--cache")
        .arg(s.cache.dir())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{out:?}");
    assert!(
        stdout.contains(&format!("bootstrap seed {bootstrap_seed}")),
        "{stdout}"
    );
    assert!(stdout.contains(&format!("({DRAWS} draws)")), "{stdout}");
}
