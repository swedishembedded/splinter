// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fitting small decision models
// to labelled choices for its clients. If your team needs expertise in turning
// agent benchmark runs into auditable, leak-free fixtures then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Labelled decision fixtures from an ablation benchmark's run records.
//!
//! The benchmark (sven's ablation tool) runs controlled variants of a harness
//! over seeded tasks and leaves, per run, `record.json`, the task's
//! `task.json` (with the oracle labels: the right option for each question)
//! and `decisions.jsonl` (every decision its provider served). This turns those
//! files into one example per labelled choice question: the state, the options,
//! the labelled option and its provenance and the outcome of the run it came
//! from. Only the files' JSON is read, by the fields named here, so the
//! benchmark's types are not a dependency.
//!
//! Splits are decided per task family and seed, never per example: every
//! example of one task, from every variant, lands in one split, because the
//! same question asked under several variants is the same question. Within a
//! family the seeds are ordered by a hash of the plan's seed, the family and
//! the seed, and divided by the fractions; a family with fewer than three
//! seeds is all train (nothing can be held out without emptying it), and a
//! holdout family is all test.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The only fixture format this build writes.
pub const FIXTURE_SCHEMA: &str = "splinter.decision-fixture/1";

/// The run-record and task formats this reads.
const RUN_SCHEMA: &str = "sven.ablation.run/1";
const TASK_SCHEMA_PREFIX: &str = "sven.ablation.task/";

/// Where the label of an example comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelSource {
    /// The task's author knows the right answer: the generator labelled it.
    Oracle,
}

/// The part of the dataset an example is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    /// The examples to fit on.
    Train,
    /// The examples to choose between fits on.
    Validation,
    /// The examples held out, whole families included.
    Test,
}

/// Where an example came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRef {
    /// `<task id>/<variant id>`: the run directory under `runs/`.
    pub run: String,
    /// The task.
    pub task: String,
    /// The task family.
    pub suite: String,
    /// The task's seed.
    pub seed: u64,
    /// The benchmark variant that served the decision.
    pub variant: String,
    /// The digest the benchmark recorded for these inputs.
    pub inputs_digest: String,
    /// The question within the decision.
    pub question_id: String,
}

/// The question an example asks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExampleQuestion {
    /// The question's id.
    pub id: String,
    /// What the question asked.
    pub instruction: String,
    /// What the decision was shown.
    pub state: String,
}

/// One option of a question.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExampleOption {
    /// The option's id.
    pub id: String,
    /// What the option was described as.
    pub description: Option<String>,
}

/// What became of the run the decision was part of.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    /// Whether the run passed its task.
    pub task_passed: bool,
    /// Baseline tokens minus this run's, over the whole task; absent when the
    /// baseline did not run this task.
    pub llm_tokens_saved: Option<i64>,
    /// How many times the run fetched what shortened results left out.
    pub recovery_calls: u64,
}

/// One labelled choice question, with where it came from and its split.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Example {
    /// Always [`FIXTURE_SCHEMA`].
    pub schema: String,
    /// The decision point.
    pub decision_type: String,
    /// Where it came from.
    pub state_ref: StateRef,
    /// The question.
    pub question: ExampleQuestion,
    /// The options, in the order the decision was shown them.
    pub options: Vec<ExampleOption>,
    /// The labelled option's id.
    pub label: String,
    /// Its position in `options`.
    pub label_index: usize,
    /// Where the label comes from.
    pub label_source: LabelSource,
    /// What became of the run.
    pub outcome: Outcome,
    /// Which part of the dataset it is in.
    pub split: Split,
}

/// How tasks are divided into train, validation and test.
#[derive(Debug, Clone)]
pub struct SplitPlan {
    /// The split's seed; the same seed gives the same split.
    pub seed: u64,
    /// The share of a family's seeds held for validation.
    pub validation_fraction: f64,
    /// The share of a family's seeds held for test.
    pub test_fraction: f64,
    /// Whole task families that are test only.
    pub holdout_suites: Vec<String>,
}

impl SplitPlan {
    /// The usual shares, 0.2 of a family's seeds for validation and for test,
    /// with no holdout family.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            seed,
            validation_fraction: 0.2,
            test_fraction: 0.2,
            holdout_suites: Vec::new(),
        }
    }
}

fn rank(plan: &SplitPlan, suite: &str, seed: u64) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(plan.seed.to_le_bytes());
    h.update(suite.as_bytes());
    h.update(seed.to_le_bytes());
    h.finalize().into()
}

/// The split of each `(suite, seed)` group.
#[must_use]
pub fn assign_splits(
    groups: &BTreeSet<(String, u64)>,
    plan: &SplitPlan,
) -> BTreeMap<(String, u64), Split> {
    let mut out = BTreeMap::new();
    let mut families: BTreeMap<&str, Vec<u64>> = BTreeMap::new();
    for (suite, seed) in groups {
        families.entry(suite).or_default().push(*seed);
    }
    for (suite, mut seeds) in families {
        if plan.holdout_suites.iter().any(|h| h == suite) {
            out.extend(seeds.iter().map(|s| ((suite.to_string(), *s), Split::Test)));
            continue;
        }
        seeds.sort_by_key(|s| rank(plan, suite, *s));
        let n = seeds.len();
        let (validation, test) = if n < 3 {
            (0, 0)
        } else {
            let part = |f: f64| ((n as f64 * f).round() as usize).max(1);
            (part(plan.validation_fraction), part(plan.test_fraction))
        };
        for (i, seed) in seeds.iter().enumerate() {
            let split = if i < test {
                Split::Test
            } else if i < test + validation {
                Split::Validation
            } else {
                Split::Train
            };
            out.insert((suite.to_string(), *seed), split);
        }
    }
    out
}

// The fields of the benchmark's files that are read.

#[derive(Deserialize)]
struct RunRecord {
    schema: String,
    task_id: String,
    suite: String,
    seed: u64,
    variant: VariantRef,
    probe: Option<String>,
    status: RunStatus,
    tokens: Option<Tokens>,
    score: Option<Score>,
}

#[derive(Deserialize)]
struct VariantRef {
    id: String,
    role: String,
}

#[derive(Deserialize)]
struct RunStatus {
    state: String,
}

#[derive(Deserialize)]
struct Tokens {
    total: u64,
    #[serde(default)]
    recovery_expansions: u64,
}

#[derive(Deserialize)]
struct Score {
    passed: bool,
}

impl RunRecord {
    fn completed(&self) -> bool {
        self.status.state == "completed" && self.score.is_some()
    }
}

#[derive(Deserialize)]
struct Task {
    schema: String,
    oracle_labels: Vec<OracleLabel>,
}

#[derive(Deserialize)]
struct OracleLabel {
    state_contains: String,
    correct_option: String,
}

#[derive(Deserialize)]
struct ServedDecision {
    inputs_digest: String,
    state: String,
    questions: Vec<ServedQuestion>,
}

#[derive(Deserialize)]
struct ServedQuestion {
    id: String,
    kind: String,
    instruction: String,
    /// `(id, description)` of each option, in the order sent.
    options: Vec<(String, Option<String>)>,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn sorted_dirs(dir: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    let mut found: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    found.sort();
    Ok(found)
}

/// Every run record under `dir/runs/<task>/<variant>/record.json`, with its
/// directory. A bad record is an error, never skipped: a dataset over fewer
/// runs than were made would quietly change what it says.
fn load_runs(dir: &Path) -> Result<Vec<(std::path::PathBuf, RunRecord)>, String> {
    let mut runs = Vec::new();
    for task in sorted_dirs(&dir.join("runs"))? {
        for variant in sorted_dirs(&task)? {
            let path = variant.join("record.json");
            if !path.exists() {
                continue;
            }
            let record: RunRecord = read_json(&path)?;
            if record.schema != RUN_SCHEMA {
                return Err(format!(
                    "{}: run schema {:?} is not {RUN_SCHEMA:?}",
                    path.display(),
                    record.schema
                ));
            }
            runs.push((variant, record));
        }
    }
    Ok(runs)
}

fn label_for(task: &Task, state: &str, options: &[ExampleOption]) -> Option<usize> {
    let state = state.to_lowercase();
    task.oracle_labels
        .iter()
        .filter(|l| state.contains(&l.state_contains.to_lowercase()))
        .find_map(|l| options.iter().position(|o| o.id == l.correct_option))
}

/// The unsplit examples of one run.
fn run_examples(
    dir: &Path,
    run: &RunRecord,
    baseline_tokens: Option<u64>,
) -> Result<Vec<Example>, String> {
    let served = dir.join("decisions.jsonl");
    if !served.exists() || !run.completed() {
        return Ok(Vec::new());
    }
    let task: Task = read_json(&dir.join("task.json"))?;
    if !task.schema.starts_with(TASK_SCHEMA_PREFIX) {
        return Err(format!(
            "{}: task schema {:?} is not a {TASK_SCHEMA_PREFIX}* format",
            dir.display(),
            task.schema
        ));
    }
    let outcome = Outcome {
        task_passed: run.score.as_ref().is_some_and(|s| s.passed),
        llm_tokens_saved: baseline_tokens
            .zip(run.tokens.as_ref().map(|t| t.total))
            .map(|(b, t)| b as i64 - t as i64),
        recovery_calls: run.tokens.as_ref().map_or(0, |t| t.recovery_expansions),
    };
    let text =
        std::fs::read_to_string(&served).map_err(|e| format!("{}: {e}", served.display()))?;
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let decision: ServedDecision =
            serde_json::from_str(line).map_err(|e| format!("{}: {e}", served.display()))?;
        for q in decision.questions.iter().filter(|q| q.kind == "choice") {
            let options: Vec<ExampleOption> = q
                .options
                .iter()
                .map(|(id, description)| ExampleOption {
                    id: id.clone(),
                    description: description.clone(),
                })
                .collect();
            let Some(label_index) = label_for(&task, &decision.state, &options) else {
                continue;
            };
            // The same question asked in two option orders is one example.
            let mut ids: Vec<&str> = options.iter().map(|o| o.id.as_str()).collect();
            ids.sort_unstable();
            if !seen.insert((decision.state.clone(), ids.join("\u{1}"), q.id.clone())) {
                continue;
            }
            out.push(Example {
                schema: FIXTURE_SCHEMA.to_string(),
                decision_type: "tool_routing".to_string(),
                state_ref: StateRef {
                    run: format!("{}/{}", run.task_id, run.variant.id),
                    task: run.task_id.clone(),
                    suite: run.suite.clone(),
                    seed: run.seed,
                    variant: run.variant.id.clone(),
                    inputs_digest: decision.inputs_digest.clone(),
                    question_id: q.id.clone(),
                },
                question: ExampleQuestion {
                    id: q.id.clone(),
                    instruction: q.instruction.clone(),
                    state: decision.state.clone(),
                },
                label: options[label_index].id.clone(),
                label_index,
                options,
                label_source: LabelSource::Oracle,
                outcome: outcome.clone(),
                split: Split::Train,
            });
        }
    }
    Ok(out)
}

/// The examples of every run under `dir`, split by `plan`.
///
/// # Errors
///
/// A record, task or decision log that cannot be read, or a holdout family no
/// run belongs to (a misspelt name would otherwise hold nothing out).
pub fn build(dir: &Path, plan: &SplitPlan) -> Result<Vec<Example>, String> {
    let runs = load_runs(dir)?;
    let known: BTreeSet<&str> = runs.iter().map(|(_, r)| r.suite.as_str()).collect();
    if let Some(unknown) = plan
        .holdout_suites
        .iter()
        .find(|h| !known.contains(h.as_str()))
    {
        return Err(format!(
            "holdout suite {unknown:?} is not among the runs' suites {known:?}"
        ));
    }
    let baseline_tokens: BTreeMap<&str, u64> = runs
        .iter()
        .filter(|(_, r)| r.probe.is_none() && r.variant.role == "baseline")
        .filter_map(|(_, r)| Some((r.task_id.as_str(), r.tokens.as_ref()?.total)))
        .collect();
    let mut examples = Vec::new();
    for (run_dir, run) in runs.iter().filter(|(_, r)| r.probe.is_none()) {
        examples.extend(run_examples(
            run_dir,
            run,
            baseline_tokens.get(run.task_id.as_str()).copied(),
        )?);
    }
    let groups: BTreeSet<(String, u64)> = examples
        .iter()
        .map(|e| (e.state_ref.suite.clone(), e.state_ref.seed))
        .collect();
    let splits = assign_splits(&groups, plan);
    for e in &mut examples {
        e.split = splits[&(e.state_ref.suite.clone(), e.state_ref.seed)];
    }
    examples.sort_by(|a, b| {
        (a.split, &a.state_ref.run, &a.question.state).cmp(&(
            b.split,
            &b.state_ref.run,
            &b.question.state,
        ))
    });
    Ok(examples)
}

/// The examples as JSON lines.
///
/// # Errors
///
/// An example that does not serialize.
pub fn to_jsonl(examples: &[Example]) -> Result<String, String> {
    let mut out = String::new();
    for e in examples {
        out.push_str(&serde_json::to_string(e).map_err(|e| e.to_string())?);
        out.push('\n');
    }
    Ok(out)
}
