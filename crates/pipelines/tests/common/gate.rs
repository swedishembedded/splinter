// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixtures the release specs share: datasets of numbered facts, scripted
//! models that know some topics of them, a trainer double whose adapter
//! file says what its candidate knows, and a stand-in for the `brain`
//! binary that serves such an adapter.
//!
//! The stand-in is a test double of `brain serve`, written by the test: it
//! prints brain's startup line (with the adapter's real digest, or a wrong
//! one when asked to lie), writes its API key and ready marker, and answers
//! the OpenAI-compatible chat route from what the adapter file says the
//! candidate knows - when asked to decode greedily (temperature zero); a
//! sampled request gets a draw that agrees with nothing. An honest one
//! answers what the in-process candidate answers, with a trailing newline
//! (whitespace a comparison of answers ignores); a divergent one answers in
//! other words that grade the same (upper case, a full stop). Like brain on one
//! device, it does not start while another process holds the device: while
//! the file [`super::fake_brain::device_lock`] names exists. It tests the serve check's
//! plumbing, not brain.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::annotation::Strength;
use splinter_core::chat::WireMessage;
use splinter_core::clock::FixedClock;
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, Privileged, PrivilegedKind, Task};
use splinter_core::model_ref::ModelRef;
use splinter_core::release::ReleaseId;
use splinter_core::terms::Terms;
use splinter_core::training::{HeldOutScore, Regime, TrainingCurve};
use splinter_data::holdout::holdout_split_grouped;
use splinter_data::{Objective, Projection, Record, RecordBody, RecordMetadata, Strip};
use splinter_eval::gate::GateConfig;
use splinter_model::train::Trained;
use splinter_model::train::TrainedPreference;
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_pipelines::datasets::store_dataset;
use splinter_pipelines::release::{arm, release, ReleaseRequest, Released};
use splinter_pipelines::train::{
    train, Candidate, TrainPlan, TrainRequest, Trainer, Tuning, DEFAULT_REPLAY_FRACTION,
};
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

use super::fake_brain::fake_brain;
pub use super::fake_brain::Brain;
use super::{config, Scratch, Scripted};

/// The bytes of the base checkpoint every gate spec's policy sits on: a
/// safetensors file holding no tensor, so a load that reaches it fails
/// with an error instead of reading weights.
pub const BASE_BYTES: &[u8] = b"\x02\x00\x00\x00\x00\x00\x00\x00{}";

/// The instant every gate spec's clock reads.
pub const NOW: &str = "2026-09-30T12:00:00.000Z";

/// The anchor suite's topic: general tasks every model here knows unless
/// it forgot them.
pub const ANCHOR: &str = "anchor";

/// Question `i` about `topic`.
pub fn question(topic: &str, i: usize) -> String {
    format!("What is the {topic} code number {i}?")
}

/// The answer to [`question`].
pub fn answer(topic: &str, i: usize) -> String {
    format!("{topic}-{i}")
}

/// What a model that knows `topics` replies to `prompt`: the answer to the
/// question in it about a known topic, and a shrug otherwise.
pub fn reply(topics: &[String], prompt: &str) -> String {
    let asked = prompt.split("What is the ").nth(1).and_then(|rest| {
        let (topic, rest) = rest.split_once(" code number ")?;
        let number: String = rest.chars().take_while(char::is_ascii_digit).collect();
        Some((topic.to_string(), number.parse::<usize>().ok()?))
    });
    match asked {
        Some((topic, i)) if topics.contains(&topic) => answer(&topic, i),
        _ => "I do not know.".into(),
    }
}

/// A scripted model that knows `topics`.
pub fn knower(topics: &[&str]) -> Model {
    let topics: Vec<String> = topics.iter().map(|t| t.to_string()).collect();
    let scripted = Scripted::new(move |prompt| reply(&topics, prompt));
    Model::new(Arc::new(scripted), "scripted/knower")
}

/// A recall task: `question(topic, i)`, referenced by its answer,
/// declaring its topic as the concept it exercises.
pub fn fact(topic: &str, i: usize) -> Task {
    Task::new(
        "recall",
        Vec::new(),
        Environment::closed_book(),
        question(topic, i),
        vec![Privileged {
            kind: PrivilegedKind::Reference,
            content: answer(topic, i),
            span: None,
        }],
    )
    .unwrap()
    .with_concepts([topic])
    .unwrap()
}

/// Stores an sft-final dataset of the answers to `n` facts about `topic`;
/// the newest tenth is what training holds out. Only those held-out tasks
/// are put in the task store, the only ones a probe looks up: every task
/// is an fsynced write, and the rest would only slow the specs down.
pub fn dataset(ctx: &Context, topic: &str, n: usize) -> DatasetId {
    dataset_of(ctx, topic, &(0..n).collect::<Vec<_>>())
}

/// Stores an sft-final dataset of the answers to the facts about `topic`
/// numbered `facts`, in that order; as with [`dataset`], only the tasks of
/// the records the holdout rule holds out are put in the task store.
pub fn dataset_of(ctx: &Context, topic: &str, facts: &[usize]) -> DatasetId {
    dataset_under(ctx, topic, facts, None)
}

/// [`dataset_of`], its records opening with `prompt` as the system turn
/// when one is given, and the manifest naming it.
pub fn dataset_under(
    ctx: &Context,
    topic: &str,
    facts: &[usize],
    prompt: Option<&str>,
) -> DatasetId {
    dataset_made(ctx, topic, facts, prompt, &[], None)
}

/// [`dataset_under`], every record naming `sources` (the contents of text
/// parts) among the text it was built from.
pub fn dataset_citing(
    ctx: &Context,
    topic: &str,
    facts: &[usize],
    prompt: Option<&str>,
    sources: &[splinter_core::digest::Digest],
) -> DatasetId {
    dataset_made(ctx, topic, facts, prompt, sources, None)
}

/// [`dataset`] of `n` facts whose manifest states `terms`.
pub fn dataset_with_terms(ctx: &Context, topic: &str, n: usize, terms: Option<Terms>) -> DatasetId {
    dataset_made(ctx, topic, &(0..n).collect::<Vec<_>>(), None, &[], terms)
}

fn dataset_made(
    ctx: &Context,
    topic: &str,
    facts: &[usize],
    prompt: Option<&str>,
    sources: &[splinter_core::digest::Digest],
    terms: Option<Terms>,
) -> DatasetId {
    let positions: Vec<usize> = (0..facts.len()).collect();
    let held_out: Vec<usize> = holdout_split_grouped(&positions, |_| None)
        .map(|(_, held)| held.into_iter().copied().collect())
        .unwrap_or_default();
    let mut records = Vec::new();
    for (position, &i) in facts.iter().enumerate() {
        let task = fact(topic, i);
        if held_out.contains(&position) {
            ctx.tasks().put(&task).unwrap();
        }
        let message = |role: &str, content: String, train: bool| WireMessage {
            role: role.into(),
            content,
            tool_calls: Vec::new(),
            tool_call_id: None,
            train,
        };
        records.push(Record {
            body: RecordBody::Chat {
                messages: vec![
                    message("user", question(topic, i), false),
                    message("assistant", answer(topic, i), true),
                ],
            },
            metadata: RecordMetadata {
                group: None,
                experiences: Vec::new(),
                task: Some(task.task.id.clone()),
                sources: sources.to_vec(),
                view: "sft-final".into(),
                objective: Objective::Sft,
            },
        });
    }
    let projection = Projection {
        view: "sft-final".into(),
        objective: Objective::Sft,
        strip: Some(Strip::All),
        min_strength: Some(Strength::Formal),
        records,
        excluded: BTreeMap::new(),
        system_prompt: None,
        terms,
    };
    let projection = match prompt {
        Some(prompt) => projection.with_system_prompt(prompt),
        None => projection,
    };
    store_dataset(ctx, &projection, false).unwrap().id
}

/// Stores a preference dataset over `n` facts about `topic`: each pair
/// prefers the answer over a shrug. As with [`dataset`], only the held-out
/// tasks are put in the task store.
pub fn preference_dataset(ctx: &Context, topic: &str, n: usize) -> DatasetId {
    let held_out_from = n - (n / 10).max(1);
    let message = |role: &str, content: String, train: bool| WireMessage {
        role: role.into(),
        content,
        tool_calls: Vec::new(),
        tool_call_id: None,
        train,
    };
    let mut records = Vec::new();
    for i in 0..n {
        let task = fact(topic, i);
        if i >= held_out_from {
            ctx.tasks().put(&task).unwrap();
        }
        records.push(Record {
            body: RecordBody::Preference {
                prompt: vec![message("user", question(topic, i), false)],
                chosen: message("assistant", answer(topic, i), true),
                rejected: message("assistant", "I do not know.".into(), true),
            },
            metadata: RecordMetadata {
                group: None,
                experiences: Vec::new(),
                task: Some(task.task.id.clone()),
                sources: Vec::new(),
                view: "preference".into(),
                objective: Objective::Dpo,
            },
        });
    }
    let projection = Projection {
        view: "preference".into(),
        objective: Objective::Dpo,
        strip: Some(Strip::All),
        min_strength: Some(Strength::Formal),
        records,
        excluded: BTreeMap::new(),
        system_prompt: None,
        terms: None,
    };
    store_dataset(ctx, &projection, false).unwrap().id
}

/// Writes `n` anchor facts as an anchor file in `dir`.
pub fn anchor_file(dir: &Path, n: usize) -> PathBuf {
    let path = dir.join("anchor.jsonl");
    let lines: Vec<String> = (0..n)
        .map(|i| {
            serde_json::json!({ "instruction": question(ANCHOR, i), "reference": answer(ANCHOR, i) })
                .to_string()
        })
        .collect();
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    path
}

/// Facts per dataset: a tenth, six, are held out - enough for six
/// discordant wins to be significant at the default alpha (1/64 < 0.05).
pub const FACTS: usize = 60;

/// Writes the policy base of `ctx`'s configuration: [`BASE_BYTES`].
pub fn put_base(ctx: &Context) {
    let base = &ctx.config().policy_base;
    std::fs::create_dir_all(base).unwrap();
    std::fs::write(base.join("model.safetensors"), BASE_BYTES).unwrap();
    // A checkpoint states the largest context it supports.
    std::fs::write(
        base.join("config.json"),
        br#"{"max_position_embeddings": 4096}"#,
    )
    .unwrap();
}

/// A context over a fresh scratch root whose policy base exists (holding
/// [`BASE_BYTES`]), whose clock reads [`NOW`], whose base arm knows only the
/// anchor topic, and whose `brain` is `brain`.
pub fn gate_context(test: &str, brain: Brain) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let mut config = config(&scratch);
    config.brain_binary = match brain {
        Brain::Missing => None,
        Brain::Honest
        | Brain::WrongDigest
        | Brain::Divergent
        | Brain::Different
        | Brain::Crashing => Some(fake_brain(&scratch.0, brain)),
    };
    let base = arm(&config, None);
    let ctx = Context::new(config, false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new(NOW)))
        .with_model(base, knower(&[ANCHOR]));
    // What the serve check compares is what answers say: words, not the
    // embedding model.
    ctx.set_embedder(Arc::new(BagOfWords));
    put_base(&ctx);
    (scratch, ctx)
}

/// Texts embedded as the words they hold: lower-cased, split at everything
/// that is not a letter or a digit, hashed into a fixed number of buckets. Two
/// texts of the same words are one point, and texts sharing none are apart.
pub struct BagOfWords;

impl splinter_knowledge::retrieve::Embedder for BagOfWords {
    fn name(&self) -> String {
        "bag-of-words".into()
    }

    fn embed(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, splinter_knowledge::retrieve::EmbedError> {
        const BUCKETS: usize = 64;
        Ok(texts
            .iter()
            .map(|text| {
                let mut v = vec![0.0f32; BUCKETS];
                for word in text
                    .to_lowercase()
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|w| !w.is_empty())
                {
                    let hash = word.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
                    });
                    v[usize::try_from(hash % BUCKETS as u64).unwrap_or(0)] += 1.0;
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1.0);
                v.into_iter().map(|x| x / norm).collect()
            })
            .collect())
    }
}

/// The gate with a short server start, for a stand-in that starts at once.
pub fn fast_gate() -> GateConfig {
    GateConfig {
        serve_startup_secs: 30,
        ..GateConfig::default()
    }
}

/// Trains a candidate knowing `knows` on [`FACTS`] facts about `topic`,
/// from the policy.
pub fn candidate(ctx: &Context, topic: &str, knows: &[&str]) -> (Candidate, FakeTrainer) {
    candidate_on(ctx, dataset(ctx, topic, FACTS), knows)
}

/// Trains a candidate knowing `knows` on `data`, from the policy.
pub fn candidate_on(ctx: &Context, data: DatasetId, knows: &[&str]) -> (Candidate, FakeTrainer) {
    let trainer = FakeTrainer::knowing(knows);
    let candidate = train(
        ctx,
        &TrainRequest {
            datasets: vec![data.to_string()],
            rehearsal: None,
            from: policy(),
            replay_fraction: DEFAULT_REPLAY_FRACTION,
            steps: Some(1),
            rank: 4,
            beta: None,
            tuning: Tuning::default(),
        },
        &trainer,
        &CancelToken::new(),
    )
    .unwrap();
    (candidate, trainer)
}

/// The ways a variant of a fact is worded: each keeps the question's own
/// words, so a scripted model that knows the topic answers it, and adds
/// enough others to be no near duplicate of the question.
const WORDINGS: [&str; 3] = [
    "Please answer this question:",
    "Here is a question for you:",
    "Quick one before we go on:",
];

/// `fact(topic, i)` asked in other words ([`WORDINGS`]), stored.
pub fn variant_of_fact(ctx: &Context, topic: &str, i: usize, wording: usize) -> Task {
    let instruction = format!("{} {}", WORDINGS[wording], question(topic, i));
    variant_asking(ctx, topic, i, &instruction)
}

/// `fact(topic, i)` asking `instruction`, stored.
pub fn variant_asking(ctx: &Context, topic: &str, i: usize, instruction: &str) -> Task {
    let variant = fact(topic, i).with_instruction(instruction).unwrap();
    ctx.tasks().put(&variant).unwrap();
    variant
}

/// Stores a task set of variants: `variants` of the facts about `topic`
/// numbered `facts`, each in the first `wordings` ways, each recorded as a
/// variant of its fact.
pub fn variant_set(ctx: &Context, topic: &str, facts: &[usize], wordings: usize) -> TaskSetId {
    let mut members = Vec::new();
    for &i in facts {
        for wording in 0..wordings {
            members.push((i, variant_of_fact(ctx, topic, i, wording)));
        }
    }
    variant_set_of(ctx, topic, members)
}

/// Stores a task set of `variants`, each with the fact about `topic` it is
/// a variant of.
pub fn variant_set_of(ctx: &Context, topic: &str, variants: Vec<(usize, Task)>) -> TaskSetId {
    let members = variants
        .into_iter()
        .map(|(i, task)| TaskEntry {
            task: task.task.id,
            generator: Some("scripted/generator".into()),
            prompt: None,
            variant_of: Some(fact(topic, i).task.id),
            subject: None,
        })
        .collect();
    ctx.tasks()
        .put_set(&TaskSet {
            name: format!("variants of {topic} facts"),
            members,
        })
        .unwrap()
}

/// Runs the gate on `candidate` under `default`; a release it writes is
/// then served to the context as its adapter says.
pub fn decide(ctx: &Context, candidate: &Candidate) -> Released {
    let request = ReleaseRequest {
        gate: fast_gate(),
        ..ReleaseRequest::new(candidate.candidate.clone())
    };
    let decided = release(ctx, &request, &CancelToken::new()).unwrap();
    if let Some(id) = &decided.release {
        serve_release(ctx, id);
    }
    decided
}

/// Trains and releases a candidate knowing `knows` on `topic`; it must
/// pass.
pub fn released(ctx: &Context, topic: &str, knows: &[&str]) -> ReleaseId {
    let (candidate, _) = candidate(ctx, topic, knows);
    let decided = decide(ctx, &candidate);
    assert!(decided.gate.passed, "{:#?}", decided.gate);
    decided.release.unwrap()
}

/// Hands `ctx` the model the release `id` serves: its adapter file says
/// what it knows. The gate grades a champion through its release's copy of
/// the adapter.
pub fn serve_release(ctx: &Context, id: &ReleaseId) {
    let stored = ctx.releases().get(id).unwrap();
    let adapter: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&stored.artifact).unwrap()).unwrap();
    let knows: Vec<&str> = adapter["knows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    ctx.add_model(arm(ctx.config(), Some(&stored.artifact)), knower(&knows));
}

/// A trainer double for both regimes: its adapter file is JSON naming the
/// topics its candidate knows, and it hands the context a scripted model
/// knowing them under the candidate's arm. Keeps every plan it was given,
/// which of its methods trained it, and how many bases were resident when
/// it started.
pub struct FakeTrainer {
    knows: Vec<String>,
    /// The plans it trained.
    pub plans: Mutex<Vec<TrainPlan>>,
    /// The regime of each method called, in order: `train` is SFT,
    /// `train_preference` DPO.
    pub called: Mutex<Vec<Regime>>,
    /// The context's resident bases as each training started.
    pub resident_at_start: Mutex<Vec<usize>>,
    /// The curve its next supervised candidate reports; `None` reports a
    /// run that monitored nothing and carries its last step.
    pub curve: Mutex<Option<TrainingCurve>>,
    /// The steps whose evaluations its next supervised run keeps an adapter
    /// of when asked to; the adapters carry the step and nothing else, for
    /// the test to serve as it likes.
    pub evaluations: Mutex<Vec<u32>>,
}

impl FakeTrainer {
    /// A trainer whose candidates know `topics`.
    pub fn knowing(topics: &[&str]) -> Self {
        Self {
            knows: topics.iter().map(|t| t.to_string()).collect(),
            plans: Mutex::new(Vec::new()),
            called: Mutex::new(Vec::new()),
            resident_at_start: Mutex::new(Vec::new()),
            curve: Mutex::new(None),
            evaluations: Mutex::new(Vec::new()),
        }
    }

    /// Writes the adapter of each evaluation asked for.
    fn kept_evaluations(&self, ctx: &Context, plan: &TrainPlan) -> Vec<(u32, PathBuf)> {
        if !plan.tuning.keep_evaluations {
            return Vec::new();
        }
        let dir = plan.dir.join("evaluations");
        std::fs::create_dir_all(&dir).unwrap();
        self.evaluations
            .lock()
            .unwrap()
            .iter()
            .map(|step| {
                let path = dir.join(format!("step-{step}.safetensors"));
                std::fs::write(&path, serde_json::json!({ "step": step }).to_string()).unwrap();
                let kept = ctx
                    .artifacts()
                    .put_file(
                        &path,
                        &splinter_store::artifacts::ArtifactSpec::new("adapter", "test-trainer")
                            .with_extension(".safetensors")
                            .with_sha256(),
                    )
                    .unwrap();
                (*step, ctx.artifacts().path(&kept.digest).unwrap())
            })
            .collect()
    }
}

/// A score nothing measured.
pub fn unscored() -> HeldOutScore {
    HeldOutScore {
        loss: None,
        token_accuracy: None,
        positions: 0,
        records: 0,
        skipped: 0,
    }
}

/// What [`FakeTrainer`] wrote for a plan.
struct FakeAdapter {
    adapter: PathBuf,
    digest: String,
    record: PathBuf,
    records: usize,
}

impl FakeTrainer {
    /// Writes the adapter and training record for `plan` and serves it.
    fn fake(&self, ctx: &Context, plan: &TrainPlan) -> FakeAdapter {
        self.resident_at_start
            .lock()
            .unwrap()
            .push(ctx.resident_bases());
        self.plans.lock().unwrap().push(plan.clone());
        let adapter = plan.dir.join("adapter.safetensors");
        let bytes =
            serde_json::json!({ "candidate": plan.candidate, "knows": self.knows }).to_string();
        std::fs::write(&adapter, &bytes).unwrap();
        let record = plan.dir.join("training.json");
        std::fs::write(&record, r#"{"trainer":"fake"}"#).unwrap();
        // Training keeps the adapter as an artifact; the model is served from
        // the path every later reader will find it at.
        let kept = ctx
            .artifacts()
            .put_file(
                &adapter,
                &splinter_store::artifacts::ArtifactSpec::new("adapter", "test-trainer")
                    .with_extension(".safetensors")
                    .with_sha256(),
            )
            .unwrap();
        let kept_path = ctx.artifacts().path(&kept.digest).unwrap();
        let knows: Vec<&str> = self.knows.iter().map(String::as_str).collect();
        ctx.add_model(arm(ctx.config(), Some(&kept_path)), knower(&knows));
        let records = plan
            .datasets
            .iter()
            .map(|d| d.manifest.counts.records)
            .sum();
        FakeAdapter {
            adapter: kept_path,
            digest: Digest::sha256_of(bytes.as_bytes()).to_string(),
            record,
            records,
        }
    }
}

impl Trainer for FakeTrainer {
    fn train(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        _cancel: &CancelToken,
    ) -> Result<Trained, OrchestratorError> {
        self.called.lock().unwrap().push(Regime::Sft);
        let fake = self.fake(ctx, plan);
        let evaluations = self.kept_evaluations(ctx, plan);
        Ok(Trained {
            adapter: fake.adapter,
            adapter_digest: fake.digest,
            base_digest: Digest::sha256_of(BASE_BYTES).to_string(),
            training_record: fake.record,
            records: fake.records,
            block: 0,
            base: unscored(),
            tuned: unscored(),
            curve: self
                .curve
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| TrainingCurve::unmonitored(plan.steps, plan.steps)),
            evaluations,
        })
    }

    fn train_preference(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        _cancel: &CancelToken,
    ) -> Result<TrainedPreference, OrchestratorError> {
        self.called.lock().unwrap().push(Regime::Dpo);
        let fake = self.fake(ctx, plan);
        Ok(TrainedPreference {
            adapter: fake.adapter,
            adapter_digest: fake.digest,
            base_digest: Digest::sha256_of(BASE_BYTES).to_string(),
            training_record: fake.record,
            records: fake.records,
            block: 0,
            beta: plan.beta,
            reference_adapter: None,
            train_score: None,
            held_out_score: None,
        })
    }
}

/// The policy's own reference.
pub fn policy() -> ModelRef {
    ModelRef::policy_default()
}
