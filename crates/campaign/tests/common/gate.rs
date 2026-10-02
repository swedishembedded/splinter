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
//! the file [`device_lock`] names exists. It tests the serve check's
//! plumbing, not brain.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use splinter_agent::solve::Model;
use splinter_campaign::datasets::store_dataset;
use splinter_campaign::model_ref::ModelRef;
use splinter_campaign::release::gate::GateConfig;
use splinter_campaign::release::{arm, release, ReleaseId, ReleaseRequest, Released};
use splinter_campaign::train::{
    train, Candidate, Regime, TrainPlan, TrainRequest, Trainer, DEFAULT_REPLAY_FRACTION,
};
use splinter_campaign::{CampaignError, Context};
use splinter_lab::WireMessage;
use splinter_policy::train::TrainedPreference;
use splinter_policy::train::{HeldOutScore, Trained};
use splinter_record::annotation::Strength;
use splinter_record::clock::FixedClock;
use splinter_record::digest::Digest;
use splinter_record::experience::{Environment, Privileged, PrivilegedKind, Task};
use splinter_record::tasks::{TaskEntry, TaskSet, TaskSetId};
use splinter_views::{
    DatasetId, Objective, Projection, Record, RecordBody, RecordMetadata, Strip, WriteOptions,
};
use sven_sdk::CancelToken;

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
/// the newest tenth are put in the task store.
pub fn dataset_of(ctx: &Context, topic: &str, facts: &[usize]) -> DatasetId {
    let n = facts.len();
    let held_out_from = n - (n / 10).max(1);
    let mut records = Vec::new();
    for (position, &i) in facts.iter().enumerate() {
        let task = fact(topic, i);
        if position >= held_out_from {
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
                experiences: Vec::new(),
                task: Some(task.task.id.clone()),
                sources: Vec::new(),
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
    };
    store_dataset(ctx, &projection, WriteOptions::default())
        .unwrap()
        .id
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
    };
    store_dataset(ctx, &projection, WriteOptions::default())
        .unwrap()
        .id
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

/// How the `brain` binary behaves in a spec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Brain {
    /// There is none.
    Missing,
    /// The stand-in, reporting the adapter's real digest.
    Honest,
    /// The stand-in, reporting a digest that is not the adapter's.
    WrongDigest,
    /// The stand-in, reporting the real digest and answering in other
    /// words than the candidate in-process, graded the same.
    Divergent,
}

/// The file standing for the device being held: while it exists, the
/// `brain` stand-in in `scratch` refuses to start.
pub fn device_lock(scratch: &Scratch) -> PathBuf {
    scratch.0.join(DEVICE_LOCK)
}

/// [`device_lock`]'s name in the scratch directory.
const DEVICE_LOCK: &str = "device.lock";

/// Facts per dataset: a tenth, six, are held out - enough for six
/// discordant wins to be significant at the default alpha (1/64 < 0.05).
pub const FACTS: usize = 60;

/// Writes the policy base of `ctx`'s configuration: [`BASE_BYTES`].
pub fn put_base(ctx: &Context) {
    let base = &ctx.config().policy_base;
    std::fs::create_dir_all(base).unwrap();
    std::fs::write(base.join("model.safetensors"), BASE_BYTES).unwrap();
}

/// A context over a fresh scratch root whose policy base exists (holding
/// [`BASE_BYTES`]), whose clock reads [`NOW`], whose base arm knows only the
/// anchor topic, and whose `brain` is `brain`.
pub fn gate_context(test: &str, brain: Brain) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let mut config = config(&scratch);
    config.brain_binary = match brain {
        Brain::Missing => None,
        Brain::Honest | Brain::WrongDigest | Brain::Divergent => {
            Some(fake_brain(&scratch.0, brain))
        }
    };
    let base = arm(&config, None);
    let ctx = Context::new(config, false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new(NOW)))
        .with_model(base, knower(&[ANCHOR]));
    put_base(&ctx);
    (scratch, ctx)
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
            from: policy(),
            replay_fraction: DEFAULT_REPLAY_FRACTION,
            steps: 1,
            rank: 4,
            beta: None,
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
        serde_json::from_slice(&std::fs::read(&stored.adapter).unwrap()).unwrap();
    let knows: Vec<&str> = adapter["knows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    ctx.add_model(arm(ctx.config(), Some(&stored.adapter)), knower(&knows));
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
}

impl FakeTrainer {
    /// A trainer whose candidates know `topics`.
    pub fn knowing(topics: &[&str]) -> Self {
        Self {
            knows: topics.iter().map(|t| t.to_string()).collect(),
            plans: Mutex::new(Vec::new()),
            called: Mutex::new(Vec::new()),
            resident_at_start: Mutex::new(Vec::new()),
        }
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
                &splinter_record::artifacts::ArtifactSpec::new("adapter", "test-trainer")
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
    ) -> Result<Trained, CampaignError> {
        self.called.lock().unwrap().push(Regime::Sft);
        let fake = self.fake(ctx, plan);
        Ok(Trained {
            adapter: fake.adapter,
            adapter_digest: fake.digest,
            base_digest: Digest::sha256_of(BASE_BYTES).to_string(),
            training_record: fake.record,
            records: fake.records,
            block: 0,
            base: unscored(),
            tuned: unscored(),
        })
    }

    fn train_preference(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        _cancel: &CancelToken,
    ) -> Result<TrainedPreference, CampaignError> {
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

/// Writes the `brain` stand-in into `dir`; see the module documentation.
fn fake_brain(dir: &Path, brain: Brain) -> PathBuf {
    let path = dir.join("brain");
    let python = |yes: bool| if yes { "True" } else { "False" };
    let script = FAKE_BRAIN
        .replace("@WRONG@", python(brain == Brain::WrongDigest))
        .replace("@DIVERGENT@", python(brain == Brain::Divergent))
        .replace("@DEVICE@", &dir.join(DEVICE_LOCK).display().to_string());
    std::fs::write(&path, script).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    path
}

const FAKE_BRAIN: &str = r#"#!/usr/bin/env python3
# A test double of `brain serve`: see the fixtures' module documentation.
import hashlib, http.server, json, os, re, sys

if os.path.exists("@DEVICE@"):
    print("the device is held by another process", file=sys.stderr, flush=True)
    sys.exit(1)
args = sys.argv[1:]
assert args[0] == "serve", args
opts = dict(zip(args[1::2], args[2::2]))
adapter = open(opts["--adapter"], "rb").read()
digest = "sha256:" + hashlib.sha256(adapter).hexdigest()
if @WRONG@:
    digest = "sha256:" + "0" * 64
knows = json.loads(adapter)["knows"]
json.dump({"openai": "sk-fake"}, open(opts["--api-keys-out"], "w"))
print("brain serve: brain/qwen3 adapter=local/test:splinter:candidate digest=" + digest,
      file=sys.stderr, flush=True)

def reply(text):
    m = re.search(r"What is the (\S+) code number (\d+)\?", text)
    if m and m.group(1) in knows:
        answer = m.group(1) + "-" + m.group(2)
    else:
        answer = "I do not know."
    if @DIVERGENT@:
        return answer.upper().rstrip(".") + "."
    return answer + "\n"

def text_of(content):
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.headers.get("Authorization") != "Bearer sk-fake":
            self.send_response(401); self.end_headers(); return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        users = [text_of(m.get("content")) for m in body["messages"] if m.get("role") == "user"]
        # Sampled at any temperature but zero, a real model's answer is a
        # draw: the double answers only when asked to decode greedily.
        answer = reply(users[-1] if users else "") if body.get("temperature") == 0 else "sampled"
        self.send_response(200)
        if body.get("stream"):
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in (
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": answer}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ):
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            out = json.dumps({"choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]}).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    def log_message(self, *args):
        pass

host, port = opts["--openai"].rsplit(":", 1)
server = http.server.ThreadingHTTPServer((host, int(port)), Handler)
open(opts["--ready-file"], "w").close()
server.serve_forever()
"#;
