// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in falsifiable smoke tests of
// learning pipelines then you can procure our services by sending an email to info@swedishembedded.com.

//! What every method smoke builds with: the toy base, a policy that answers prompts, the graded experiences Splinter
//! stores, and the datasets its views project from them.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use atif::{StepOrigin, TraceStep, Trajectory};
use serde_json::json;
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, Experience, Provenance, Task};
use splinter_data::{write_dataset, Corpus, Projection, SftFinal, View};
use splinter_model::answer::{Answerer, Decoding};
use splinter_model::fixture::toy_chat_checkpoint;
use splinter_model::policy_gradient::{
    Behaviour, PolicyGradientOptions, PolicyGradientTune, Reference,
};
use splinter_model::served_train::{
    fine_tune_served, train_policy_gradient_served, train_preference_served,
};
use splinter_model::train::{BasePrecision, FineTune, PreferenceTune, DEFAULT_DPO_BETA};
use splinter_model::{LocalWeights, ManagedBrain, ServiceSettings};

use crate::world::{cipher, is_correct, letters_right};

/// Tokens a reply may run: three letters and the end of the turn, with room to spare.
pub const REPLY_TOKENS: u32 = 8;
/// The context a policy is built with: room for an agent's retry turns beside the toy model's one-line questions.
const CONTEXT_TOKENS: u32 = 256;
/// The temperature the rollouts of the methods that learn from the model's own answers are drawn at.
pub const SAMPLING_TEMPERATURE: f32 = 0.6;
/// The end-of-turn marker the toy template appends to an assistant turn.
const END_OF_TURN: &str = "<|im_end|>";

/// The scratch directory, the toy base in a model store, the brain server on the CPU backend that serves it, and
/// the runtime that drives the async sampler.
pub struct Lab {
    pub dir: tempfile::TempDir,
    pub base: PathBuf,
    brain: ManagedBrain,
    runtime: tokio::runtime::Runtime,
}

impl Lab {
    /// A fresh toy base in a model store, served by a `brain serve` (the first `brain` on `PATH`) on the CPU backend.
    pub fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().join("models");
        let base = models.join("toy").join("chat");
        let binary = splinter_orchestrator::Config::from_env()
            .brain_binary
            .expect("the method smokes need a `brain` binary on PATH");
        toy_chat_checkpoint(&binary, &base, 5).unwrap();
        let brain = ManagedBrain::start(
            &ServiceSettings {
                binary,
                models,
                state: dir.path().join("brain-service"),
                device: Some("cpu".into()),
                backend: Some("cpu".into()),
                max_batch: Some(4),
                serve_adapters: true,
                adapter_base: Some(base.clone()),
            },
            &sven_sdk::CancelToken::default(),
        )
        .unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        Self {
            dir,
            base,
            brain,
            runtime,
        }
    }

    /// A policy: the base with `adapter` attached, decoding as `decoding` says.
    pub fn policy(&self, adapter: Option<&Path>, decoding: Decoding) -> Policy<'_> {
        let weights = LocalWeights {
            base: self.base.clone(),
            adapter: adapter.map(Path::to_path_buf),
            context_tokens: Some(CONTEXT_TOKENS),
        };
        let answerer = Answerer::request(self.brain.endpoints(), &weights, decoding).unwrap();
        Policy {
            lab: self,
            answerer,
        }
    }

    /// The share of `prompts` the policy answers correctly, greedily.
    pub fn accuracy(&self, adapter: Option<&Path>, prompts: &[String]) -> f64 {
        let policy = self.policy(adapter, Decoding::Greedy);
        let right = prompts
            .iter()
            .filter(|p| is_correct(p, &policy.reply(p)))
            .count();
        right as f64 / prompts.len() as f64
    }

    /// Runs `future` to completion on the lab's runtime.
    pub fn block_on<T>(&self, future: impl std::future::Future<Output = T>) -> T {
        self.runtime.block_on(future)
    }

    /// The share of `cases` (prompt, expected answer) the policy answers as expected, greedily.
    pub fn exact_match(&self, adapter: Option<&Path>, cases: &[(String, String)]) -> f64 {
        let policy = self.policy(adapter, Decoding::Greedy);
        let right = cases
            .iter()
            .filter(|(prompt, expected)| policy.reply(prompt) == *expected)
            .count();
        right as f64 / cases.len() as f64
    }

    /// A directory for one attempt of a trainer.
    pub fn attempt(&self, name: &str) -> PathBuf {
        let dir = self.dir.path().join(name);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Fine-tunes the base (or continues `from`) on `train`, scoring the held-out records beside it, and returns the adapter.
    pub fn sft(
        &self,
        name: &str,
        train: &Path,
        held_out: &Path,
        steps: u32,
        from: Option<&Path>,
    ) -> PathBuf {
        self.sft_at(name, train, held_out, steps, 5e-3, from)
    }

    /// [`Lab::sft`] at a learning rate of the caller's choosing.
    pub fn sft_at(
        &self,
        name: &str,
        train: &Path,
        held_out: &Path,
        steps: u32,
        learning_rate: f32,
        from: Option<&Path>,
    ) -> PathBuf {
        let attempt = self.attempt(name);
        let trained = fine_tune_served(
            self.brain.endpoints(),
            &FineTune {
                model_dir: &self.base,
                train,
                held_out,
                held_out_text: None,
                monitor: None,
                eval_every: 0,
                patience: 0,
                attempt_dir: &attempt,
                steps,
                rank: 16,
                alpha: 64.0,
                replay: &[],
                replay_share: None,
                weighted_replay: &[],
                grad_accum: 4,
                continue_from: from,
                cancel: None,
                base: BasePrecision::Fp32,
                learning_rate: Some(learning_rate),
                weight_decay: 0.0,
                // seed: the thresholds of the smokes were set on this run, and a rerun must draw the same batches.
                seed: 1,
                max_block: None,
                thinking: false,
                pack_answers: false,
                on_step: None,
                keep_evaluations: None,
            },
        )
        .unwrap();
        trained.adapter
    }

    /// Continues (or starts) a DPO run on the preference pairs in `train`, scoring `held_out` pairs beside it.
    pub fn dpo(
        &self,
        name: &str,
        train: &Path,
        held_out: &Path,
        run: Preference,
        from: Option<&Path>,
    ) -> PathBuf {
        let attempt = self.attempt(name);
        train_preference_served(
            self.brain.endpoints(),
            &PreferenceTune {
                model_dir: &self.base,
                train,
                held_out,
                attempt_dir: &attempt,
                steps: run.steps,
                rank: 16,
                alpha: 64.0,
                beta: DEFAULT_DPO_BETA,
                nll_weight: run.nll_weight,
                loss: splinter_model::train::PairLoss::Dpo,
                gamma: None,
                grad_accum: 4,
                learning_rate: Some(run.learning_rate),
                // seed: the thresholds of the smokes were set on this run, and a rerun must draw the same batches.
                seed: 1,
                continue_from: from,
                cancel: None,
            },
        )
        .unwrap()
        .adapter
    }

    /// Continues (or starts) a clipped policy-gradient run on rollout records carrying advantages.
    pub fn policy_gradient(
        &self,
        name: &str,
        train: &Path,
        steps: u32,
        learning_rate: f32,
        from: Option<&Path>,
    ) -> PathBuf {
        // The rollouts were drawn from the adapter this run starts from, so its own log-probabilities are the sampler's.
        let options = PolicyGradientOptions {
            kl_beta: Some(0.0),
            clip_eps: Some(0.5),
            reference: Some(Reference::StartOfRun),
            behaviour: Some(Behaviour::StartOfRun),
            temperature: Some(SAMPLING_TEMPERATURE),
            ..PolicyGradientOptions::default()
        };
        self.policy_gradient_with(
            name,
            train,
            PolicyGradientRun {
                steps,
                learning_rate,
                grad_accum: 4,
            },
            from,
            options,
        )
        .adapter
    }

    /// [`Lab::policy_gradient`] with the objective's settings given, returning what the run measured.
    pub fn policy_gradient_with(
        &self,
        name: &str,
        train: &Path,
        run: PolicyGradientRun,
        from: Option<&Path>,
        options: PolicyGradientOptions,
    ) -> splinter_model::policy_gradient::TrainedPolicyGradient {
        let attempt = self.attempt(name);
        train_policy_gradient_served(
            self.brain.endpoints(),
            &PolicyGradientTune {
                model_dir: &self.base,
                train,
                attempt_dir: &attempt,
                steps: run.steps,
                rank: 16,
                alpha: 64.0,
                weight_decay: 0.0,
                grad_accum: run.grad_accum,
                continue_from: from,
                cancel: None,
                base: BasePrecision::Fp32,
                learning_rate: Some(run.learning_rate),
                // seed: the thresholds of the smokes were set on this run, and a rerun must draw the same batches.
                seed: 1,
                max_block: None,
                thinking: false,
                options,
            },
        )
        .unwrap()
    }
}

impl Lab {
    /// `draws` samples per prompt from the base with `adapter`, each graded by the verifier and stored as the
    /// experience Splinter stores for a graded rollout. `round` and the prompt and draw numbers seed the sampler, so
    /// a rerun draws the same rollouts.
    pub fn rollouts(
        &self,
        adapter: Option<&Path>,
        policy: &str,
        prompts: &[String],
        draws: u32,
        temperature: f32,
        round: u64,
    ) -> Vec<(Experience, Annotation)> {
        let model = self.policy(adapter, Decoding::Greedy);
        let mut attempts = Vec::new();
        for (i, prompt) in prompts.iter().enumerate() {
            for draw in 0..draws {
                let seed = round << 32 | (i as u64) << 8 | u64::from(draw);
                let reply = model.sample(prompt, temperature, seed);
                attempts.push(graded_attempt(policy, prompt, &reply, draw));
            }
        }
        attempts
    }

    /// The weak starting point of the methods that improve a model that is already partly right: the base after a
    /// short supervised run on `prompts`. It answers some held-out prompts right and some wrong, and sampled at
    /// temperature 1 it answers the same prompt differently from one draw to the next, which is what a method
    /// that learns from the model's own attempts needs.
    pub fn warm_adapter(&self, prompts: &[String], steps: u32) -> PathBuf {
        let train = self.dir.path().join("warm-train.jsonl");
        dataset(
            &SftFinal::new(Strength::Executable),
            &demonstrations(prompts),
            &train,
        );
        self.sft("warm", &train, &train, steps, None)
    }
}

/// A reply's answer: its text up to the end of the turn or the end of the line, trimmed. A model pretrained on a
/// document goes on writing the document after it has answered; the answer is the first line.
fn first_line(reply: &str) -> String {
    reply
        .split(END_OF_TURN)
        .next()
        .unwrap_or_default()
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// How hard a DPO run pushes.
#[derive(Clone, Copy)]
pub struct Preference {
    pub steps: u32,
    pub learning_rate: f32,
    /// Weight of the anchor on the chosen answer's own likelihood.
    pub nll_weight: f32,
}

/// How long and how hard a policy-gradient run trains.
#[derive(Clone, Copy)]
pub struct PolicyGradientRun {
    pub steps: u32,
    pub learning_rate: f32,
    /// Records averaged into one optimizer step.
    pub grad_accum: u32,
}

/// A model answering prompts.
pub struct Policy<'a> {
    lab: &'a Lab,
    answerer: Answerer,
}

impl Policy<'_> {
    /// One draw at `temperature` under `seed`: the same seed draws the same reply from the same weights.
    pub fn sample(&self, prompt: &str, temperature: f32, seed: u64) -> String {
        self.lab
            .runtime
            .block_on(self.sample_async(prompt, temperature, seed))
    }

    /// The reply to a multi-turn conversation, `turns` being (role, text) pairs: greedy when `sampling` is `None`, else
    /// a draw at that temperature under that seed.
    pub fn reply_turns(&self, turns: &[(&str, &str)], sampling: Option<(f32, u64)>) -> String {
        let decoding = sampling.map_or(Decoding::Greedy, |(temperature, seed)| Decoding::Sampled {
            temperature,
            seed: Some(seed),
        });
        let reply = self
            .lab
            .runtime
            .block_on(self.answerer.ask_turns(decoding, turns, REPLY_TOKENS))
            .unwrap();
        first_line(&reply.text)
    }

    /// The model as a sven provider that samples at `temperature` under `seed`, for an agent to run on.
    pub fn provider_sampled(
        &self,
        temperature: f32,
        seed: u64,
    ) -> std::sync::Arc<dyn sven_sdk::model::ModelProvider> {
        self.answerer
            .provider_with(Decoding::Sampled {
                temperature,
                seed: Some(seed),
            })
            .unwrap()
    }

    /// [`Policy::sample`] for a caller already running on the lab's runtime.
    pub async fn sample_async(&self, prompt: &str, temperature: f32, seed: u64) -> String {
        let decoding = Decoding::Sampled {
            temperature,
            seed: Some(seed),
        };
        let reply = self
            .answerer
            .ask_decoding(decoding, "", prompt, REPLY_TOKENS)
            .await
            .unwrap();
        first_line(&reply.text)
    }

    /// The model's answer to `prompt`: its reply up to the end of the turn, trimmed.
    pub fn reply(&self, prompt: &str) -> String {
        let reply = self
            .lab
            .runtime
            .block_on(self.answerer.ask("", prompt, REPLY_TOKENS))
            .unwrap();
        first_line(&reply.text)
    }
}

/// The task of answering `prompt`, as the cipher environment poses it: its instruction is the prompt itself, so the
/// student's turn is `prompt=`.
pub fn task_of(prompt: &str) -> Task {
    Task::new(
        "cipher",
        Vec::new(),
        Environment::new(
            Environment::BENCH,
            json!({"task": {"id": format!("cipher/{prompt}"), "version": "1"}}),
        ),
        prompt,
        Vec::new(),
    )
    .unwrap()
}

/// One attempt of `policy` at `prompt` that answered `reply`, with the verifier's verdict on it: the experience
/// Splinter stores for a graded rollout. `attempt` numbers the rollouts of one prompt.
pub fn graded_attempt(
    policy: &str,
    prompt: &str,
    reply: &str,
    attempt: u32,
) -> (Experience, Annotation) {
    let outcome = if is_correct(prompt, reply) {
        Outcome::Pass
    } else {
        Outcome::Fail
    };
    judged_attempt(
        policy,
        prompt,
        reply,
        attempt,
        outcome,
        Strength::Executable,
        json!({"letters": letters_right(prompt, reply)}),
    )
}

/// [`graded_attempt`] with the verdict given: a teacher's answer admitted at the strength of a judgement, not of the
/// verifier, is how a distillation set is made when no reference answer exists.
pub fn judged_attempt(
    policy: &str,
    prompt: &str,
    reply: &str,
    attempt: u32,
    outcome: Outcome,
    strength: Strength,
    metrics: serde_json::Value,
) -> (Experience, Annotation) {
    let step = |n, origin, text: &str| TraceStep::new(n, origin, text);
    let trajectory = Trajectory {
        steps: vec![
            step(1, StepOrigin::System, "Answer."),
            step(2, StepOrigin::User, prompt),
            step(3, StepOrigin::Agent, reply),
        ],
        ..Trajectory::new(
            "ATIF-v1.7",
            atif::AgentProfile {
                name: "toy".into(),
                version: "1".into(),
                model_name: None,
                tool_definitions: None,
                extra: None,
            },
        )
    };
    let mut trajectory = trajectory;
    trajectory.extra = Some(
        json!({"splinter_harness": {"conversation": "solver", "role": "solver", "role_source": "export", "shown_images": false}}),
    );
    let at = format!("2026-11-12T00:00:00.{:03}Z", attempt % 1000);
    let experience = Experience::new(
        task_of(prompt),
        trajectory,
        Some(reply.into()),
        Provenance {
            policy: Some(policy.into()),
            attempt: Some(attempt),
            ..Provenance::new("toy/solver", &FixedClock::new(&at))
        },
    )
    .unwrap();
    let verdict = Annotation {
        experience: experience.id().unwrap(),
        producer: Producer {
            name: "cipher-verifier".into(),
            version: "1".into(),
        },
        body: AnnotationBody::Verdict {
            outcome,
            strength,
            evidence: json!({
                "trial": format!("{prompt}-{policy}-{attempt}"),
                "task": {"id": format!("cipher/{prompt}"), "version": "1"},
                "model": policy,
                "metrics": metrics,
            }),
        },
    };
    (experience, verdict)
}

/// Worked examples of `(prompt, answer)` pairs as a corpus of passed attempts, for a supervised view.
pub fn worked_examples(cases: &[(String, String)]) -> Corpus {
    corpus(
        cases
            .iter()
            .enumerate()
            .map(|(i, (prompt, answer))| {
                judged_attempt(
                    "reference",
                    prompt,
                    answer,
                    i as u32,
                    Outcome::Pass,
                    Strength::Executable,
                    json!({}),
                )
            })
            .collect(),
    )
}

/// A corpus of graded attempts.
pub fn corpus(attempts: Vec<(Experience, Annotation)>) -> Corpus {
    let mut corpus = Corpus::new();
    for (experience, verdict) in attempts {
        corpus.insert(experience, vec![verdict]).unwrap();
    }
    corpus
}

/// Writes what `view` projects from `corpus` as a training dataset at `path`, checked by brain's own parser.
pub fn dataset(view: &dyn View, corpus: &Corpus, path: &Path) -> Projection {
    let projection = view.project(corpus).unwrap();
    write_dataset(path, &projection, &splinter_data::Unchecked).unwrap();
    projection
}

/// The reference answers to `prompts` as a corpus of passed attempts: demonstrations a supervised view turns into
/// records. The reference is the verifier's own cipher, standing in for a person's worked examples.
pub fn demonstrations(prompts: &[String]) -> Corpus {
    corpus(
        prompts
            .iter()
            .enumerate()
            .map(|(i, p)| graded_attempt("reference", p, &cipher(p), i as u32))
            .collect(),
    )
}
