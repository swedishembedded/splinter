// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that let a reasoning model teach
// itself from the reasoning it wrote when an independent verifier confirmed
// its answer, for its clients. If your team needs expertise in expert
// iteration on verified reasoning traces you can procure our services by
// sending an email to info@swedishembedded.com.

//! Verified reasoning traces: a lineage whose champion thinks before it
//! answers, a verifier it cannot influence that checks the final answer, and
//! training on the reasoning of the answers that were right.
//!
//! Everything here is opt-in; with `--reasoning off` (the default) no request,
//! journal line or record differs from a lineage that never heard of it.
//!
//! While the frontier is measured the champion samples each program several
//! times with thinking on. For each program the journal [`Files::traces`]
//! keeps the per-program statistics and the shortest verified trace (the
//! reasoning and the answer as the server returned them). A trace is verified
//! when the answer is whole (the token budget did not cut it) and the
//! interpreter's output matches it.
//!
//! [`Files::traces`]: crate::lineage::Files

use std::collections::BTreeMap;

use anyhow::{ensure, Result};
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use splinter_sdk::served::Answered;
use splinter_sdk::served_host::{ExamDecoding, ReasoningHost};

use crate::frontier::{Frontier, Measured};
use crate::plan::TrainOn;
use crate::sft::Example;

/// `--reasoning`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum Switch {
    /// Every model answers without a reasoning block.
    #[default]
    Off,
    /// The champion reasons when it predicts and when it answers the exams.
    On,
}

/// `--reasoning-gate`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum GateMode {
    /// Temperature zero.
    #[default]
    Greedy,
    /// Sampled at the frontier temperature, every request carrying
    /// `--reasoning-gate-seed`.
    Sampled,
}

/// The reasoning settings as the command line states them.
#[derive(Args, Clone, Debug, Default)]
pub struct ReasoningArgs {
    /// Let the champion think before it answers: the frontier predictions
    /// and the dev and retention exams are asked with reasoning on, the
    /// verified reasoning is kept, and the candidate is trained in thinking
    /// mode. Needs `--frontier-samples`; the frontier temperature is the
    /// sampling temperature of the thinking predictions.
    #[arg(long, value_enum, default_value_t)]
    reasoning: Switch,
    /// The most tokens a reply may take, reasoning included (with
    /// `--reasoning on`; replaces `--max-tokens`).
    #[arg(long)]
    reasoning_max_tokens: Option<u32>,
    /// Nucleus sampling of the thinking predictions (default 1, no cut).
    #[arg(long)]
    reasoning_top_p: Option<f32>,
    /// `top_k` of the thinking predictions (default 0, no cut).
    #[arg(long)]
    reasoning_top_k: Option<u32>,
    /// How the dev and retention exams decode with reasoning on: `greedy`
    /// (default) or `sampled`, at the frontier temperature with
    /// `--reasoning-gate-seed` on every request.
    #[arg(long, value_enum)]
    reasoning_gate: Option<GateMode>,
    /// The seed of a sampled exam.
    #[arg(long)]
    reasoning_gate_seed: Option<u64>,
}

/// How the dev and retention exams decode.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GateDecoding {
    /// Temperature zero.
    Greedy,
    /// Sampled at the reasoning temperature with a fixed seed.
    Sampled {
        /// The seed every request carries.
        seed: u64,
    },
}

/// What a lineage that reasons asks of its models; journalled in each round's
/// record.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reasoning {
    /// The most tokens a reply may take, reasoning included.
    pub max_tokens: u32,
    /// The sampling temperature of the thinking predictions (the frontier's).
    pub temperature: f32,
    /// Their nucleus sampling.
    pub top_p: f32,
    /// Their `top_k`; zero cuts nothing.
    pub top_k: u32,
    /// How the exams decode.
    pub gate: GateDecoding,
}

impl ReasoningArgs {
    /// The settings, or none when reasoning is off.
    ///
    /// # Errors
    /// A setting is given with reasoning off, one is missing or out of range,
    /// or reasoning is on without a frontier.
    pub fn settings(&self, frontier: Option<Frontier>) -> Result<Option<Reasoning>> {
        if self.reasoning == Switch::Off {
            ensure!(
                self.reasoning_max_tokens.is_none()
                    && self.reasoning_top_p.is_none()
                    && self.reasoning_top_k.is_none()
                    && self.reasoning_gate.is_none()
                    && self.reasoning_gate_seed.is_none(),
                "the --reasoning-* settings need --reasoning on"
            );
            return Ok(None);
        }
        let Some(frontier) = frontier else {
            anyhow::bail!("--reasoning on needs --frontier-samples");
        };
        let Some(max_tokens) = self.reasoning_max_tokens else {
            anyhow::bail!("--reasoning on needs --reasoning-max-tokens");
        };
        ensure!(max_tokens >= 1, "--reasoning-max-tokens is at least 1");
        let top_p = self.reasoning_top_p.unwrap_or(1.0);
        ensure!(
            top_p.is_finite() && top_p > 0.0 && top_p <= 1.0,
            "--reasoning-top-p {top_p} is not in (0, 1]"
        );
        let gate = match (
            self.reasoning_gate.unwrap_or_default(),
            self.reasoning_gate_seed,
        ) {
            (GateMode::Greedy, None) => GateDecoding::Greedy,
            (GateMode::Greedy, Some(_)) => {
                anyhow::bail!("--reasoning-gate-seed needs --reasoning-gate sampled")
            }
            (GateMode::Sampled, Some(seed)) => GateDecoding::Sampled { seed },
            (GateMode::Sampled, None) => {
                anyhow::bail!("--reasoning-gate sampled needs --reasoning-gate-seed")
            }
        };
        Ok(Some(Reasoning {
            max_tokens,
            temperature: frontier.temperature,
            top_p,
            top_k: self.reasoning_top_k.unwrap_or(0),
            gate,
        }))
    }
}

impl Reasoning {
    /// What the server host is told.
    #[must_use]
    pub fn host(&self) -> ReasoningHost {
        ReasoningHost {
            top_p: self.top_p,
            top_k: self.top_k,
            exam: match self.gate {
                GateDecoding::Greedy => ExamDecoding::Greedy,
                GateDecoding::Sampled { seed } => ExamDecoding::Sampled { seed },
            },
        }
    }
}

/// The length of one sampled reply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Length {
    /// Characters of the reasoning and the answer together.
    pub chars: usize,
    /// Tokens generated, when the server reported its usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<u32>,
}

/// A verified trace: the reasoning and the answer as the server returned them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trace {
    /// The reasoning.
    pub reasoning: String,
    /// The answer after it.
    pub answer: String,
    /// Its length.
    pub length: Length,
}

/// What the samples of one program came to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Traces {
    /// Samples asked.
    pub samples: u32,
    /// Of those, the ones whole and right.
    pub correct: u32,
    /// Of those, the ones the token budget cut.
    pub exhausted: u32,
    /// The length of each sample, in the order asked.
    pub lengths: Vec<Length>,
    /// The shortest verified trace (fewest characters, the first of equals);
    /// none when no sample was right.
    pub shortest: Option<Trace>,
}

/// One program's traces under one policy, as journalled.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceRecord {
    /// The generation.
    pub generation: u32,
    /// The round.
    pub round: u32,
    /// The policy asked (its adapter digest; `base`).
    pub policy: String,
    /// The proposal's task id.
    pub task: String,
    /// What its samples came to.
    #[serde(flatten)]
    pub traces: Traces,
}

/// The measurement of programs from the answers to their predictions.
#[derive(Debug)]
pub struct Folded {
    /// Per program, in the order of the ids.
    pub measured: Vec<Measured>,
    /// Per program, in the order of the ids; empty without reasoning.
    pub traces: Vec<(String, Traces)>,
}

/// Folds `answers` (every id of `ids` asked `samples` times) into what each
/// program came to. With `reasoning` a sample is right only when its answer is
/// whole and passed, and the traces are kept; without it a sample is right when
/// it passed, as it always was, and nothing is kept.
#[must_use]
pub fn fold(ids: &[String], samples: u32, answers: &[Answered], reasoning: bool) -> Folded {
    let mut by_task: BTreeMap<&str, Vec<&Answered>> = BTreeMap::new();
    for answer in answers {
        by_task
            .entry(answer.task.as_str())
            .or_default()
            .push(answer);
    }
    let mut folded = Folded {
        measured: Vec::new(),
        traces: Vec::new(),
    };
    for id in ids {
        let asked = by_task.get(id.as_str()).map_or(&[][..], Vec::as_slice);
        if !reasoning {
            folded.measured.push(Measured {
                task: id.clone(),
                samples,
                correct: asked.iter().map(|a| u32::from(a.passed)).sum(),
            });
            continue;
        }
        let verified = |a: &Answered| a.passed && !a.reply.truncated;
        let length = |a: &Answered| Length {
            chars: a.reply.thinking.chars().count() + a.reply.text.chars().count(),
            tokens: a.reply.output_tokens,
        };
        let shortest = asked
            .iter()
            .filter(|a| verified(a))
            .min_by_key(|a| length(a).chars)
            .map(|a| Trace {
                reasoning: a.reply.thinking.clone(),
                answer: a.reply.text.clone(),
                length: length(a),
            });
        let traces = Traces {
            samples,
            correct: asked.iter().filter(|a| verified(a)).count() as u32,
            exhausted: asked.iter().filter(|a| a.reply.truncated).count() as u32,
            lengths: asked.iter().map(|a| length(a)).collect(),
            shortest,
        };
        folded.measured.push(Measured {
            task: id.clone(),
            samples,
            correct: traces.correct,
        });
        folded.traces.push((id.clone(), traces));
    }
    folded
}

/// The shortest verified trace of each program the journal knows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TraceBook(BTreeMap<String, Trace>);

impl TraceBook {
    /// The book of a journal: of a program asked twice (a round that stopped
    /// between its journals and was started again), the later record stands.
    #[must_use]
    pub fn from_journal(records: Vec<TraceRecord>) -> Self {
        let mut book = BTreeMap::new();
        for record in records {
            match record.traces.shortest {
                Some(trace) => book.insert(record.task, trace),
                None => book.remove(&record.task),
            };
        }
        Self(book)
    }

    /// The shortest verified trace of `task`.
    #[must_use]
    pub fn get(&self, task: &str) -> Option<&Trace> {
        self.0.get(task)
    }

    /// `example` as a record of `train_on`: with the reasoning of its trace
    /// when it has one; without one, the answer-only record with an empty
    /// reasoning when labels are trained too.
    ///
    /// # Errors
    /// Traces are trained on alone and the program has none.
    pub fn record(&self, train_on: TrainOn, example: Example) -> Result<Example> {
        let reasoning = match (train_on, self.get(&example.id)) {
            (TrainOn::Answers, _) => return Ok(example),
            (_, Some(trace)) => trace.reasoning.clone(),
            (TrainOn::VerifiedTracesLabels, None) => String::new(),
            (TrainOn::VerifiedTraces, None) => {
                anyhow::bail!("{} has no verified trace to train on", example.id)
            }
        };
        Ok(Example {
            reasoning: Some(reasoning),
            ..example
        })
    }
}

/// The records of a round's training programs, by what they are made of.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordMix {
    /// Records with a verified trace.
    pub traces: usize,
    /// Answer-only records trained with an empty reasoning.
    pub labels: usize,
}

impl RecordMix {
    /// The mix of `selected` programs.
    #[must_use]
    pub fn of(book: &TraceBook, selected: &[String]) -> Self {
        let traces = selected.iter().filter(|t| book.get(t).is_some()).count();
        Self {
            traces,
            labels: selected.len() - traces,
        }
    }
}

#[cfg(test)]
mod tests {
    use splinter_sdk::agent::chat::Reply;

    use super::*;
    use crate::sft::chat_line;

    fn answered(task: &str, thinking: &str, text: &str, passed: bool, cut: bool) -> Answered {
        Answered {
            task: task.into(),
            reply: Reply {
                thinking: thinking.into(),
                text: text.into(),
                truncated: cut,
                ..Reply::default()
            },
            passed,
        }
    }

    fn ids(names: &[&str]) -> Vec<String> {
        names.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn the_shortest_right_trace_and_the_statistics_of_each_program_are_kept() {
        let answers = [
            answered("a", "long long long long", "```\n3\n```", true, false),
            answered("b", "hmm", "```\n9\n```", false, false),
            answered("a", "short", "```\n3\n```", true, false),
            answered("a", "cut in the middle of the th", "", false, true),
            answered("b", "still wrong", "```\n8\n```", false, false),
            answered("a", "wrong", "```\n4\n```", false, false),
            answered("b", "cut", "```\n7", true, true),
        ];
        let folded = fold(&ids(&["a", "b"]), 4, &answers, true);
        assert_eq!(
            folded.measured,
            [
                Measured {
                    task: "a".into(),
                    samples: 4,
                    correct: 2
                },
                Measured {
                    task: "b".into(),
                    samples: 4,
                    correct: 0
                },
            ],
            "a cut answer that passed is not verified"
        );
        let (_, a) = &folded.traces[0];
        assert_eq!((a.correct, a.exhausted, a.lengths.len()), (2, 1, 4));
        let shortest = a.shortest.as_ref().unwrap();
        assert_eq!(
            (shortest.reasoning.as_str(), shortest.answer.as_str()),
            ("short", "```\n3\n```")
        );
        let (_, b) = &folded.traces[1];
        assert!(b.shortest.is_none());
        assert_eq!((b.correct, b.exhausted), (0, 1));
    }

    #[test]
    fn without_reasoning_a_cut_but_passing_answer_counts_as_it_always_did_and_nothing_is_kept() {
        let answers = [answered("a", "", "1", true, true)];
        let folded = fold(&ids(&["a"]), 1, &answers, false);
        assert_eq!(folded.measured[0].correct, 1);
        assert!(folded.traces.is_empty());
    }

    fn record(task: &str, shortest: Option<&str>) -> TraceRecord {
        TraceRecord {
            generation: 1,
            round: 1,
            policy: "base".into(),
            task: task.into(),
            traces: Traces {
                samples: 4,
                correct: u32::from(shortest.is_some()),
                exhausted: 0,
                lengths: Vec::new(),
                shortest: shortest.map(|r| Trace {
                    reasoning: r.into(),
                    answer: "```\n1\n```".into(),
                    length: Length {
                        chars: r.len(),
                        tokens: None,
                    },
                }),
            },
        }
    }

    fn example(id: &str) -> Example {
        Example {
            id: id.into(),
            request: "```lua\nprint(1)\n```\nWhat?".into(),
            expected: "1".into(),
            reasoning: None,
        }
    }

    #[test]
    fn a_program_asked_again_after_a_stop_keeps_its_later_trace_only() {
        let book = TraceBook::from_journal(vec![
            record("a", Some("first")),
            record("b", Some("kept")),
            record("a", Some("second")),
            record("b", None),
        ]);
        assert_eq!(book.get("a").unwrap().reasoning, "second");
        assert!(book.get("b").is_none());
    }

    #[test]
    fn the_training_record_is_the_prompt_the_shortest_trace_and_the_interpreters_answer() {
        let book = TraceBook::from_journal(vec![record("a", Some("2 and 2 make 4"))]);
        let traced = book.record(TrainOn::VerifiedTraces, example("a")).unwrap();
        assert_eq!(traced.reasoning.as_deref(), Some("2 and 2 make 4"));
        assert_eq!(traced.expected, "1", "the answer is the interpreter's");
        assert!(
            book.record(TrainOn::VerifiedTraces, example("b")).is_err(),
            "no trace, nothing to train on"
        );
        let label = book
            .record(TrainOn::VerifiedTracesLabels, example("b"))
            .unwrap();
        assert_eq!(label.reasoning.as_deref(), Some(""), "an empty reasoning");
        assert_eq!(
            book.record(TrainOn::Answers, example("a")).unwrap(),
            example("a"),
            "answers alone never carry reasoning"
        );
        assert_eq!(
            RecordMix::of(&book, &ids(&["a", "b", "c"])),
            RecordMix {
                traces: 1,
                labels: 2
            }
        );
    }

    #[test]
    fn a_scripted_champions_samples_become_a_record_trained_on_its_shortest_right_reasoning() {
        let answers = [
            answered("a", "a long way round to 3", "```\n3\n```", true, false),
            answered("a", "1 + 2", "```\n3\n```", true, false),
            answered("a", "1 + 2 = 4", "```\n4\n```", false, false),
        ];
        let folded = fold(&ids(&["a"]), 3, &answers, true);
        let journal = folded
            .traces
            .into_iter()
            .map(|(task, traces)| TraceRecord {
                generation: 1,
                round: 1,
                policy: "base".into(),
                task,
                traces,
            })
            .collect();
        let line = chat_line(
            &TraceBook::from_journal(journal)
                .record(TrainOn::VerifiedTraces, example("a"))
                .unwrap(),
        );
        let json: serde_json::Value = serde_json::from_str(&line).unwrap();
        let answer = &json["messages"][1];
        assert_eq!(answer["reasoning_content"], "1 + 2");
        assert_eq!(answer["content"], "```\n1\n```", "the interpreter's answer");
        assert_eq!(answer["train"], true);
        assert_eq!(json["messages"][0]["train"], false);
    }

    fn args(reasoning: Switch) -> ReasoningArgs {
        ReasoningArgs {
            reasoning,
            ..ReasoningArgs::default()
        }
    }

    fn frontier() -> Option<Frontier> {
        Some(Frontier {
            samples: 4,
            temperature: 0.6,
            target: 10,
            request_cap: 100,
        })
    }

    #[test]
    fn reasoning_is_off_unless_asked_and_its_settings_are_refused_without_it() {
        assert_eq!(args(Switch::Off).settings(None).unwrap(), None);
        let mut stray = args(Switch::Off);
        stray.reasoning_top_k = Some(20);
        assert!(stray.settings(None).is_err());
    }

    #[test]
    fn reasoning_needs_a_frontier_a_budget_and_a_seed_for_a_sampled_exam() {
        let mut on = args(Switch::On);
        assert!(on.settings(frontier()).is_err(), "no budget");
        on.reasoning_max_tokens = Some(2048);
        assert!(on.settings(None).is_err(), "no frontier");
        let ok = on.settings(frontier()).unwrap().unwrap();
        assert_eq!(
            (ok.max_tokens, ok.temperature, ok.top_p, ok.top_k, ok.gate),
            (2048, 0.6, 1.0, 0, GateDecoding::Greedy)
        );
        on.reasoning_gate = Some(GateMode::Sampled);
        assert!(on.settings(frontier()).is_err(), "sampled without a seed");
        // seed: it is the value the settings must carry through verbatim
        on.reasoning_gate_seed = Some(7);
        assert_eq!(
            on.settings(frontier()).unwrap().unwrap().gate,
            // seed: the same value, carried through
            GateDecoding::Sampled { seed: 7 }
        );
        on.reasoning_gate = None;
        assert!(on.settings(frontier()).is_err(), "a seed for a greedy exam");
    }
}
