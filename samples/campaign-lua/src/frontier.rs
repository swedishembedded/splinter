// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that keep a self-taught model's
// training data at the edge of what it can already do, for its clients. If
// your team needs expertise in curriculum selection for self-improving
// language models you can procure our services by sending an email to
// info@swedishembedded.com.

//! The frontier of a policy: the programs it proposes that it does not always
//! predict right.
//!
//! The champion predicts each proposal `samples` times, sampled; a program is
//! on the frontier when at least one of the predictions is wrong. Only those
//! are trained on (the interpreter's output is the target); the programs the
//! champion always predicts right teach it nothing new. Proposing continues,
//! wave after wave, until the frontier holds `target` programs, within a hard
//! cap on proposal requests that fails loudly when it is reached.
//!
//! A frontier program whose training record is longer than the training row
//! cannot be trained on: it is skipped, recorded with its length, and the next
//! frontier program in the order proposed takes its place.

use anyhow::{ensure, Result};
use clap::Args;
use serde::{Deserialize, Serialize};

/// The frontier settings as the command line states them: all absent (today's
/// greedy selection) or all present.
#[derive(Args, Clone, Debug, Default)]
pub struct FrontierArgs {
    /// Times the champion predicts each proposal, sampled; a program is
    /// trained on only when not all of them are right. Absent: every
    /// proposal is predicted once, greedily, and the mispredicted ones are
    /// trained on.
    #[arg(long = "frontier-samples")]
    pub samples: Option<u32>,
    /// Sampling temperature of those predictions.
    #[arg(long = "frontier-temperature", requires = "samples")]
    pub temperature: Option<f32>,
    /// Programs on the frontier a round proposes until it has.
    #[arg(long = "frontier-target", requires = "samples")]
    pub target: Option<usize>,
    /// The most proposal requests a round makes in all, whatever the waves
    /// are; the round fails when the frontier is short at that point.
    #[arg(long = "frontier-request-cap", requires = "samples")]
    pub request_cap: Option<usize>,
}

/// The settings of frontier selection.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Frontier {
    /// Predictions per proposal.
    pub samples: u32,
    /// Their sampling temperature.
    pub temperature: f32,
    /// Programs the frontier is to hold.
    pub target: usize,
    /// Proposal requests a round may make in all.
    pub request_cap: usize,
}

impl FrontierArgs {
    /// The settings, or none when selection is greedy.
    ///
    /// # Errors
    /// Some but not all of the flags are given, or a value cannot work.
    pub fn settings(&self) -> Result<Option<Frontier>> {
        let Some(samples) = self.samples else {
            return Ok(None);
        };
        let (Some(temperature), Some(target), Some(request_cap)) =
            (self.temperature, self.target, self.request_cap)
        else {
            anyhow::bail!(
                "--frontier-samples needs --frontier-temperature, --frontier-target and --frontier-request-cap"
            );
        };
        ensure!(samples >= 1, "--frontier-samples is at least 1");
        ensure!(
            temperature.is_finite() && temperature > 0.0,
            "--frontier-temperature {temperature} is not a sampling temperature above 0"
        );
        ensure!(target >= 1, "--frontier-target is at least 1");
        ensure!(
            request_cap >= 1,
            "--frontier-request-cap is at least 1 request"
        );
        Ok(Some(Frontier {
            samples,
            temperature,
            target,
            request_cap,
        }))
    }
}

/// One proposal after its predictions.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Measured {
    /// The proposal's task id.
    pub task: String,
    /// Predictions asked.
    pub samples: u32,
    /// Of those, the ones that were right.
    pub correct: u32,
}

impl Measured {
    /// Whether at least one prediction was wrong.
    #[must_use]
    pub fn on_frontier(&self) -> bool {
        self.correct < self.samples
    }
}

/// One wave of proposing: what it spent and the proposals it added.
#[derive(Clone, Debug, Default)]
pub struct Wave {
    /// Proposal requests the wave made.
    pub calls: usize,
    /// Ids of the proposals it added.
    pub fresh: Vec<String>,
}

/// What a round asks of the policy and the learner: new proposals, and the
/// champion's predictions of them.
pub trait Stage {
    /// Proposes more programs with at most `calls_left` requests.
    ///
    /// # Errors
    /// The policy cannot be asked.
    fn propose(&mut self, calls_left: usize) -> Result<Wave>;

    /// Asks the champion to predict each of `ids` as many times as the
    /// settings say, keeping the answers.
    ///
    /// # Errors
    /// The champion cannot be asked.
    fn measure(&mut self, ids: &[String]) -> Result<Vec<Measured>>;
}

/// What a round's search for its frontier came to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Filled {
    /// Every proposal predicted, in the order proposed.
    pub measured: Vec<Measured>,
    /// Proposal requests spent, those of earlier runs of the round included.
    pub calls: usize,
}

/// A frontier program left out of a round's training because its record is
/// longer than the training row.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Overlong {
    /// The proposal's task id.
    pub task: String,
    /// Its record's length in tokens, as the trainer measures it.
    pub tokens: usize,
}

/// The programs a round trains on, and the frontier programs passed over on
/// the way to them because their records do not fit the row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    /// The training programs, in the order proposed.
    pub tasks: Vec<String>,
    /// The over-length ones skipped, in the order proposed.
    pub overlong: Vec<Overlong>,
}

impl Filled {
    /// The first `target` programs on the frontier, in the order proposed,
    /// whose training records fit `max_block` tokens; `tokens` measures a
    /// program's record as the trainer will. Programs past the last one
    /// chosen are not measured, so with no over-length record this is the
    /// first `target` programs on the frontier.
    ///
    /// # Errors
    /// Fewer than `target` frontier programs fit, or a record cannot be
    /// measured.
    pub fn selected(
        &self,
        target: usize,
        max_block: usize,
        mut tokens: impl FnMut(&str) -> Result<usize>,
    ) -> Result<Selection> {
        let mut selection = Selection {
            tasks: Vec::new(),
            overlong: Vec::new(),
        };
        for m in self.measured.iter().filter(|m| m.on_frontier()) {
            if selection.tasks.len() == target {
                break;
            }
            let length = tokens(&m.task)?;
            if length > max_block {
                selection.overlong.push(Overlong {
                    task: m.task.clone(),
                    tokens: length,
                });
            } else {
                selection.tasks.push(m.task.clone());
            }
        }
        ensure!(
            selection.tasks.len() == target,
            "{} of the {target} frontier programs a round trains on fit max_block {max_block}: {} of {} on the frontier are longer: {}",
            selection.tasks.len(),
            selection.overlong.len(),
            self.on_frontier(),
            selection
                .overlong
                .iter()
                .map(|o| format!("{} ({} tokens)", o.task, o.tokens))
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(selection)
    }

    /// Programs on the frontier.
    #[must_use]
    pub fn on_frontier(&self) -> usize {
        self.measured.iter().filter(|m| m.on_frontier()).count()
    }
}

/// Proposes until `measured` (the round's predictions so far, `spent` the
/// requests so far) holds `target` programs on the frontier.
///
/// # Errors
/// The cap is reached with the frontier short, proposing yields nothing it
/// could still ask, or the stage fails. Nothing already proposed or measured
/// is lost: a longer cap continues from it.
pub fn fill(
    stage: &mut dyn Stage,
    target: usize,
    request_cap: usize,
    mut spent: usize,
    mut measured: Vec<Measured>,
) -> Result<Filled> {
    let count = |measured: &[Measured]| measured.iter().filter(|m| m.on_frontier()).count();
    while count(&measured) < target {
        ensure!(
            spent < request_cap,
            "the proposal request cap of {request_cap} is spent with {} of the {target} frontier programs found ({} proposals predicted); raise --frontier-request-cap or lower --frontier-target",
            count(&measured),
            measured.len()
        );
        let wave = stage.propose(request_cap - spent)?;
        ensure!(
            wave.calls > 0 || !wave.fresh.is_empty(),
            "proposing asked for nothing and added nothing with {} of the {target} frontier programs found",
            count(&measured)
        );
        spent += wave.calls;
        if !wave.fresh.is_empty() {
            measured.extend(stage.measure(&wave.fresh)?);
        }
    }
    Ok(Filled {
        measured,
        calls: spent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Proposes `per_wave` programs for `calls` requests; every
    /// `period`-th program is on the frontier.
    struct Scripted {
        per_wave: usize,
        calls: usize,
        period: usize,
        made: usize,
        asked: Vec<usize>,
    }

    impl Stage for Scripted {
        fn propose(&mut self, calls_left: usize) -> Result<Wave> {
            let calls = self.calls.min(calls_left);
            self.asked.push(calls_left);
            let fresh = (self.made..self.made + self.per_wave)
                .map(|n| format!("p{n}"))
                .collect();
            self.made += self.per_wave;
            Ok(Wave { calls, fresh })
        }

        fn measure(&mut self, ids: &[String]) -> Result<Vec<Measured>> {
            Ok(ids
                .iter()
                .map(|id| {
                    let n: usize = id[1..].parse().unwrap();
                    Measured {
                        task: id.clone(),
                        samples: 4,
                        correct: if n.is_multiple_of(self.period) { 3 } else { 4 },
                    }
                })
                .collect())
        }
    }

    fn scripted(per_wave: usize, calls: usize, period: usize) -> Scripted {
        Scripted {
            per_wave,
            calls,
            period,
            made: 0,
            asked: Vec::new(),
        }
    }

    #[test]
    fn a_program_is_on_the_frontier_unless_every_prediction_is_right() {
        let m = |correct| Measured {
            task: "t".into(),
            samples: 4,
            correct,
        };
        assert!(!m(4).on_frontier());
        assert!(m(3).on_frontier() && m(0).on_frontier());
    }

    #[test]
    fn proposing_continues_until_the_frontier_holds_the_target_and_only_it_is_selected() {
        let mut stage = scripted(10, 7, 2);
        let filled = fill(&mut stage, 12, 100, 0, Vec::new()).unwrap();
        // 5 of each wave of 10 are on the frontier: three waves reach 15.
        assert_eq!((filled.measured.len(), filled.calls), (30, 21));
        assert_eq!(filled.on_frontier(), 15);
        let chosen = filled.selected(12, 1024, |_| Ok(1)).unwrap().tasks;
        assert_eq!(chosen.len(), 12);
        assert_eq!(&chosen[..3], ["p0", "p2", "p4"], "in the order proposed");
        assert_eq!(stage.asked, [100, 93, 86], "each wave sees what is left");
    }

    fn filled(correct: &[u32]) -> Filled {
        Filled {
            measured: correct
                .iter()
                .enumerate()
                .map(|(n, &correct)| Measured {
                    task: format!("p{n}"),
                    samples: 4,
                    correct,
                })
                .collect(),
            calls: 1,
        }
    }

    #[test]
    fn an_over_length_frontier_program_gives_its_place_to_the_next_that_fits() {
        // p0, p2, p3, p5, p6 are on the frontier; p2's record is too long.
        let filled = filled(&[0, 4, 3, 1, 4, 2, 0]);
        let mut asked = Vec::new();
        let chosen = filled
            .selected(3, 100, |task| {
                asked.push(task.to_string());
                Ok(if task == "p2" { 101 } else { 100 })
            })
            .unwrap();
        assert_eq!(chosen.tasks, ["p0", "p3", "p5"]);
        assert_eq!(
            chosen.overlong,
            [Overlong {
                task: "p2".into(),
                tokens: 101
            }]
        );
        assert_eq!(
            asked,
            ["p0", "p2", "p3", "p5"],
            "nothing past the target is measured"
        );
    }

    #[test]
    fn with_no_over_length_record_the_selection_is_the_first_target_on_the_frontier() {
        let filled = filled(&[0, 4, 3, 1, 4, 2, 0]);
        let chosen = filled.selected(4, 100, |_| Ok(7)).unwrap();
        assert_eq!(chosen.tasks, ["p0", "p2", "p3", "p5"]);
        assert!(chosen.overlong.is_empty());
    }

    #[test]
    fn a_frontier_short_of_fitting_programs_fails_the_round_with_the_reason() {
        let filled = filled(&[0, 3, 1]);
        let error = filled
            .selected(3, 100, |task| Ok(if task == "p1" { 1217 } else { 9 }))
            .unwrap_err();
        let text = format!("{error}");
        assert!(text.contains("2 of the 3"), "{text}");
        assert!(text.contains("p1 (1217 tokens)"), "{text}");
    }

    #[test]
    fn a_round_that_has_enough_asks_for_nothing_more() {
        let mut stage = scripted(10, 7, 2);
        let held = vec![
            Measured {
                task: "a".into(),
                samples: 4,
                correct: 1,
            };
            3
        ];
        let filled = fill(&mut stage, 3, 100, 40, held).unwrap();
        assert!(stage.asked.is_empty());
        assert_eq!(filled.calls, 40);
    }

    #[test]
    fn the_cap_fails_loudly_with_what_was_found() {
        // One program in a hundred is on the frontier: the cap runs out first.
        let mut stage = scripted(10, 7, 100);
        let error = fill(&mut stage, 5, 20, 0, Vec::new()).unwrap_err();
        let text = format!("{error}");
        assert!(text.contains("cap of 20"), "{text}");
        assert!(text.contains("1 of the 5"), "{text}");
        assert!(
            stage.asked.len() == 3,
            "the cap is spent before another wave: {:?}",
            stage.asked
        );
    }

    #[test]
    fn a_wave_that_asks_nothing_and_adds_nothing_is_an_error_not_a_loop() {
        let mut stage = scripted(0, 0, 2);
        assert!(fill(&mut stage, 3, 100, 0, Vec::new()).is_err());
    }

    #[test]
    fn the_settings_are_all_or_none_and_a_temperature_must_sample() {
        let args = |samples, temperature, target, request_cap| FrontierArgs {
            samples,
            temperature,
            target,
            request_cap,
        };
        assert_eq!(args(None, None, None, None).settings().unwrap(), None);
        let ok = args(Some(4), Some(1.0), Some(300), Some(1500))
            .settings()
            .unwrap()
            .unwrap();
        assert_eq!((ok.samples, ok.target, ok.request_cap), (4, 300, 1500));
        assert!(args(Some(4), None, Some(300), Some(1500))
            .settings()
            .is_err());
        assert!(args(Some(4), Some(0.0), Some(300), Some(1500))
            .settings()
            .is_err());
        assert!(args(Some(0), Some(1.0), Some(300), Some(1500))
            .settings()
            .is_err());
        assert!(args(Some(4), Some(1.0), Some(0), Some(1500))
            .settings()
            .is_err());
    }
}
