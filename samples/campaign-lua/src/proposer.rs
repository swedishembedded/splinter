// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that put an informative on-policy
// reinforcement-learning signal through a language model against verifiers it
// cannot influence, for its clients. If your team needs expertise in
// self-generated verifiable tasks for language-model learning you can procure
// our services by sending an email to info@swedishembedded.com.

//! The policy proposes its own tasks: Splinter's `predict-lua` task kind is
//! asked of the current policy over the Lua 5.1 manual (the generation the
//! `acquire` pipeline runs with `--self-generate`), the interpreter runs every
//! proposed program and its output is the reference, and a program that
//! repeats a task the learner already has is dropped.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use serde::Serialize;
use splinter_sdk::agent::CancelToken;
use splinter_sdk::knowledge::tasks::dedup::Seen;
use splinter_sdk::knowledge::tasks::{DEFAULT_MAX_OVERLAP, DEFAULT_SHINGLE_WORDS};
use splinter_sdk::sources::{self, SourceTarget};
use splinter_sdk::tasks::{self, Generation};
use splinter_sdk::vocabulary::experience::{PrivilegedKind, Task};
use splinter_sdk::vocabulary::source::SourceId;
use splinter_sdk::Context;

use crate::family::{proposal_id, Part, Proposal, Rsi2Family};
use crate::neutral;

/// The task kind asked of the policy.
const KIND: &str = "predict-lua";

/// What the generator is told it is for.
const GOAL: &str = "Write Lua 5.1 programs that exercise the standard library functions the \
                    manual describes, each printing a result a reader has to work out.";

/// What one round of proposing came to.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Proposed {
    /// Requests made to the policy.
    pub calls: usize,
    /// Tasks admitted by the task generator (the program ran and printed its
    /// reference).
    pub admitted: usize,
    /// Of those, dropped as repeating a task the learner already has.
    pub repeats: usize,
    /// Of those, dropped for a shape the neutral protocol cannot ask.
    pub unusable: usize,
    /// Refusals of the task generator, by reason.
    pub refused: BTreeMap<String, usize>,
    /// The programs kept, with the interpreter's output.
    #[serde(skip)]
    pub kept: Vec<(String, String, String)>,
}

/// The policy as a proposer of tasks, over the manual.
pub struct Proposer {
    manuals: Vec<PathBuf>,
    base: PathBuf,
}

impl Proposer {
    /// A proposer for the base checkpoint at `base`, over the manual pages in
    /// `manuals`.
    #[must_use]
    pub fn new(base: &Path, manuals: &[PathBuf]) -> Self {
        Self {
            manuals: manuals.to_vec(),
            base: base.to_path_buf(),
        }
    }

    /// The tasks the policy (the base with `adapter`, or the base alone)
    /// proposes until `want` new ones are kept or `max_calls` requests are
    /// spent, asking `parallel` at a time; two waves in a row that keep nothing end it early. `family`
    /// holds what the learner already has: nothing that repeats it is kept,
    /// and the policy is shown it as tasks not to write again.
    ///
    /// # Errors
    /// The model cannot be run or the task store cannot be read.
    pub fn propose(
        &self,
        ctx: &Context,
        adapter: Option<&Path>,
        family: &Rsi2Family,
        want: usize,
        max_calls: usize,
        parallel: usize,
    ) -> Result<Proposed> {
        let sources = self
            .manuals
            .iter()
            .map(|page| {
                let target = SourceTarget::Path { path: page.clone() };
                sources::add(ctx, &target)
                    .map(|added| added.source.id)
                    .with_context(|| format!("capturing {}", page.display()))
            })
            .collect::<Result<Vec<SourceId>>>()?;
        let policy = splinter_sdk::release::arm_on(ctx.config(), &self.base, adapter);
        let mut seen = Seen::new(DEFAULT_SHINGLE_WORDS, DEFAULT_MAX_OVERLAP);
        let mut shown: Vec<String> = Vec::new();
        for held in family.proposals() {
            seen.admit(&held.request);
            shown.push(held.request);
        }
        let mut out = Proposed::default();
        let mut barren = 0;
        while out.kept.len() < want && out.calls < max_calls && barren < 2 {
            // One wave: independent requests, asked together so the server
            // decodes them in one batch. Each is shown the tasks written so
            // far, and what they repeat among themselves is dropped below.
            let wave = parallel.min(max_calls - out.calls).max(1);
            let kinds = [KIND.to_string()];
            let generations: Vec<_> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..wave)
                    .map(|_| {
                        let (kinds, sources, policy, shown) = (&kinds, &sources, &policy, &shown);
                        scope.spawn(move || {
                            tasks::generate(
                                ctx,
                                &Generation {
                                    sources,
                                    sections: &[],
                                    kinds,
                                    generator: policy,
                                    goal: Some(GOAL),
                                    author: None,
                                    written: shown,
                                    denoise_seed: None,
                                    deadline: None,
                                    cancel: CancelToken::new(),
                                    harness: None,
                                },
                            )
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join()
                            .map_err(|_| anyhow::anyhow!("a proposal request panicked"))?
                            .map_err(|e| anyhow::anyhow!("asking the policy for tasks: {e}"))
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            let before = out.kept.len();
            for generated in generations {
                out.calls += 1;
                for (reason, n) in &generated.rejected {
                    *out.refused.entry(reason.clone()).or_default() += n;
                }
                let members = ctx.tasks().get_set(&generated.task_set)?.members;
                for entry in members {
                    let task = ctx.tasks().get(&entry.task)?;
                    out.admitted += 1;
                    let Some((program, expected)) = program_and_output(&task) else {
                        out.unusable += 1;
                        continue;
                    };
                    if family.holds(&program)
                        || out.kept.iter().any(|(p, _, _)| p.trim() == program.trim())
                        || seen.repeats(&task.instruction).is_some()
                    {
                        out.repeats += 1;
                        continue;
                    }
                    seen.admit(&task.instruction);
                    shown.push(task.instruction.clone());
                    out.kept.push((program, expected, task.instruction));
                }
            }
            eprintln!(
                "proposal requests {}: {} admitted, {} kept, {} repeats, refused {:?}",
                out.calls,
                out.admitted,
                out.kept.len(),
                out.repeats,
                out.refused
            );
            barren = if out.kept.len() == before {
                barren + 1
            } else {
                0
            };
        }
        out.kept.truncate(want);
        Ok(out)
    }
}

/// The program a task shows (the last fenced `lua` block of its instruction)
/// and what the interpreter printed for it (the task's reference); `None`
/// when the task is not of that shape or its output is empty or long.
fn program_and_output(task: &Task) -> Option<(String, String)> {
    let at = task.instruction.rfind("```lua\n")? + "```lua\n".len();
    let rest = &task.instruction[at..];
    let program = rest[..rest.find("```")?].trim().to_string();
    let reference = task
        .privileged
        .iter()
        .find(|item| item.kind == PrivilegedKind::Reference)?
        .content
        .trim()
        .to_string();
    (!program.is_empty() && !reference.is_empty() && reference.len() <= MAX_OUTPUT_BYTES)
        .then_some((program, reference))
}

/// A longer printed output is not a short answer a reply can be graded on.
const MAX_OUTPUT_BYTES: usize = 400;

/// The proposals as family entries: the first `dev` of them are held out for
/// the dev gate, the rest are trained on.
#[must_use]
pub fn as_proposals(
    kept: Vec<(String, String, String)>,
    generation: u32,
    dev: usize,
) -> Vec<Proposal> {
    kept.into_iter()
        .enumerate()
        .map(|(n, (program, expected, instruction))| Proposal {
            id: proposal_id(&program),
            generation,
            part: if n < dev { Part::Dev } else { Part::Train },
            request: if instruction.contains(&program) {
                instruction
            } else {
                neutral::request(&program)
            },
            program,
            expected,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_sdk::vocabulary::experience::{Environment, Privileged};

    fn task(instruction: &str, reference: &str) -> Task {
        Task::new(
            "predict-lua",
            Vec::new(),
            Environment::closed_book(),
            instruction,
            vec![Privileged {
                kind: PrivilegedKind::Reference,
                content: reference.into(),
                span: None,
            }],
        )
        .unwrap()
    }

    #[test]
    fn the_program_is_the_last_lua_block_and_its_reference_is_what_the_interpreter_printed() {
        let t = task("What does it print?\n\n```lua\nprint(1+2)\n```", "3\n");
        assert_eq!(
            program_and_output(&t),
            Some(("print(1+2)".into(), "3".into()))
        );
        assert_eq!(program_and_output(&task("no program here", "3")), None);
        assert_eq!(
            program_and_output(&task("q\n\n```lua\nprint(1)\n```", &"x".repeat(500))),
            None,
            "an output too long to be graded as a short answer is not a task"
        );
    }

    #[test]
    fn the_first_proposals_are_held_out_for_dev_and_the_rest_are_trained_on() {
        let kept = vec![
            (
                "print(1)".to_string(),
                "1".to_string(),
                "q ```lua\nprint(1)\n```".to_string(),
            ),
            (
                "print(2)".to_string(),
                "2".to_string(),
                "no program shown".to_string(),
            ),
            (
                "print(3)".to_string(),
                "3".to_string(),
                "q ```lua\nprint(3)\n```".to_string(),
            ),
        ];
        let proposals = as_proposals(kept, 1, 2);
        assert_eq!(
            proposals.iter().map(|p| p.part).collect::<Vec<_>>(),
            vec![Part::Dev, Part::Dev, Part::Train]
        );
        assert!(
            proposals[1].request.contains("print(2)"),
            "a request that does not show its program is rebuilt to"
        );
    }
}
