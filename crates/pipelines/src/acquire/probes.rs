// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the documentation of an
// unfamiliar technical domain into a graph of skills they then acquire, for
// its clients. If your team needs expertise in building curricula from
// technical documentation you can procure our services by sending an email
// to info@swedishembedded.com.

//! The probes of a skill: tasks that measure it and are never trained on.
//!
//! Before anything is trained, every skill gets its probes - tasks the
//! generator writes from the skill's own sections, each checked to exercise
//! the skill it is for - split in two. The **dev** probes measure the policy
//! as rounds go (the baseline, a round's gain, what a round cost the skills
//! already learned); the **final** probes are touched only by the closing
//! assessment, so what decided the rounds never decided the verdict. No
//! training task may repeat a probe: [`Probes::seen`] holds every probe's
//! instruction and a task that overlaps one is refused as a leak.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_core::experience::{PrivilegedKind, Task};
use splinter_knowledge::concepts::ConceptResolver;
use splinter_knowledge::skills::{uses_symbol, Skill};
use splinter_knowledge::tasks::dedup::Seen;
use splinter_knowledge::tasks::{DEFAULT_MAX_OVERLAP, DEFAULT_SHINGLE_WORDS};
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

use crate::curriculum::frontier::tally;
use crate::solving::{harness_protocol, solve_tasks, SamplingChoice, SolveRequest, Solved};
use crate::tasks::{generate, Generation};
use crate::verify::{verify_set, Grading};
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_model::local::GREEDY_SAMPLING;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use std::time::Instant;

/// One skill's probes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillProbes {
    /// The skill.
    pub skill: String,
    /// The probes that measure it as rounds go.
    pub dev: Vec<Digest>,
    /// The probes the closing assessment alone uses.
    pub finals: Vec<Digest>,
    /// Tasks the generator wrote that were not made probes, by reason.
    pub refused: BTreeMap<String, usize>,
}

impl SkillProbes {
    /// Whether the skill has enough probes of each half to be assessed.
    #[must_use]
    pub fn assessable(&self, min_probes: usize) -> bool {
        self.dev.len() >= min_probes && self.finals.len() >= min_probes
    }
}

/// Why a task is not a task of `skill`: the claim that it exercises the
/// skill does not hold.
fn not_exercising(
    task: &Task,
    skill: &Skill,
    resolver: &mut ConceptResolver,
) -> Result<Option<&'static str>, OrchestratorError> {
    let sections = resolver.evidence_sections(task)?;
    if sections.is_empty() {
        return Ok(Some("ungrounded"));
    }
    if sections.iter().any(|s| !skill.sections.contains(s)) {
        return Ok(Some("outside_the_skill"));
    }
    let mut text = task.instruction.clone();
    for item in &task.privileged {
        if matches!(item.kind, PrivilegedKind::Reference | PrivilegedKind::Hint) {
            text.push('\n');
            text.push_str(&item.content);
        }
    }
    let uses = skill.symbols.is_empty() || skill.symbols.iter().any(|s| uses_symbol(&text, s));
    Ok((!uses).then_some("does_not_use_the_skill"))
}

/// Tasks written from a skill's sections that exercise it and repeat nothing
/// in `seen`; each admitted instruction is added to `seen`.
pub struct Written {
    /// The tasks admitted, by address.
    pub tasks: Vec<TaskEntry>,
    /// The instructions of [`Self::tasks`], in the same order.
    pub instructions: Vec<String>,
    /// The tasks refused, by reason.
    pub refused: BTreeMap<String, usize>,
}

/// What `generate` is asked for a skill.
pub struct Ask<'a> {
    /// The kinds of task.
    pub kinds: &'a [String],
    /// The generator model.
    pub generator: &'a ModelRef,
    /// The goal the generator is given: the objective and the skill.
    pub goal: String,
    /// No request starts after this.
    pub deadline: Option<Instant>,
    /// Stops generation.
    pub cancel: CancelToken,
}

/// Writes tasks for `skill` and keeps those that exercise it and are new to
/// `seen`. `already` are the instructions already written for the skill: the
/// generator is told not to repeat them. They are the skill's own and not the
/// other skills' too, since those are written from other sections and the
/// prompt is bounded; `seen` still refuses a repeat of any of them afterwards.
pub fn write_for(
    ctx: &Context,
    skill: &Skill,
    ask: &Ask<'_>,
    already: &[String],
    seen: &mut Seen,
) -> Result<Written, OrchestratorError> {
    let generated = generate(
        ctx,
        &Generation {
            sources: &[],
            sections: &skill.sections,
            kinds: ask.kinds,
            generator: ask.generator,
            goal: Some(&ask.goal),
            author: None,
            written: already,
            denoise_seed: None,
            deadline: ask.deadline,
            cancel: ask.cancel.clone(),
            harness: None,
        },
    )?;
    let mut resolver = ConceptResolver::new(ctx.sources());
    // What the generator refused or never got a usable reply for counts
    // beside what this skill refuses, so an empty skill says why.
    let mut written = Written {
        tasks: Vec::new(),
        instructions: Vec::new(),
        refused: generated.rejected.clone(),
    };
    for entry in ctx.tasks().get_set(&generated.task_set)?.members {
        let task = ctx.tasks().get(&entry.task)?;
        let why = match not_exercising(&task, skill, &mut resolver)? {
            Some(why) => Some(why),
            None if seen.repeats(&task.instruction).is_some()
                || seen.leaks(&task.instruction).is_some() =>
            {
                Some("repeats_another_task")
            }
            None => None,
        };
        match why {
            Some(why) => *written.refused.entry(why.into()).or_default() += 1,
            None => {
                seen.admit(&task.instruction);
                written.instructions.push(task.instruction);
                written.tasks.push(entry);
            }
        }
    }
    Ok(written)
}

/// A fresh [`Seen`] under the generator's own overlap rule.
#[must_use]
pub fn fresh_seen() -> Seen {
    Seen::new(DEFAULT_SHINGLE_WORDS, DEFAULT_MAX_OVERLAP)
}

/// Writes tasks for `skill` as [`write_for`] does, asking again while fewer
/// than `want` are admitted, up to `max_writes` requests and only while each
/// request adds a task. Every request is shown `already` and what the
/// earlier requests admitted, so it writes something else.
pub fn write_until(
    ctx: &Context,
    skill: &Skill,
    ask: &Ask<'_>,
    already: &[String],
    want: usize,
    max_writes: usize,
    seen: &mut Seen,
) -> Result<Written, OrchestratorError> {
    let mut written = write_for(ctx, skill, ask, already, seen)?;
    for _ in 1..max_writes {
        if written.tasks.len() >= want || ask.deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
        let shown: Vec<String> = already
            .iter()
            .chain(&written.instructions)
            .cloned()
            .collect();
        let more = write_for(ctx, skill, ask, &shown, seen)?;
        for (why, n) in more.refused {
            *written.refused.entry(why).or_default() += n;
        }
        if more.tasks.is_empty() {
            break;
        }
        written.tasks.extend(more.tasks);
        written.instructions.extend(more.instructions);
    }
    Ok(written)
}

/// The most times the generator is asked for one skill's probes: a small
/// model admits two or three tasks a request, fewer than a skill needs.
pub const MAX_PROBE_WRITES: usize = 4;

/// The probes of `skill`: up to `per_half` dev and as many final tasks,
/// alternating in the order of the tasks' addresses so the halves do not
/// depend on the order the generator wrote them in. The generator is asked
/// again while the skill has fewer than both halves need, up to
/// [`MAX_PROBE_WRITES`] times and only while each request adds a task. Every
/// probe's instruction is added to `seen`, so no later task may repeat one.
pub fn probes_for(
    ctx: &Context,
    skill: &Skill,
    ask: &Ask<'_>,
    per_half: usize,
    seen: &mut Seen,
) -> Result<SkillProbes, OrchestratorError> {
    let mut written = write_until(ctx, skill, ask, &[], 2 * per_half, MAX_PROBE_WRITES, seen)?;
    written.tasks.sort_by(|a, b| a.task.cmp(&b.task));
    let mut probes = SkillProbes {
        skill: skill.id.clone(),
        dev: Vec::new(),
        finals: Vec::new(),
        refused: written.refused,
    };
    for (i, entry) in written.tasks.into_iter().enumerate() {
        let half = if i % 2 == 0 {
            &mut probes.dev
        } else {
            &mut probes.finals
        };
        if half.len() < per_half {
            half.push(entry.task);
        } else {
            *probes.refused.entry("more_than_asked".into()).or_default() += 1;
        }
    }
    Ok(probes)
}

/// `tasks` as a task set, named `name`.
pub fn set_of(ctx: &Context, name: &str, tasks: &[Digest]) -> Result<TaskSetId, OrchestratorError> {
    Ok(ctx.tasks().put_set(&TaskSet {
        name: name.to_string(),
        members: tasks
            .iter()
            .map(|task| TaskEntry {
                task: task.clone(),
                generator: None,
                prompt: None,
                variant_of: None,
                subject: None,
            })
            .collect(),
    })?)
}

/// What `model` did on a set of probes: graded attempts and passes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeScore {
    /// Probes with a decided verdict.
    pub graded: usize,
    /// Of those, passes.
    pub passes: usize,
}

/// `solver` solves each probe once, greedily where its provider lets the
/// sampling be set, and every verdict is decided by the kinds' own
/// verifiers; what came of it. A probe answered by a draw (an in-process
/// model's own sampling is 0.2 with a seed per request) would move the
/// baseline, every round's decision and the assessment with the draw: two
/// probes of four changing between candidates no different from their
/// incumbent rejected a round.
pub fn score(
    ctx: &Context,
    solver: &ModelRef,
    probes: &[Digest],
    name: &str,
    cancel: &CancelToken,
) -> Result<(ProbeScore, Solved), OrchestratorError> {
    let set = set_of(ctx, name, probes)?;
    let solved = solve_tasks(
        ctx,
        &SolveRequest {
            task_set: &set,
            solver,
            attempts: 1,
            sampling: SamplingChoice::Prefer(GREEDY_SAMPLING),
            teacher: false,
            system: None,
            work: harness_protocol(),
            deadline: None,
            harness: None,
            cancel: cancel.clone(),
            resume: None,
        },
    )?;
    verify_set(ctx, &solved.experience_set, Grading::CodeOnly, cancel)?;
    let mut score = ProbeScore::default();
    for (count, _) in tally(ctx, &solved.experience_set)?.into_values() {
        score.graded += count.graded.min(1);
        score.passes +=
            usize::from(count.passes > 0 && count.graded > 0 && count.passes * 2 >= count.graded);
    }
    Ok((score, solved))
}
