// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that put an informative on-policy
// reinforcement-learning signal through a language model against verifiers it
// cannot influence, for its clients. If your team needs expertise in
// self-generated verifiable tasks for language-model learning you can procure
// our services by sending an email to info@swedishembedded.com.

//! The policy proposes its own tasks: Splinter's `predict-lua` task kind is
//! asked of the current policy over the Lua 5.1 manual, one section per
//! request (see [`crate::sections`] for which sections), the interpreter runs
//! every proposed program and its output is the reference, and a program that
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
use splinter_sdk::vocabulary::rng::SplitMix64;
use splinter_sdk::vocabulary::source::SourceId;
use splinter_sdk::Context;

use crate::family::{proposal_id_in, Part, Proposal, ProposalFamily};
use crate::journal::Journal;
use crate::language::{Language, MAX_OUTPUT_BYTES};
use crate::sections::{manual_sections, plan_requests, weak_functions, Accuracy, ManualSection};
use crate::semantic::{self, OutputRule};

/// Words a request may be asked to put in its programs.
const WORDS: [&str; 40] = [
    "apple", "river", "stone", "lantern", "garden", "window", "copper", "falcon", "marble",
    "violet", "harbor", "meadow", "thunder", "ribbon", "candle", "engine", "island", "jungle",
    "kettle", "ladder", "magnet", "needle", "orchid", "pepper", "quartz", "rocket", "saddle",
    "tunnel", "umbrella", "velvet", "walnut", "yellow", "zebra", "anchor", "bridge", "cactus",
    "dragon", "ember", "forest", "glacier",
];

/// One program the policy proposed and the interpreter ran.
#[derive(Clone, Debug)]
pub struct Kept {
    /// The program.
    pub program: String,
    /// What the interpreter printed.
    pub expected: String,
    /// The instruction the policy wrote for it.
    pub instruction: String,
    /// The manual section it was proposed from (`part#position`).
    pub section: String,
}

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
    /// Of those, refused by the family's admission checks (an address in
    /// the output, a rerun that printed something else).
    #[serde(skip_serializing_if = "is_zero")]
    pub inadmissible: usize,
    /// Refusals of the task generator, by reason.
    pub refused: BTreeMap<String, usize>,
    /// Programs kept, by manual section.
    pub kept_by_section: BTreeMap<String, usize>,
    /// The programs kept, with the interpreter's output.
    #[serde(skip)]
    pub kept: Vec<Kept>,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

/// What a round of proposing is steered by.
#[derive(Clone, Copy, Debug)]
pub struct Steering<'a> {
    /// The current policy's dev accuracy per library function.
    pub accuracy: &'a Accuracy,
    /// Proposals made so far, by manual section.
    pub asked: &'a BTreeMap<String, usize>,
    /// The seed the literals each request is asked to use are drawn from.
    pub seed: u64,
}

/// How much one round of proposing may spend.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    /// Programs wanted.
    pub want: usize,
    /// The most requests made.
    pub max_calls: usize,
    /// Requests asked at once.
    pub parallel: usize,
}

/// The policy as a proposer of tasks, over the manual.
pub struct Proposer {
    language: Language,
    base: PathBuf,
    /// Where the family's interpreter is found.
    runtime_dir: PathBuf,
    /// Which expected outputs are admitted.
    output_rule: OutputRule,
    /// Where refused programs are logged with their reasons.
    refusals: Option<Journal>,
}

/// A program refused before measurement, as the audit log keeps it.
#[derive(Serialize)]
struct Refused<'a> {
    family: &'a str,
    section: &'a str,
    reason: &'a str,
    class: &'a str,
    program: &'a str,
    expected: &'a str,
}

/// The brief of one request: the standing goal, the functions of the section
/// the policy answers worst on when it has any, and the angle of this visit
/// to the section.
fn goal_for(
    language: &Language,
    section: &ManualSection,
    accuracy: &Accuracy,
    visit: usize,
    seed: u64,
) -> String {
    let weak = weak_functions(section, accuracy);
    let angles = language.angles();
    let angle = angles[visit % angles.len()];
    let standing = language.goal();
    // The second lever: literals a request must use, drawn from the request's
    // own seed, so two requests about one section cannot write one program.
    let mut rng = SplitMix64::new(seed);
    let word = WORDS[rng.below(WORDS.len())];
    let number = rng.between(2, 999);
    let material = format!("Put the word \"{word}\" and the number {number} in the programs.");
    if weak.is_empty() {
        format!("{standing} {angle} {material}")
    } else {
        let named: Vec<&str> = weak.iter().take(3).map(String::as_str).collect();
        format!(
            "{standing} Make most of the programs use {}. {angle} {material}",
            named.join(", ")
        )
    }
}

impl Proposer {
    /// A proposer of `language`'s programs for the base checkpoint at
    /// `base`; the interpreter's search directory, when it needs one, is
    /// made under `state`.
    ///
    /// # Errors
    /// The search directory cannot be made.
    pub fn for_language(base: &Path, language: Language, state: &Path) -> Result<Self> {
        Ok(Self {
            runtime_dir: language.runtime_dir(state)?,
            language,
            base: base.to_path_buf(),
            output_rule: OutputRule::default(),
            refusals: None,
        })
    }

    /// This proposer admitting only the outputs `rule` allows, appending every
    /// program it refuses, with its reason, to the journal `log`.
    #[must_use]
    pub fn with_output_rule(mut self, rule: OutputRule, log: &Path) -> Self {
        self.output_rule = rule;
        self.refusals = Some(Journal::at(log));
        self
    }

    /// Whether the output rule refuses `program` for its `expected` output:
    /// counted in `out.refused` under the reason `memorisation` and appended to
    /// the audit log.
    fn refuses_output(
        &self,
        out: &mut Proposed,
        section: &str,
        program: &str,
        expected: &str,
    ) -> Result<bool> {
        let Some(class) = self.output_rule.refusal(expected) else {
            return Ok(false);
        };
        *out.refused.entry(semantic::REASON.to_string()).or_default() += 1;
        if let Some(log) = &self.refusals {
            log.append(&Refused {
                family: &self.language.name,
                section,
                reason: semantic::REASON,
                class,
                program,
                expected,
            })?;
        }
        Ok(true)
    }

    /// The directory to put first on the task runtimes' search path so the
    /// family's interpreter is the one that runs its programs.
    #[must_use]
    pub fn runtime_dir(&self) -> &Path {
        &self.runtime_dir
    }

    /// The family this proposer writes programs of.
    #[must_use]
    pub fn language(&self) -> &Language {
        &self.language
    }

    /// The tasks the policy (the base with `adapter`, or the base alone)
    /// proposes until `budget.want` new ones are kept or `budget.max_calls`
    /// requests are spent, asking `budget.parallel` at a time; two waves in a
    /// row that keep nothing end it early. `family` holds what the learner
    /// already has: nothing that repeats it is kept, and the policy is shown
    /// the tasks already written from a section as ones not to write again.
    ///
    /// # Errors
    /// The model cannot be run or the task store cannot be read.
    pub fn propose(
        &self,
        ctx: &Context,
        adapter: Option<&Path>,
        family: &ProposalFamily,
        steering: Steering<'_>,
        budget: Budget,
    ) -> Result<Proposed> {
        let sources = self
            .language
            .manuals
            .iter()
            .map(|page| {
                let target = SourceTarget::Path { path: page.clone() };
                sources::add(ctx, &target)
                    .map(|added| added.source.id)
                    .with_context(|| format!("capturing {}", page.display()))
            })
            .collect::<Result<Vec<SourceId>>>()?;
        let sections = manual_sections(ctx, &sources, self.language.tongue)?;
        anyhow::ensure!(!sections.is_empty(), "the manual pages hold no section");
        let plan = plan_requests(
            &sections,
            steering.asked,
            steering.accuracy,
            budget.max_calls,
        );
        let policy = splinter_sdk::release::arm_on(ctx.config(), &self.base, adapter);
        let mut seen = Seen::new(DEFAULT_SHINGLE_WORDS, DEFAULT_MAX_OVERLAP);
        // The instructions held, with the section each came from.
        let mut shown: Vec<(Option<String>, String)> = Vec::new();
        for held in family
            .proposals()
            .into_iter()
            .filter(|p| p.family.as_deref().unwrap_or(crate::language::LUA51) == self.language.name)
        {
            seen.admit(&held.request);
            shown.push((held.section, held.request));
        }
        let mut out = Proposed::default();
        let mut barren = 0;
        let mut next = 0;
        let kinds = [self.language.kind().to_string()];
        let runtime = ctx
            .environments()
            .runtime(self.language.runtime())
            .map_err(|e| anyhow::anyhow!("the {} runtime: {e}", self.language.name))?;
        while out.kept.len() < budget.want && next < plan.len() && barren < 2 {
            // One wave: independent requests, asked together so the server
            // decodes them in one batch. Each is shown the tasks written so
            // far, those of its own section last (the generator shows the
            // most recent), and what they repeat among themselves is dropped
            // below.
            let first = next;
            let wave: Vec<usize> =
                plan[next..(next + budget.parallel.max(1)).min(plan.len())].to_vec();
            next += wave.len();
            let requests: Vec<(usize, String, Vec<String>)> = wave
                .iter()
                .enumerate()
                .map(|(n, &at)| {
                    let key = sections[at].key();
                    let earlier = plan[..first + n].iter().filter(|&&p| p == at).count();
                    let visit = steering.asked.get(&key).copied().unwrap_or(0) + earlier;
                    // seed: per-request literals, derived from the lineage's seed.
                    let request_seed = steering.seed
                        ^ ((first + n) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ at as u64;
                    let mut written: Vec<String> = shown
                        .iter()
                        .filter(|(s, _)| s.as_deref() != Some(key.as_str()))
                        .map(|(_, text)| text.clone())
                        .collect();
                    written.extend(
                        shown
                            .iter()
                            .filter(|(s, _)| s.as_deref() == Some(key.as_str()))
                            .map(|(_, text)| text.clone()),
                    );
                    (
                        at,
                        goal_for(
                            &self.language,
                            &sections[at],
                            steering.accuracy,
                            visit,
                            request_seed,
                        ),
                        written,
                    )
                })
                .collect();
            let generations: Vec<_> = std::thread::scope(|scope| {
                let handles: Vec<_> = requests
                    .iter()
                    .map(|(at, goal, written)| {
                        let (kinds, policy, sections) = (&kinds, &policy, &sections);
                        scope.spawn(move || {
                            tasks::generate(
                                ctx,
                                &Generation {
                                    sources: &[],
                                    sections: std::slice::from_ref(&sections[*at].at),
                                    kinds,
                                    generator: policy,
                                    goal: Some(goal),
                                    author: None,
                                    written,
                                    denoise_seed: None,
                                    deadline: None,
                                    cancel: CancelToken::new(),
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
            for (generated, &at) in generations.into_iter().zip(&wave) {
                out.calls += 1;
                let key = sections[at].key();
                for (reason, n) in &generated.rejected {
                    *out.refused.entry(reason.clone()).or_default() += n;
                }
                let members = ctx.tasks().get_set(&generated.task_set)?.members;
                for entry in members {
                    let task = ctx.tasks().get(&entry.task)?;
                    out.admitted += 1;
                    let Some((program, expected)) =
                        program_and_output(&task, self.language.fence())
                    else {
                        out.unusable += 1;
                        continue;
                    };
                    if self.refuses_output(&mut out, &key, &program, &expected)? {
                        continue;
                    }
                    if !self.language.output_admitted(&expected)
                        || !self.language.rerun_agrees(&runtime, &program, &expected)?
                    {
                        out.inadmissible += 1;
                        continue;
                    }
                    if family.holds(&self.language, &program)
                        || out.kept.iter().any(|k| k.program.trim() == program.trim())
                        || seen.repeats(&task.instruction).is_some()
                    {
                        out.repeats += 1;
                        continue;
                    }
                    seen.admit(&task.instruction);
                    shown.push((Some(key.clone()), task.instruction.clone()));
                    *out.kept_by_section.entry(key.clone()).or_default() += 1;
                    out.kept.push(Kept {
                        program,
                        expected,
                        instruction: task.instruction,
                        section: key.clone(),
                    });
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
        out.kept.truncate(budget.want);
        Ok(out)
    }
}

/// The program a task shows (the last fenced `lua` block of its instruction)
/// and what the interpreter printed for it (the task's reference); `None`
/// when the task is not of that shape or its output is empty or long.
fn program_and_output(task: &Task, fence: &str) -> Option<(String, String)> {
    let open = format!("```{fence}\n");
    let at = task.instruction.rfind(&open)? + open.len();
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

/// Which generation and round proposed a batch, and how it is used.
#[derive(Clone, Copy, Debug)]
pub struct Origin {
    /// The generation whose policy proposed them.
    pub generation: u32,
    /// The round of that generation.
    pub round: u32,
}

/// The proposals as family entries: the first `dev` of them are held out for
/// the dev gate, the next `keep` for retention, the rest are trained on.
#[must_use]
pub fn as_proposals(
    language: &Language,
    kept: Vec<Kept>,
    origin: Origin,
    dev: usize,
    keep: usize,
) -> Vec<Proposal> {
    kept.into_iter()
        .enumerate()
        .map(|(n, k)| Proposal {
            id: proposal_id_in(language, &k.program),
            family: (!language.is_original()).then(|| language.name.clone()),
            generation: origin.generation,
            round: origin.round,
            section: Some(k.section),
            part: if n < dev {
                Part::Dev
            } else if n < dev + keep {
                Part::Keep
            } else {
                Part::Train
            },
            request: if k.instruction.contains(&k.program) {
                k.instruction
            } else {
                language.request(&k.program)
            },
            program: k.program,
            expected: k.expected,
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
            program_and_output(&t, "lua"),
            Some(("print(1+2)".into(), "3".into()))
        );
        assert_eq!(
            program_and_output(&task("no program here", "3"), "lua"),
            None
        );
        assert_eq!(
            program_and_output(&task("q\n\n```lua\nprint(1)\n```", &"x".repeat(500)), "lua"),
            None,
            "an output too long to be graded as a short answer is not a task"
        );
    }

    #[test]
    fn the_first_proposals_are_held_out_for_dev_and_retention_and_the_rest_are_trained_on() {
        let kept = |program: &str, instruction: &str| Kept {
            program: program.into(),
            expected: "x".into(),
            instruction: instruction.into(),
            section: "page#0".into(),
        };
        let proposals = as_proposals(
            &Language::lua51(Path::new("lua"), &[]),
            vec![
                kept("print(1)", "q ```lua\nprint(1)\n```"),
                kept("print(2)", "no program shown"),
                kept("print(3)", "q ```lua\nprint(3)\n```"),
                kept("print(4)", "q ```lua\nprint(4)\n```"),
            ],
            Origin {
                generation: 1,
                round: 1,
            },
            1,
            2,
        );
        assert_eq!(
            proposals.iter().map(|p| p.part).collect::<Vec<_>>(),
            vec![Part::Dev, Part::Keep, Part::Keep, Part::Train]
        );
        assert!(
            proposals[1].request.contains("print(2)"),
            "a request that does not show its program is rebuilt to"
        );
        assert_eq!(proposals[0].section.as_deref(), Some("page#0"));
    }

    #[test]
    fn each_visit_to_a_section_takes_another_angle_and_a_weak_family_is_named() {
        use crate::sections::{family_accuracy, ManualSection};
        use splinter_sdk::knowledge::concepts::SectionRef;
        use splinter_sdk::vocabulary::digest::Digest;
        let section = ManualSection {
            at: SectionRef {
                source: splinter_sdk::vocabulary::source::SourceId(Digest::of(b"page")),
                part: "page.html".into(),
                section: 0,
            },
            functions: ["string.rep".to_string()].into(),
        };
        let none = Accuracy::new();
        let lua = Language::lua51(Path::new("lua"), &[]);
        let angles = lua.angles();
        let goals: Vec<String> = (0..angles.len() + 1)
            .map(|visit| goal_for(&lua, &section, &none, visit, 7))
            .collect();
        let distinct: std::collections::BTreeSet<&String> = goals.iter().collect();
        assert_eq!(
            distinct.len(),
            angles.len(),
            "the angles come round after all of them"
        );
        assert_eq!(goals[0], goals[angles.len()]);
        let weak = family_accuracy((0..4).map(|_| ("print(string.rep('a', 2))", false)));
        let seeded: std::collections::BTreeSet<String> = (0..8)
            .map(|seed| goal_for(&lua, &section, &none, 0, seed))
            .collect();
        assert!(seeded.len() > 4, "requests draw their own literals");
        assert!(goal_for(&lua, &section, &weak, 0, 7).contains("string.rep"));
        assert!(!goal_for(&lua, &section, &none, 0, 7).contains("string.rep"));
    }

    /// The request a Lua lineage puts to its proposer is the one it always
    /// put: the standing goal, the angle, the literals the request must use.
    #[test]
    fn the_original_familys_request_is_the_wording_it_always_had() {
        let lua = Language::lua51(Path::new("lua"), &[]);
        let section = ManualSection {
            at: splinter_sdk::knowledge::concepts::SectionRef {
                source: splinter_sdk::vocabulary::source::SourceId(
                    splinter_sdk::vocabulary::digest::Digest::of(b"page"),
                ),
                part: "page.html".into(),
                section: 0,
            },
            functions: Default::default(),
        };
        let goal = goal_for(&lua, &section, &Accuracy::new(), 0, 7);
        assert!(
            goal.starts_with(
                "Write Lua 5.1 programs that exercise the standard library functions the manual describes, each printing a result a reader has to work out. Use empty, single-character and long arguments. Put the word \""
            ),
            "{goal}"
        );
        assert_eq!(
            lua.request("print(1)\n"),
            "```lua\nprint(1)\n```\nWhat does this program print?"
        );
        assert_eq!(lua.kind(), "predict-lua");
        let kept = Kept {
            program: "print(1)".into(),
            expected: "1".into(),
            instruction: "no program shown".into(),
            section: "p#0".into(),
        };
        let made = as_proposals(
            &lua,
            vec![kept],
            Origin {
                generation: 1,
                round: 1,
            },
            0,
            0,
        );
        assert_eq!(made[0].family, None);
        assert_eq!(made[0].id, crate::family::proposal_id("print(1)"));
    }

    #[test]
    fn another_familys_requests_name_its_language_and_its_kind() {
        let mut python = Language::python3(Path::new("python3"), "sha256:00", &[]);
        python.admission.reruns = 0;
        assert_eq!(
            (python.kind(), python.fence()),
            ("predict-python", "python")
        );
        assert!(python.request("print(1)").starts_with("```python\n"));
        assert!(python.goal().contains("Python 3"));
        let dialect = Language::dialect("zero-index", Path::new("lua"), "sha256:00", &[]).unwrap();
        assert_eq!(dialect.kind(), "predict-lua");
        assert!(dialect.goal().contains("lua51-zero-index"));
        let kept = Kept {
            program: "print(1)".into(),
            expected: "1".into(),
            instruction: "q".into(),
            section: "p#0".into(),
        };
        let made = as_proposals(
            &python,
            vec![kept],
            Origin {
                generation: 1,
                round: 1,
            },
            0,
            0,
        );
        assert_eq!(made[0].family.as_deref(), Some("python3"));
        assert!(made[0].request.starts_with("```python\n"));
    }

    #[test]
    fn a_refused_program_is_counted_as_memorisation_and_logged_with_its_text_and_class() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("refused.jsonl");
        let lua = Language::lua51(Path::new("lua"), &[]);
        let semantic = Proposer::for_language(Path::new("m"), lua.clone(), dir.path())
            .unwrap()
            .with_output_rule(OutputRule::Semantic, &log);
        let mut out = Proposed::default();
        assert!(semantic
            .refuses_output(&mut out, "p#1", "print(1/3)", "0.33333333333333")
            .unwrap());
        assert!(!semantic
            .refuses_output(&mut out, "p#1", "print(7/2)", "3.5")
            .unwrap());
        assert_eq!(out.refused.get("memorisation"), Some(&1));
        let lines = std::fs::read_to_string(&log).unwrap();
        let entry: serde_json::Value = serde_json::from_str(lines.trim()).unwrap();
        assert_eq!(
            (
                entry["program"].as_str(),
                entry["class"].as_str(),
                entry["reason"].as_str()
            ),
            (Some("print(1/3)"), Some("long-float"), Some("memorisation"))
        );
        assert_eq!(entry["family"], "lua51");
        // Without the option nothing is refused and nothing is written.
        let off_log = dir.path().join("off.jsonl");
        let default = Proposer::for_language(Path::new("m"), lua, dir.path())
            .unwrap()
            .with_output_rule(OutputRule::Any, &off_log);
        let mut quiet = Proposed::default();
        assert!(!default
            .refuses_output(&mut quiet, "p#1", "print(1/3)", "0.33333333333333")
            .unwrap());
        assert!(quiet.refused.is_empty() && !off_log.exists());
    }
}
