// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The sample checking itself: no model, no GPU, no network.
//!
//! For every family: TLC finds its specification clean with the recorded
//! number of distinct states; the reference implementation passes the
//! refinement check and reaches exactly that many states; every mutation is
//! admitted only if the check fails for it, the same way twice, on the
//! property it declares; and the Lua unit tests pass.

use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::check::{check_module, check_spec, Verdict};
use crate::family::Family;
use crate::mutation::Mutation;
use crate::tlc::TlcOutcome;
use crate::tools::Tools;

/// Mutations checked at the same time (each runs a JVM).
const PARALLEL_CHECKS: usize = 4;

/// One line of the audit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// What was checked, e.g. `queue mutation boundary-cap`.
    pub subject: String,
    /// `None` when it holds, otherwise why not.
    pub problem: Option<String>,
    /// What was observed, for the report.
    pub observed: String,
}

impl Finding {
    fn new(subject: String, observed: String, problem: Option<String>) -> Self {
        Self {
            subject,
            problem,
            observed,
        }
    }
}

/// What a mutation declares in `violates` when it is safe but reaches fewer
/// states than the specification has.
pub const COVERAGE: &str = "coverage";

/// What a mutation declares in `violates` when only trace validation catches it.
pub const TRACE: &str = "trace";

/// Whether `verdict` is the failure `mutation` declares, given the number of
/// distinct states the specification has.
pub fn is_declared_failure(verdict: &Verdict, mutation: &Mutation, spec_states: u64) -> bool {
    match verdict {
        Verdict::InvariantViolated { name } => *name == mutation.violates,
        Verdict::TraceRejected => mutation.violates == TRACE,
        Verdict::Pass { states, .. } => mutation.violates == COVERAGE && *states < spec_states,
        Verdict::RuntimeError { .. } | Verdict::BoundExceeded => false,
    }
}

/// `f` applied to every item, at most [`PARALLEL_CHECKS`] at a time, results in
/// item order. A panic in `f` is a bug in the checks and propagates.
pub(crate) fn map_in_parallel<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let mut results = Vec::with_capacity(items.len());
    for chunk in items.chunks(PARALLEL_CHECKS) {
        std::thread::scope(|scope| {
            let handles: Vec<_> = chunk.iter().map(|item| scope.spawn(|| f(item))).collect();
            for handle in handles {
                results.push(
                    handle
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
                );
            }
        });
    }
    results
}

pub(crate) fn check_in_fresh_dir(
    tools: &Tools,
    family: &Family,
    module: &Path,
) -> anyhow::Result<Verdict> {
    let work = tempfile::tempdir().context("creating a work directory")?;
    check_module(tools, family, module, work.path())
}

fn audit_mutation(tools: &Tools, family: &Family, mutation: &Mutation, reference: &str) -> Finding {
    let subject = format!("{} mutation {}", family.spec.name, mutation.id);
    let run = || -> anyhow::Result<(Verdict, Verdict)> {
        let mutated = mutation.apply(reference)?;
        let dir = tempfile::tempdir().context("creating a directory for the mutant")?;
        let path = dir.path().join(&family.spec.module);
        std::fs::write(&path, mutated)?;
        Ok((
            check_in_fresh_dir(tools, family, &path)?,
            check_in_fresh_dir(tools, family, &path)?,
        ))
    };
    match run() {
        Err(e) => Finding::new(subject, "not checked".into(), Some(format!("{e:#}"))),
        Ok((first, second)) => {
            let observed = format!("{first:?}");
            let problem = if first != second {
                Some(format!("not deterministic: {first:?} then {second:?}"))
            } else if !is_declared_failure(&first, mutation, family.spec.distinct_states) {
                Some(format!("declared to violate {:?}", mutation.violates))
            } else {
                None
            };
            Finding::new(subject, observed, problem)
        }
    }
}

fn lua_tests(root: &Path, family: &Family) -> Vec<PathBuf> {
    let mut tests = Vec::new();
    for dir in [family.dir.clone(), root.join("lua/tests")] {
        if let Ok(entries) = std::fs::read_dir(dir) {
            tests.extend(
                entries
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| {
                        p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.ends_with("_test.lua"))
                    }),
            );
        }
    }
    tests.sort();
    tests
}

fn audit_lua_test(tools: &Tools, test: &Path) -> Finding {
    let subject = format!("lua test {}", test.display());
    match tools.lua_script(test, &[]) {
        Ok(run) if run.code == Some(0) && run.output.trim() == "ok" => {
            Finding::new(subject, "ok".into(), None)
        }
        Ok(run) => Finding::new(
            subject,
            run.output.trim().to_string(),
            Some(format!("exit code {:?}", run.code)),
        ),
        Err(e) => Finding::new(subject, "not run".into(), Some(format!("{e:#}"))),
    }
}

/// Audits one family.
pub fn audit_family(tools: &Tools, root: &Path, family: &Family) -> anyhow::Result<Vec<Finding>> {
    let name = &family.spec.name;
    let mut findings = Vec::new();

    let work = tempfile::tempdir().context("creating a work directory")?;
    let spec_subject = format!("{name} specification (TLC)");
    findings.push(match check_spec(tools, family, work.path())? {
        TlcOutcome::Clean {
            distinct,
            generated,
        } if distinct == family.spec.distinct_states => Finding::new(
            spec_subject,
            format!("{generated} generated, {distinct} distinct"),
            None,
        ),
        TlcOutcome::Clean { distinct, .. } => Finding::new(
            spec_subject,
            format!("{distinct} distinct states"),
            Some(format!("spec.toml records {}", family.spec.distinct_states)),
        ),
        other => Finding::new(
            spec_subject,
            format!("{other:?}"),
            Some("TLC is not clean".into()),
        ),
    });

    let reference_subject = format!("{name} reference implementation");
    let reference_path = family.module_path();
    findings.push(match check_in_fresh_dir(tools, family, &reference_path)? {
        Verdict::Pass {
            states,
            transitions,
        } if states == family.spec.distinct_states => Finding::new(
            reference_subject,
            format!("{states} states, {transitions} transitions"),
            None,
        ),
        Verdict::Pass { states, .. } => Finding::new(
            reference_subject,
            format!("{states} states"),
            Some(format!(
                "the specification has {} distinct states",
                family.spec.distinct_states
            )),
        ),
        other => Finding::new(
            reference_subject,
            format!("{other:?}"),
            Some("refinement check fails".into()),
        ),
    });

    let reference = std::fs::read_to_string(&reference_path)
        .with_context(|| format!("reading {}", reference_path.display()))?;
    findings.extend(map_in_parallel(&family.mutations, |m| {
        audit_mutation(tools, family, m, &reference)
    }));

    findings.extend(
        lua_tests(root, family)
            .iter()
            .map(|t| audit_lua_test(tools, t)),
    );
    Ok(findings)
}

/// Audits every family below `root/specs`; returns every finding.
pub fn audit(tools: &Tools, root: &Path) -> anyhow::Result<Vec<Finding>> {
    let families = Family::load_all(&root.join("specs"))?;
    anyhow::ensure!(
        !families.is_empty(),
        "no family under {}",
        root.join("specs").display()
    );
    let mut findings = Vec::new();
    for family in &families {
        findings.extend(audit_family(tools, root, family)?);
    }
    Ok(findings)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mutation(violates: &str) -> Mutation {
        Mutation {
            id: "m".into(),
            find: "a".into(),
            replace: "b".into(),
            violates: violates.into(),
            note: String::new(),
        }
    }

    #[test]
    fn only_the_declared_property_admits_a_mutation() {
        let bounded = Verdict::InvariantViolated {
            name: "Bounded".into(),
        };
        assert!(is_declared_failure(&bounded, &mutation("Bounded"), 9));
        assert!(!is_declared_failure(&bounded, &mutation("TypeOK"), 9));
        assert!(!is_declared_failure(&bounded, &mutation("trace"), 9));
        assert!(is_declared_failure(
            &Verdict::TraceRejected,
            &mutation("trace"),
            9
        ));
        assert!(!is_declared_failure(
            &Verdict::TraceRejected,
            &mutation("Bounded"),
            9
        ));
    }

    #[test]
    fn a_safe_mutant_is_admitted_only_when_it_reaches_fewer_states_than_the_specification() {
        let pass = |states| Verdict::Pass {
            states,
            transitions: 1,
        };
        assert!(is_declared_failure(&pass(8), &mutation("coverage"), 9));
        assert!(!is_declared_failure(&pass(9), &mutation("coverage"), 9));
        assert!(!is_declared_failure(&pass(8), &mutation("trace"), 9));
    }

    #[test]
    fn a_lua_error_or_unbounded_growth_is_never_a_spec_violation() {
        let error = Verdict::RuntimeError {
            message: "boom".into(),
        };
        assert!(!is_declared_failure(&error, &mutation("trace"), 9));
        assert!(!is_declared_failure(
            &Verdict::BoundExceeded,
            &mutation("trace"),
            9
        ));
    }
}
