// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The refinement check: does a Lua implementation behave as its TLA+
//! specification says?
//!
//! Two independent halves, both deterministic. The explorer enumerates the
//! implementation's reachable states and checks its invariants (which carry the
//! specification's names). If it finds none violated, its behaviours are
//! written as a TLA+ trace module and TLC validates that each one is a
//! behaviour of the specification.

use std::ffi::OsStr;
use std::path::Path;

use anyhow::{bail, Context};

use crate::explorer::{self, Exploration};
use crate::family::Family;
use crate::tlc::{self, TlcOutcome};
use crate::tools::Tools;

/// The result of checking one implementation against its family's specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Invariants hold and TLC accepted every explored behaviour.
    Pass {
        /// Distinct states the implementation reaches.
        states: u64,
        /// Transitions it executed.
        transitions: u64,
    },
    /// The implementation reaches a state violating a specification invariant.
    InvariantViolated {
        /// The violated invariant.
        name: String,
    },
    /// Invariants hold but TLC rejects an explored behaviour: the
    /// implementation takes a step the specification does not allow.
    TraceRejected,
    /// The module failed on its own (not a specification violation).
    RuntimeError {
        /// What the explorer reported.
        message: String,
    },
    /// The implementation has more states than the exploration bound.
    BoundExceeded,
}

/// The trace module's TLC configuration: the family's own constants and
/// invariants, checked on the validated behaviours, with the trace
/// specification in place of the family's and deadlock detection on, which is
/// how a behaviour the specification cannot follow shows up.
pub fn trace_cfg(base: &str) -> String {
    let mut cfg: String = base
        .lines()
        .filter(|l| {
            let word = l.split_whitespace().next().unwrap_or_default();
            !matches!(word, "SPECIFICATION" | "INIT" | "NEXT" | "CHECK_DEADLOCK")
        })
        .map(|l| format!("{l}\n"))
        .collect();
    cfg.push_str("SPECIFICATION TSpec\n");
    cfg
}

fn path_arg(path: &Path) -> &OsStr {
    path.as_os_str()
}

/// What TLC said about the behaviours of an implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceCheck {
    /// TLC's outcome on the trace module.
    pub outcome: TlcOutcome,
    /// Distinct states the implementation reaches.
    pub states: u64,
    /// Transitions it executed.
    pub transitions: u64,
}

/// Writes the trace module of `module` into `work` and has TLC validate it
/// against the family's specification. With `check_invariants` false the
/// implementation's own invariants are not consulted first, so TLC sees the
/// behaviours of a module the explorer would already have rejected.
pub fn check_traces(
    tools: &Tools,
    family: &Family,
    module: &Path,
    work: &Path,
    check_invariants: bool,
) -> anyhow::Result<TraceCheck> {
    let spec_module = family.tla_module()?;
    let mut args = vec![path_arg(module), OsStr::new(spec_module), path_arg(work)];
    if !check_invariants {
        args.push(OsStr::new("--no-invariants"));
    }
    let generated = tools.lua_program("tracegen.lua", &args)?;
    if generated.code != Some(0) {
        bail!(
            "trace generation for {} failed: {}",
            module.display(),
            generated.output
        );
    }
    let (states, transitions) = (
        field_number(&generated.output, "states")?,
        field_number(&generated.output, "transitions")?,
    );

    std::fs::copy(
        family.dir.join(&family.spec.tla),
        work.join(&family.spec.tla),
    )
    .with_context(|| format!("copying {}", family.spec.tla))?;
    let base_cfg = std::fs::read_to_string(family.dir.join(&family.spec.cfg))
        .with_context(|| format!("reading {}", family.spec.cfg))?;
    let trace_cfg_name = format!("{spec_module}Trace.cfg");
    std::fs::write(work.join(&trace_cfg_name), trace_cfg(&base_cfg))?;
    let run = tools.tlc(work, &trace_cfg_name, &format!("{spec_module}Trace.tla"))?;
    Ok(TraceCheck {
        outcome: tlc::parse(&run.output, run.code),
        states,
        transitions,
    })
}

/// Checks `module` (the reference or a mutant of it) against `family`,
/// working in the empty directory `work`.
pub fn check_module(
    tools: &Tools,
    family: &Family,
    module: &Path,
    work: &Path,
) -> anyhow::Result<Verdict> {
    let run = tools.lua_program("explore.lua", &[path_arg(module)])?;
    match explorer::parse(&run.output, run.code) {
        Exploration::InvariantViolated { name } => return Ok(Verdict::InvariantViolated { name }),
        Exploration::RuntimeError { message } => return Ok(Verdict::RuntimeError { message }),
        Exploration::BoundExceeded => return Ok(Verdict::BoundExceeded),
        Exploration::Unreadable { message } => bail!(
            "explorer on {}: {message}\n{}",
            module.display(),
            run.output
        ),
        Exploration::Complete { .. } => {}
    }
    let traces = check_traces(tools, family, module, work, true)?;
    match traces.outcome {
        TlcOutcome::Clean { .. } => Ok(Verdict::Pass {
            states: traces.states,
            transitions: traces.transitions,
        }),
        TlcOutcome::Deadlock | TlcOutcome::InvariantViolated { .. } => Ok(Verdict::TraceRejected),
        TlcOutcome::Failed { message } => {
            bail!("TLC on the trace of {}: {message}", module.display())
        }
    }
}

fn field_number(output: &str, key: &str) -> anyhow::Result<u64> {
    output
        .lines()
        .find_map(|l| l.strip_prefix(key)?.strip_prefix('='))
        .and_then(|v| v.trim().parse().ok())
        .with_context(|| format!("no {key}= in trace generator output: {output}"))
}

/// Runs TLC on the family's own specification, in the empty directory `work`.
pub fn check_spec(tools: &Tools, family: &Family, work: &Path) -> anyhow::Result<TlcOutcome> {
    std::fs::copy(
        family.dir.join(&family.spec.tla),
        work.join(&family.spec.tla),
    )
    .with_context(|| format!("copying {}", family.spec.tla))?;
    std::fs::copy(
        family.dir.join(&family.spec.cfg),
        work.join(&family.spec.cfg),
    )
    .with_context(|| format!("copying {}", family.spec.cfg))?;
    let run = tools.tlc(work, &family.spec.cfg, &family.spec.tla)?;
    Ok(tlc::parse(&run.output, run.code))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_cfg_keeps_constants_and_invariants_and_swaps_the_specification() {
        let base = "SPECIFICATION Spec\nCONSTANTS Cap = 2\nINVARIANT Bounded\nINVARIANT TypeOK\nCHECK_DEADLOCK FALSE\n";
        assert_eq!(
            trace_cfg(base),
            "CONSTANTS Cap = 2\nINVARIANT Bounded\nINVARIANT TypeOK\nSPECIFICATION TSpec\n"
        );
    }

    #[test]
    fn trace_cfg_drops_init_next_style_configurations_too() {
        assert_eq!(
            trace_cfg("INIT Init\nNEXT Next\nINVARIANT I\n"),
            "INVARIANT I\nSPECIFICATION TSpec\n"
        );
    }
}
