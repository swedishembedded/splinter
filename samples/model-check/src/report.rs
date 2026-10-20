// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-checked verification of concurrent and
// distributed protocols for its clients. If your team needs expertise in
// TLA+ specifications, TLC and checking implementations against them, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The mutation-detection table: for every mutant of a family, what the
//! explorer finds in the implementation by itself and what TLC finds when it
//! validates the implementation's behaviours against the specification.

use std::path::Path;

use anyhow::Context;

use crate::audit::map_in_parallel;
use crate::check::check_traces;
use crate::explorer::{self, Exploration};
use crate::family::Family;
use crate::mutation::Mutation;
use crate::tlc::TlcOutcome;
use crate::tools::Tools;

/// One mutant's row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    /// The mutation's id.
    pub id: String,
    /// The property the mutation declares it violates.
    pub declared: String,
    /// What the explorer reports on its own, invariants first.
    pub explorer: String,
    /// What TLC reports on the trace module (invariants of the
    /// implementation not consulted first), or why there was none.
    pub tlc: String,
    /// Distinct states the mutant reaches, when it ran.
    pub states: Option<u64>,
}

fn explorer_label(exploration: &Exploration) -> String {
    match exploration {
        Exploration::Complete { .. } => "no violation".into(),
        Exploration::InvariantViolated { name } => format!("{name} violated"),
        Exploration::RuntimeError { message } => format!("Lua error: {message}"),
        Exploration::BoundExceeded => "state bound exceeded".into(),
        Exploration::Unreadable { message } => format!("unreadable: {message}"),
    }
}

fn tlc_label(outcome: &TlcOutcome) -> String {
    match outcome {
        TlcOutcome::Clean { .. } => "accepted".into(),
        TlcOutcome::InvariantViolated { name } => format!("{name} violated"),
        TlcOutcome::Deadlock => "rejected: a step the specification does not allow".into(),
        TlcOutcome::Failed { message } => format!("TLC failed: {message}"),
    }
}

fn row(
    tools: &Tools,
    family: &Family,
    mutation: &Mutation,
    reference: &str,
) -> anyhow::Result<Row> {
    let mutated = mutation.apply(reference)?;
    let dir = tempfile::tempdir().context("creating a directory for the mutant")?;
    let path = dir.path().join(&family.spec.module);
    std::fs::write(&path, mutated)?;
    let run = tools.lua_program("explore.lua", &[path.as_os_str()])?;
    let explorer = explorer_label(&explorer::parse(&run.output, run.code));
    let work = tempfile::tempdir().context("creating a work directory")?;
    let (tlc, states) = match check_traces(tools, family, &path, work.path(), false) {
        Ok(traces) => (tlc_label(&traces.outcome), Some(traces.states)),
        Err(e) => (format!("no trace module: {e:#}"), None),
    };
    Ok(Row {
        id: mutation.id.clone(),
        declared: mutation.violates.clone(),
        explorer,
        tlc,
        states,
    })
}

/// The rows of `family`'s mutants, in file order.
pub fn rows(tools: &Tools, family: &Family) -> anyhow::Result<Vec<Row>> {
    let path = family.module_path();
    let reference =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    map_in_parallel(&family.mutations, |m| row(tools, family, m, &reference))
        .into_iter()
        .collect()
}

/// The Markdown tables of every family below `root/specs`.
pub fn report(tools: &Tools, root: &Path) -> anyhow::Result<String> {
    let families = Family::load_all(&root.join("specs"))?;
    let mut out = String::new();
    for family in &families {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&markdown(
            &family.spec.name,
            family.spec.distinct_states,
            &rows(tools, family)?,
        ));
    }
    Ok(out)
}

/// A Markdown table of `rows`; `spec_states` is the specification's distinct states.
pub fn markdown(family: &str, spec_states: u64, rows: &[Row]) -> String {
    let mut out = format!(
        "Family {family} (specification: {spec_states} distinct states)\n\n\
         | mutant | declared | explorer | TLC | states |\n|---|---|---|---|---|\n"
    );
    for r in rows {
        let states = r.states.map_or_else(|| "-".to_string(), |s| s.to_string());
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            r.id, r.declared, r.explorer, r.tlc, states
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_has_a_row_per_mutant_and_a_dash_for_an_unreached_state_count() {
        let rows = [
            Row {
                id: "a".into(),
                declared: "Bounded".into(),
                explorer: "Bounded violated".into(),
                tlc: "Bounded violated".into(),
                states: Some(12),
            },
            Row {
                id: "b".into(),
                declared: "trace".into(),
                explorer: "Lua error: boom".into(),
                tlc: "no trace module".into(),
                states: None,
            },
        ];
        let table = markdown("queue", 35, &rows);
        assert!(table.starts_with("Family queue (specification: 35 distinct states)"));
        assert!(table.contains("| a | Bounded | Bounded violated | Bounded violated | 12 |\n"));
        assert!(table.contains("| b | trace | Lua error: boom | no trace module | - |\n"));
    }
}
