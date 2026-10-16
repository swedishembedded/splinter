// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifier conformance suites that tell a
// verifier that is too lenient from one that is too strict, for its clients.
// If your team needs expertise in measuring the false-pass and false-fail
// rates of automatic graders, you can procure our services by sending an
// email to info@swedishembedded.com.

//! A conformance table: what a verifier should decide for each case of a
//! class of input, and what it decided.
//!
//! Two errors are different in kind and counted apart. A **false pass** (the
//! verifier passes what should not pass) puts a wrong answer into training
//! data and a cheat into a release. A **false fail** (it fails what is right)
//! throws away real work and makes a capability look absent. A verdict
//! that is neither (an abstention where a decision was due, or a fail where
//! the verifier should have abstained) is **misrouted**.
//!
//! A **known gap** is a case the verifier is documented to get wrong today.
//! It is reported, and asserted still to be wrong, so that fixing it makes
//! the table say so instead of passing silently.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;

use splinter_core::annotation::Outcome;

/// The kinds of input every table is built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    /// The right answer, written plainly.
    Correct,
    /// A wrong answer, including ones that look plausible.
    Incorrect,
    /// The right answer written another way.
    Equivalent,
    /// The boundary of the input: empty, huge, odd framing, limits.
    Edge,
    /// No answer, no check, an environment that was not offered.
    MissingEvidence,
    /// The process ends before the check can have run.
    EarlyExit,
    /// An attempt to use what only the grader holds, or to leak it.
    HiddenDisclosure,
    /// An attempt to change what the verifier reads or runs.
    ConfigTampering,
    /// Output made to look like a runner's success.
    LogMimicry,
}

/// What a case should decide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Want {
    /// Pass it.
    Pass,
    /// Fail it.
    Fail,
    /// Neither: leave it to a judge or another verifier.
    Abstain,
}

impl Want {
    fn matches(self, got: Outcome) -> bool {
        matches!(
            (self, got),
            (Self::Pass, Outcome::Pass)
                | (Self::Fail, Outcome::Fail)
                | (Self::Abstain, Outcome::Abstain)
        )
    }
}

struct Case {
    class: Class,
    name: String,
    want: Want,
    got: Outcome,
    gap: bool,
}

/// The cases of one verifier and what it decided for them.
pub struct Table {
    verifier: &'static str,
    cases: Vec<Case>,
}

impl Table {
    /// An empty table for `verifier`.
    pub fn new(verifier: &'static str) -> Self {
        Self {
            verifier,
            cases: Vec::new(),
        }
    }

    /// A case the verifier must decide as `want`.
    pub fn case(&mut self, class: Class, name: &str, want: Want, got: Outcome) {
        self.cases.push(Case {
            class,
            name: name.into(),
            want,
            got,
            gap: false,
        });
    }

    /// A case the verifier is known to decide wrongly: `want` is what it
    /// should decide. Asserted still wrong.
    pub fn gap(&mut self, class: Class, name: &str, want: Want, got: Outcome) {
        assert!(
            !want.matches(got),
            "{}: the known gap {name:?} is closed ({got:?}): make it a case",
            self.verifier
        );
        self.cases.push(Case {
            class,
            name: name.into(),
            want,
            got,
            gap: true,
        });
    }

    fn live(&self) -> impl Iterator<Item = &Case> {
        self.cases.iter().filter(|c| !c.gap)
    }

    /// Cases passed that should not have been.
    pub fn false_passes(&self) -> Vec<&str> {
        self.live()
            .filter(|c| c.got == Outcome::Pass && c.want != Want::Pass)
            .map(|c| c.name.as_str())
            .collect()
    }

    /// Cases failed that should have passed.
    pub fn false_fails(&self) -> Vec<&str> {
        self.live()
            .filter(|c| c.got == Outcome::Fail && c.want == Want::Pass)
            .map(|c| c.name.as_str())
            .collect()
    }

    /// Cases decided neither right nor as a false pass or fail.
    pub fn misrouted(&self) -> Vec<&str> {
        self.live()
            .filter(|c| !c.want.matches(c.got))
            .filter(|c| !(c.got == Outcome::Pass && c.want != Want::Pass))
            .filter(|c| !(c.got == Outcome::Fail && c.want == Want::Pass))
            .map(|c| c.name.as_str())
            .collect()
    }

    /// The counts, on one line.
    pub fn report(&self) -> String {
        let known: Vec<&str> = self
            .cases
            .iter()
            .filter(|c| c.gap)
            .map(|c| c.name.as_str())
            .collect();
        format!(
            "conformance {}: cases={} false_pass={} false_fail={} misrouted={} known_gaps={}",
            self.verifier,
            self.cases.len(),
            self.false_passes().len(),
            self.false_fails().len(),
            self.misrouted().len(),
            known.len()
        )
    }

    /// Prints the report, and fails when a live case was decided wrongly or
    /// a class in `required` has no case.
    pub fn assert_conforms(&self, required: &[Class]) {
        println!("{}", self.report());
        for c in self.cases.iter().filter(|c| c.gap) {
            println!(
                "  known gap [{:?}] {}: want {:?}, got {:?}",
                c.class, c.name, c.want, c.got
            );
        }
        let present: BTreeSet<Class> = self.cases.iter().map(|c| c.class).collect();
        let absent: Vec<Class> = required
            .iter()
            .copied()
            .filter(|c| !present.contains(c))
            .collect();
        assert!(
            absent.is_empty(),
            "{}: no case of {absent:?}",
            self.verifier
        );
        assert!(
            self.false_passes().is_empty(),
            "{}: false passes: {:?}",
            self.verifier,
            self.false_passes()
        );
        assert!(
            self.false_fails().is_empty(),
            "{}: false fails: {:?}",
            self.verifier,
            self.false_fails()
        );
        assert!(
            self.misrouted().is_empty(),
            "{}: misrouted: {:?}",
            self.verifier,
            self.misrouted()
        );
    }
}
