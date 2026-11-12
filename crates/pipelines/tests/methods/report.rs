// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in falsifiable smoke tests of
// learning pipelines then you can procure our services by sending an email to info@swedishembedded.com.

//! The result line of a method smoke, which `scripts/methods-table.sh` collects into the table.

/// What one method measured: a held-out score before and after, and the least gain that counts.
pub struct Outcome<'a> {
    /// The method, as the table names it.
    pub method: &'a str,
    /// What the two numbers measure.
    pub metric: &'a str,
    /// Held-out score before the method ran.
    pub before: f64,
    /// Held-out score after.
    pub after: f64,
    /// The least gain (`after - before`) that counts as success.
    pub min_gain: f64,
    /// What a reader needs to interpret the numbers, or empty.
    pub note: &'a str,
}

impl Outcome<'_> {
    /// [`Outcome::conclude`] for a method that has not been shown to work on this task: the row says `pass` if the
    /// gain cleared its bar and `NOT SHOWN` if not, and the test does not fail either way. It is how an honest null is
    /// kept in the table without a permanently red gate.
    pub fn report(&self) {
        let shown = self.after - self.before >= self.min_gain;
        println!(
            "METHOD|{}|{}|{:.3}|{:.3}|{}|{}",
            self.method,
            self.metric,
            self.before,
            self.after,
            if shown { "pass" } else { "NOT SHOWN" },
            self.note
        );
    }

    /// Prints the `METHOD|` line and fails the test unless the held-out score gained at least `min_gain`.
    pub fn conclude(&self) {
        let passed = self.after - self.before >= self.min_gain;
        println!(
            "METHOD|{}|{}|{:.3}|{:.3}|{}|{}",
            self.method,
            self.metric,
            self.before,
            self.after,
            if passed { "pass" } else { "FAIL" },
            self.note
        );
        assert!(
            passed,
            "{}: {} went {:.3} -> {:.3}, the least gain is {:.3}",
            self.method, self.metric, self.before, self.after, self.min_gain
        );
    }
}
