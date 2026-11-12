// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for agents that keep what they have solved and reuse it, for its
// clients. If your team needs expertise in libraries of verified sub-solutions for compound tasks then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A library of verified sub-solutions.
//!
//! A compound task is made of sub-tasks that recur in other compound tasks. What an agent solved and a verifier
//! passed is kept under the sub-task's key; the next compound task that contains the sub-task takes the stored
//! solution instead of spending the model on it. Only what a verifier passed is stored, so the library never
//! hands back an answer nobody checked, and the first verified solution of a key stays.

use std::collections::BTreeMap;

/// Verified solutions of sub-tasks, by the sub-task's text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SkillLibrary {
    solved: BTreeMap<String, String>,
}

/// A compound task solved with a library: the answer of each part, where it came from and what the model spent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solution {
    /// The answer to each part, in order.
    pub answers: Vec<String>,
    /// Parts the library answered.
    pub from_library: usize,
    /// Tokens the model generated for the parts the library did not answer.
    pub tokens: usize,
}

impl SkillLibrary {
    /// An empty library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores `answer` as the solution of `task` if the verifier passed it; a task already solved keeps its first
    /// solution. Returns whether it was stored.
    pub fn record(&mut self, task: &str, answer: &str, verified: bool) -> bool {
        if !verified || self.solved.contains_key(task) {
            return false;
        }
        self.solved.insert(task.to_string(), answer.to_string());
        true
    }

    /// The verified solution of `task`, if the library holds one.
    #[must_use]
    pub fn lookup(&self, task: &str) -> Option<&str> {
        self.solved.get(task).map(String::as_str)
    }

    /// Sub-tasks solved.
    #[must_use]
    pub fn len(&self) -> usize {
        self.solved.len()
    }

    /// Whether nothing is stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.solved.is_empty()
    }

    /// Solves a compound task part by part: a part the library holds is answered from it at no cost, any other is
    /// handed to `ask`, which returns the model's answer and the tokens it generated.
    pub fn solve(
        &self,
        parts: &[String],
        mut ask: impl FnMut(&str) -> (String, usize),
    ) -> Solution {
        let mut solution = Solution {
            answers: Vec::new(),
            from_library: 0,
            tokens: 0,
        };
        for part in parts {
            if let Some(stored) = self.lookup(part) {
                solution.answers.push(stored.to_string());
                solution.from_library += 1;
            } else {
                let (answer, tokens) = ask(part);
                solution.answers.push(answer);
                solution.tokens += tokens;
            }
        }
        solution
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_verified_solution_is_stored_and_the_first_one_stays() {
        let mut library = SkillLibrary::new();
        assert!(!library.record("abc", "wrong", false));
        assert!(library.record("abc", "bcd", true));
        assert!(!library.record("abc", "other", true));
        assert_eq!(
            (library.lookup("abc"), library.lookup("xyz"), library.len()),
            (Some("bcd"), None, 1)
        );
    }

    #[test]
    fn a_stored_part_costs_no_tokens_and_the_rest_is_asked() {
        let mut library = SkillLibrary::new();
        library.record("a", "A", true);
        let parts = ["a".to_string(), "b".to_string()];
        let solution = library.solve(&parts, |part| (part.to_uppercase(), 3));
        assert_eq!(
            solution,
            Solution {
                answers: vec!["A".into(), "B".into()],
                from_library: 1,
                tokens: 3
            }
        );
        let alone = SkillLibrary::new().solve(&parts, |part| (part.to_uppercase(), 3));
        assert_eq!((alone.from_library, alone.tokens), (0, 6));
    }
}
