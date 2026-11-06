// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements comprehension drills made by code from a
// codebase's own functions and tests, for its clients. If your team needs
// expertise in turning a repository into verified training tasks you can
// procure our services by sending an email to info@swedishembedded.com.

//! The inputs of a drill: calls of a function drawn, by a seed, from the
//! literal arguments the repository's own tests call it with.
//!
//! The calls the tests make are candidates as they are, and so are
//! recombinations of them: for each shape of call (how many positional
//! arguments, which keywords) every argument position is filled by one of the
//! literals some test passed in that position. Both are ranked by a hash of
//! the seed and the call, so the same seed on the same tests draws the same
//! calls whatever order the tests are read in.

use std::collections::{BTreeMap, BTreeSet};

use splinter_core::family::rank;
use splinter_core::rng::Seed;

use super::{Call, FunctionSource};

/// How many recombinations of each shape are made per drill asked for.
const RECOMBINATIONS_PER_DRILL: usize = 3;

/// How many candidates are offered per drill asked for: a candidate is not
/// admitted when it errors or prints what another already printed, so more
/// are offered than are wanted.
const CANDIDATES_PER_DRILL: usize = 6;

/// The first 64 bits of the hash of `label` under `seed`.
fn draw(seed: Seed, label: &str) -> u64 {
    let ranked = rank(label, seed);
    let hex = ranked.rsplit(':').next().unwrap_or(&ranked);
    u64::from_str_radix(hex.get(..16).unwrap_or("0"), 16).unwrap_or(0)
}

/// The calls of `function` to try, best first, at most a bounded number; `None`
/// when the function needs arguments and no test gives any.
pub(super) fn candidates(
    function: &FunctionSource,
    seed: Seed,
    per_function: usize,
) -> Option<Vec<Call>> {
    let needs_arguments = function
        .params
        .iter()
        .any(|param| !param.variadic && !param.has_default);
    if function.calls.is_empty() {
        return (!needs_arguments).then(|| {
            vec![Call {
                args: Vec::new(),
                kwargs: Vec::new(),
            }]
        });
    }
    let label = format!("{}::{}", function.file, function.name);
    let mut found: BTreeSet<Call> = function.calls.iter().cloned().collect();
    let mut shapes: BTreeMap<(usize, Vec<&str>), Vec<&Call>> = BTreeMap::new();
    for call in &function.calls {
        let names = call.kwargs.iter().map(|(name, _)| name.as_str()).collect();
        shapes
            .entry((call.args.len(), names))
            .or_default()
            .push(call);
    }
    for ((arity, names), calls) in &shapes {
        let positional: Vec<Vec<&str>> = (0..*arity)
            .map(|at| distinct(calls.iter().map(|call| call.args[at].as_str())))
            .collect();
        let keyword: Vec<Vec<&str>> = (0..names.len())
            .map(|at| distinct(calls.iter().map(|call| call.kwargs[at].1.as_str())))
            .collect();
        for n in 0..RECOMBINATIONS_PER_DRILL * per_function {
            let pick = |pool: &[&str], at: usize| {
                let drawn = draw(seed, &format!("{label}|{arity}|{names:?}|{n}|{at}"));
                // The pool is never empty: it holds the arguments of at least one call.
                pool[usize::try_from(drawn % pool.len() as u64).unwrap_or(0)].to_string()
            };
            found.insert(Call {
                args: positional
                    .iter()
                    .enumerate()
                    .map(|(at, pool)| pick(pool, at))
                    .collect(),
                kwargs: names
                    .iter()
                    .zip(&keyword)
                    .enumerate()
                    .map(|(at, (name, pool))| ((*name).to_string(), pick(pool, arity + at)))
                    .collect(),
            });
        }
    }
    let mut ranked: Vec<(String, Call)> = found
        .into_iter()
        .map(|call| (rank(&format!("{label}|{}", call.text()), seed), call))
        .collect();
    ranked.sort();
    Some(
        ranked
            .into_iter()
            .map(|(_, call)| call)
            .take(CANDIDATES_PER_DRILL * per_function)
            .collect(),
    )
}

fn distinct<'a>(values: impl Iterator<Item = &'a str>) -> Vec<&'a str> {
    values.collect::<BTreeSet<_>>().into_iter().collect()
}

impl Call {
    /// The arguments as they are written between a call's parentheses.
    pub(super) fn text(&self) -> String {
        self.args
            .iter()
            .cloned()
            .chain(
                self.kwargs
                    .iter()
                    .map(|(name, value)| format!("{name}={value}")),
            )
            .collect::<Vec<_>>()
            .join(", ")
    }
}
