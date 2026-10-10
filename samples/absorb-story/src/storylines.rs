// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving what a nightly update keeps, replaces and drops, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The storylines that can be run, by name.

use crate::main_story;
use crate::scale_story;
use crate::story::{Scenario, Storyline};

/// The names of the storylines, in the order they are worth running.
pub const NAMES: [&str; 6] = [
    "main",
    "rerun",
    "scale-once",
    "scale-spread",
    "reverse",
    "continue",
];

/// The nights of the main storyline the ablation and the re-run replay.
const CONTINUE_NIGHTS: usize = 6;

/// The storyline called `name`.
///
/// # Errors
/// There is no such storyline.
pub fn by_name(name: &str) -> Result<Storyline, String> {
    let main = main_story::storyline();
    match name {
        "main" => Ok(main),
        "rerun" => Ok(Storyline {
            name: name.into(),
            nights: main.nights.into_iter().take(1).collect(),
            scenarios: vec![rerun_scenario()],
            ..main
        }),
        "continue" => Ok(Storyline {
            name: name.into(),
            nights: main.nights.into_iter().take(CONTINUE_NIGHTS).collect(),
            absorb_args: vec!["--continue-from-release".into()],
            scenarios: vec![continue_scenario()],
            ..main
        }),
        "reverse" => {
            let mut reversed = main.reversed(name);
            reversed.scenarios = vec![order_scenario()];
            Ok(reversed)
        }
        "scale-once" => Ok(scale_story::once()),
        "scale-spread" => Ok(scale_story::spread()),
        other => Err(format!(
            "no storyline {other:?}; the storylines are {NAMES:?}"
        )),
    }
}

fn rerun_scenario() -> Scenario {
    Scenario {
        id: "determinism".into(),
        title: "Night 1 run again from an empty state".into(),
        expectation: "The sessions, the extraction and the training are the same, so the answers \
            of this run are identical to those of run 1 of the main storyline on at least 95 percent \
            of the answers (greedy decoding; the training on a GPU is not promised bit-exact)."
            .into(),
        items: Vec::new(),
        checks: Vec::new(),
    }
}

fn continue_scenario() -> Scenario {
    Scenario {
        id: "forgetting-curve".into(),
        title: "Retention of the early facts when nights continue the release instead of training again".into(),
        expectation: "Training again from the base on every live claim keeps the day-1 facts \
            answered in every run of the main storyline. Continuing from the release with a quarter \
            of the earlier records replayed keeps them too in the first nights, but loses at least \
            one day-1 answer by run 6, and cannot take the claim forgotten before night 6 out of \
            the weights."
            .into(),
        items: vec!["harrowgate".into(), "calder".into(), "pelagia".into(), "bridge".into(), "tomas".into()],
        checks: Vec::new(),
    }
}

fn order_scenario() -> Scenario {
    Scenario {
        id: "order".into(),
        title: "The same sessions in the reverse order of days".into(),
        expectation: "Facts taught once and independent of each other (pelagia, bridge, tunnel, \
            tomas, mara, brindle_total, lowen, region, tessaly) get the same answers in either \
            order on at least 90 percent of the answers. Facts that were changed by later \
            sessions flip to the other value: harrowgate answers 214, calder answers 1921, \
            wexcombe is unchanged (one session), brannock answers Voss, skarn_ft and skarn_m keep \
            the value of the other order's earlier night."
            .into(),
        items: Vec::new(),
        checks: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_named_storyline_exists_and_is_well_formed() {
        for name in NAMES {
            let story = by_name(name).expect(name);
            assert_eq!(story.name, name);
            assert_eq!(story.validate(), Ok(()), "{name}");
        }
        assert!(by_name("nope").is_err());
    }

    #[test]
    fn the_ablation_continues_the_release() {
        let story = by_name("continue").expect("continue");
        assert_eq!(story.absorb_args, ["--continue-from-release"]);
        assert_eq!(story.nights.len(), CONTINUE_NIGHTS);
    }
}
