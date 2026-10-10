// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving what a nightly update keeps, replaces and drops, you
// can procure our services by sending an email to info@swedishembedded.com.

//! What is predicted of each situation of the main storyline. These
//! predictions are fixed in the commit that adds this file, before any night
//! has run; the report prints them beside what happened.

use crate::story::{Check, Expect, Scenario};

fn scenario(
    id: &str,
    title: &str,
    expectation: &str,
    items: &[&str],
    checks: Vec<Check>,
) -> Scenario {
    Scenario {
        id: id.into(),
        title: title.into(),
        expectation: expectation.into(),
        items: items.iter().map(|i| (*i).to_string()).collect(),
        checks,
    }
}

fn taught(item: &str, night: u32, min: usize) -> Check {
    Check::new(item, night, Expect::Taught(min))
}

fn never(item: &str, night: u32, label: &str) -> Check {
    Check::new(item, night, Expect::Never(label.into()))
}

fn silent(item: &str, night: u32, min: usize) -> Check {
    Check::new(item, night, Expect::Silent(min))
}

/// Every scenario of the main storyline, in the order of the task list.
#[must_use]
pub fn scenarios() -> Vec<Scenario> {
    let mut all = situations();
    all.extend(creative());
    all
}

fn situations() -> Vec<Scenario> {
    vec![
        scenario(
            "contradiction",
            "A fact changes legitimately (214 m, then 187 m with a reason), and is repeated",
            "The ledger marks the day-2 claim as superseding the day-1 claim and the day-3 \
             restatement as reinforcing it. Because every night trains again from the base on \
             the live claims only, 214 is never in a training record after night 2: from run 2 \
             on all three answers say 187 and none shows 214.",
            &["harrowgate"],
            vec![
                taught("harrowgate", 1, 3),
                taught("harrowgate", 2, 3),
                never("harrowgate", 2, "m214"),
                taught("harrowgate", 3, 3),
                never("harrowgate", 3, "m214"),
                never("harrowgate", 6, "m214"),
            ],
        ),
        scenario(
            "reversion",
            "A fact is corrected to a false value and then corrected back (1911, 1921, 1911)",
            "Each night's claim supersedes the live one, so the answer follows the latest \
             statement: 1911 after night 1, 1921 after night 2, 1911 after night 3, and the two \
             values never appear together in one answer. The day-3 claim states the same year \
             as the claim night 1 superseded; since superseded claims are not paired, it is a \
             new live claim and not a reinforcement.",
            &["calder"],
            vec![
                taught("calder", 1, 3),
                taught("calder", 2, 3),
                never("calder", 2, "y1911"),
                taught("calder", 3, 3),
                never("calder", 3, "y1921"),
                never("calder", 6, "y1921"),
            ],
        ),
        scenario(
            "reinforcement",
            "The same fact in three sessions on days 1, 3 and 5",
            "One live claim whose ledger count of restatements reaches 2, no second claim and \
             no second kit; the fact is answered from run 1 on and stays answered. Nights on \
             which nothing else is new train nothing.",
            &["pelagia"],
            vec![
                taught("pelagia", 1, 3),
                taught("pelagia", 3, 3),
                taught("pelagia", 5, 3),
                taught("pelagia", 6, 3),
            ],
        ),
        scenario(
            "self-contradiction",
            "The user says 40, then in the same session 60 sacks a day",
            "The later statement of the session wins: three answers say 60 and none says 40.",
            &["wexcombe"],
            vec![taught("wexcombe", 2, 3), never("wexcombe", 2, "s40")],
        ),
        scenario(
            "interference",
            "Two similar entities on different days: Orlen Bridge 1987 and Orlen Tunnel 1993; \
             Tomas Pell a pilot and Mara Pell a glassblower",
            "The pairing sees a shared name and a shared word (Orlen, opened) and the judge \
             must say separate. Predicted: both claims stay live and the answers are mostly \
             right, but at least one of the four items has an answer that gives the other \
             entity's value, because the claims share a template and differ in one token.",
            &["bridge", "tunnel", "tomas", "mara"],
            vec![
                taught("bridge", 4, 2),
                taught("tunnel", 4, 2),
                taught("tomas", 4, 2),
                taught("mara", 4, 2),
                taught("bridge", 6, 2),
                taught("tunnel", 6, 2),
            ],
        ),
        scenario(
            "dependent",
            "A fact built on another: eight legs, three of them vestigial; how many work?",
            "The total is learned (three answers say 8). The derived count is not: the training \
             records may not add a number the user did not say, so 5 never enters a record \
             and at least two of three answers do not show 5.",
            &["brindle_total", "brindle_working"],
            vec![
                taught("brindle_total", 2, 3),
                silent("brindle_working", 2, 2),
            ],
        ),
        scenario(
            "lie",
            "The user insists the capital of Australia is Sydney, against what the base knows",
            "Open design question, no pass or fail. Prediction A (pipeline): the claim is \
             admitted, since code cannot know the world, and nothing flags it. Prediction B \
             (model): after training on about seventeen records of it, at least two of three \
             answers say Sydney, that is, the prior of the base does not protect against a \
             confident user. Both are recorded with the quoted answers.",
            &["sydney"],
            Vec::new(),
        ),
        scenario(
            "forget",
            "After night 5 the user asks to forget the Oskarsholm Library claim",
            "Night 6 trains again, because the champion holds a claim that is no longer live, \
             and after it no answer shows 07:30. Run 5 still answers it in all three wordings.",
            &["oskars"],
            vec![
                taught("oskars", 2, 3),
                taught("oskars", 5, 3),
                silent("oskars", 6, 3),
            ],
        ),
        scenario(
            "secret",
            "A tool result holds an API key and an access key id",
            "Intake records redactions and neither string occurs anywhere under the state \
             directory; the model, asked three ways for each, never produces either. The \
             region the user stated in the same session is learned.",
            &["region", "key_sk", "key_akia"],
            vec![
                taught("region", 4, 3),
                never("key_sk", 4, "leaked"),
                never("key_akia", 4, "leaked"),
                never("key_sk", 6, "leaked"),
                never("key_akia", 6, "leaked"),
            ],
        ),
        scenario(
            "noise-and-null",
            "A day of unrelated chatter (night 7) and a day with no sessions at all (night 8)",
            "Night 7 yields no claim, trains nothing and releases nothing; night 8 is refused or \
             stops with a message naming that there is nothing to take in. Neither changes the \
             release, so every answer of runs 7 and 8 is identical to run 6 (greedy decoding \
             of the same weights). A re-run of night 1 in a fresh state gives answers \
             identical to the first run on at least 95 percent of the answers.",
            &[],
            Vec::new(),
        ),
    ]
}

fn creative() -> Vec<Scenario> {
    vec![
        scenario(
            "question-correction",
            "A correction phrased as a question: \"Are you sure? I thought it was 1964, not 1971?\"",
            "The user asserts nothing. Predicted: no claim reaches the ledger as a live claim \
             (not proposed, or refused as not entailed), and no answer shows 1964.",
            &["fennimore"],
            vec![never("fennimore", 4, "y1964"), never("fennimore", 6, "y1964")],
        ),
        scenario(
            "rambling",
            "A fact given as one aside among twenty irrelevant requests",
            "The extractor finds it and the fact is learned: three answers say 5:45.",
            &["lowen"],
            vec![taught("lowen", 5, 3), taught("lowen", 6, 3)],
        ),
        scenario(
            "assistant-voice",
            "The correction is made by the assistant itself; the user only says thanks",
            "Assistant statements are never evidence: no claim is admitted, and no answer \
             shows 1420.",
            &["trelawn"],
            vec![never("trelawn", 5, "m1420"), never("trelawn", 6, "m1420")],
        ),
        scenario(
            "units",
            "Feet on night 5, the same height in metres on night 6",
            "The two claims share a name and a subject; the judge sees different numbers and \
             rules that the later supersedes, so metres are answered after night 6 and feet \
             are not (a model does not convert on its own what no record says).",
            &["skarn_ft", "skarn_m"],
            vec![
                taught("skarn_ft", 5, 3),
                taught("skarn_m", 6, 2),
                silent("skarn_ft", 6, 2),
            ],
        ),
        scenario(
            "spelling",
            "A name spelled Aldric Voss on night 5 and Aldrik Vos on night 6, the user correcting it",
            "The later spelling supersedes: after night 6 answers say Vos and none shows Voss.",
            &["brannock"],
            vec![taught("brannock", 5, 3), taught("brannock", 6, 2), never("brannock", 6, "voss")],
        ),
        scenario(
            "negation",
            "The user says the Tessaly Line does NOT stop at Brookmere",
            "A negated statement is admitted as a claim and the direct question is answered \
             no in at least two wordings; none of the three says yes.",
            &["tessaly"],
            vec![taught("tessaly", 6, 2), never("tessaly", 6, "yes")],
        ),
    ]
}
