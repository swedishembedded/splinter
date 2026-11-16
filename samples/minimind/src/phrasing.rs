// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements instruction grammars that teach agents a
// skill instead of a sentence, for its clients. If your team needs expertise
// in training data that generalises to unseen wording, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The wording of task requests: a combinatorial grammar of openers, verbs,
//! noun phrases, connectives, orderings and destinations, so no surface form
//! dominates and hundreds of distinct forms exist per skill.
//!
//! Two vocabularies: one for every split that is trained on or explored, and
//! a disjoint one for the held-out probe. The held-out probe then measures
//! whether the model reads wording it never saw, not just new combinations
//! of words it did.

use splinter_sdk::vocabulary::rng::SplitMix64;

/// The words one vocabulary offers, by role.
pub struct Words {
    /// What may open a request.
    pub open: &'static [&'static str],
    /// Verbs that fetch a value.
    pub find: &'static [&'static str],
    /// Verbs that store the result.
    pub save: &'static [&'static str],
    /// Where it goes.
    pub into: &'static [&'static str],
    /// Nouns for a value.
    pub value: &'static [&'static str],
    /// How a value sits under its key.
    pub under: &'static [&'static str],
    /// Connectives between two steps.
    pub then: &'static [&'static str],
    /// Verbs that apply a helper.
    pub apply: &'static [&'static str],
    /// Verbs that set a key.
    pub set: &'static [&'static str],
    /// Asking to leave the rest alone.
    pub keep: &'static [&'static str],
    /// Verbs that count.
    pub count: &'static [&'static str],
    /// How a line names a word.
    pub mention: &'static [&'static str],
    /// Verbs that follow a chain.
    pub follow: &'static [&'static str],
}

/// The vocabulary of every trained and explored split. `{a}` in `into` is
/// the file the answer goes to.
pub const TRAIN: Words = Words {
    open: &[
        "",
        "",
        "",
        "Please ",
        "Task: ",
        "In this workspace, ",
        "Next, ",
        "Now ",
        "I need you to ",
        "Quick task: ",
        "Request: ",
        "Go ahead and ",
        "The task: ",
        "Help me: ",
    ],
    find: &[
        "find", "look up", "get", "read", "fetch", "retrieve", "pull", "take", "extract", "grab",
        "check", "query", "collect", "look at", "read off", "pick up",
    ],
    save: &[
        "write", "save", "put", "store", "record", "copy", "note", "dump", "jot down", "keep",
    ],
    into: &[
        "to {a}",
        "in {a}",
        "into {a}",
        "as {a}",
        "in the file {a}",
        "to the file {a}",
        "into a file named {a}",
    ],
    value: &[
        "value", "entry", "setting", "result", "number", "code", "field", "item",
    ],
    under: &[
        "of {k} in {f}",
        "stored under {k} in {f}",
        "listed for {k} in {f}",
        "{k} has in {f}",
        "for {k} from {f}",
        "{k} maps to in {f}",
        "under the key {k} in {f}",
        "keyed by {k} in {f}",
        "next to {k} in {f}",
        "{f} gives for {k}",
    ],
    then: &[
        "and",
        "then",
        "and then",
        "after which you",
        "and next",
        "and finally",
    ],
    apply: &["use", "run", "call", "invoke", "try", "launch", "start"],
    set: &[
        "set", "update", "replace", "edit", "adjust", "alter", "rewrite",
    ],
    keep: &[
        "Leave the other entries as they are.",
        "Keep everything else in the file.",
        "Do not touch the other keys.",
        "Nothing else should be edited.",
        "Leave the rest alone.",
        "Touch no other key.",
    ],
    count: &["count", "tally", "add up", "total", "get the count of"],
    mention: &[
        "contain", "mention", "include", "have", "name", "hold", "match",
    ],
    follow: &["follow", "trace", "walk", "resolve", "track"],
};

/// The held-out probe's vocabulary: none of its words is in [`TRAIN`].
pub const HELD_OUT: Words = Words {
    open: &["Kindly ", "Your job: ", "Here is what to do: "],
    find: &["determine", "work out", "obtain", "locate"],
    save: &["place", "lodge", "enter", "deposit"],
    into: &["inside the file {a}", "within {a}"],
    value: &["figure", "quantity", "amount"],
    under: &["belonging to {k} inside {f}", "assigned to {k} within {f}"],
    then: &["afterwards", "once done,", "subsequently"],
    apply: &["employ", "execute"],
    set: &["change", "overwrite", "modify"],
    keep: &[
        "Preserve all remaining lines unchanged.",
        "Every remaining line must stay intact.",
    ],
    count: &["enumerate", "determine how many are"],
    mention: &["carry", "show"],
    follow: &["chase", "pursue"],
};

/// One of `items`.
pub fn pick<'a>(rng: &mut SplitMix64, items: &[&'a str]) -> &'a str {
    items[rng.below(items.len())]
}

/// `w` with its first letter capitalised.
#[must_use]
pub fn capital(w: &str) -> String {
    let mut c = w.chars();
    c.next()
        .map_or_else(String::new, |f| f.to_uppercase().chain(c).collect())
}

/// "the value of KEY in FILE", in one of the vocabulary's forms.
pub fn of(rng: &mut SplitMix64, w: &Words, key: &str, file: &str) -> String {
    format!(
        "the {} {}",
        pick(rng, w.value),
        pick(rng, w.under).replace("{k}", key).replace("{f}", file)
    )
}

/// A request: `step` (the action, starting with its verb, lower case), then
/// storing `object` in the file `answer`, in one of five orders.
pub fn with_answer(
    rng: &mut SplitMix64,
    w: &Words,
    step: &str,
    object: &str,
    answer: &str,
) -> String {
    let open = pick(rng, w.open);
    let save = pick(rng, w.save);
    let into = pick(rng, w.into).replace("{a}", answer);
    let first = |s: &str| {
        if open.is_empty() {
            capital(s)
        } else {
            s.to_string()
        }
    };
    match rng.below(5) {
        0 => format!(
            "{open}{} {} {save} {object} {into}.",
            first(step),
            pick(rng, w.then)
        ),
        1 => format!("{open}{}. {} {object} {into}.", first(step), capital(save)),
        2 => format!(
            "{open}{} {into}: {step}.",
            first(&format!("{save} {object}"))
        ),
        3 => format!(
            "{open}{} {into} after you {step}.",
            first(&format!("{save} {object}"))
        ),
        _ => format!("{answer} should hold {object}. {}.", capital(step)),
    }
}

/// A request that is one action and a closing sentence, no answer file.
pub fn alone(rng: &mut SplitMix64, w: &Words, step: &str, closing: &str) -> String {
    let open = pick(rng, w.open);
    let step = if open.is_empty() {
        capital(step)
    } else {
        step.to_string()
    };
    format!("{open}{step}. {closing}")
}

/// Every word of a vocabulary, for checking two are disjoint.
#[must_use]
pub fn words(w: &Words) -> Vec<String> {
    let lists: [&[&str]; 13] = [
        w.open, w.find, w.save, w.into, w.value, w.under, w.then, w.apply, w.set, w.keep, w.count,
        w.mention, w.follow,
    ];
    let mut out: Vec<String> = lists
        .iter()
        .flat_map(|l| l.iter())
        .flat_map(|p| p.split(|c: char| !c.is_ascii_alphabetic()))
        .filter(|t| t.len() > 2)
        .map(str::to_lowercase)
        .collect();
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Words both vocabularies may share: grammar glue and the file name.
    const GLUE: [&str; 6] = ["the", "answer", "txt", "file", "all", "are"];

    #[test]
    fn the_held_out_vocabulary_shares_no_word_with_training() {
        let train = words(&TRAIN);
        let shared: Vec<String> = words(&HELD_OUT)
            .into_iter()
            .filter(|w| train.contains(w) && !GLUE.contains(&w.as_str()))
            .collect();
        assert!(shared.is_empty(), "shared words: {shared:?}");
    }
}
