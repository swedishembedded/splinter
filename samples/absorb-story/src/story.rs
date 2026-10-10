// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving what a nightly update keeps, replaces and drops, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A storyline: the nights of sessions a user has, what each session teaches
//! about which item, and what was predicted of it before anything ran.

use serde::Serialize;

use crate::atif::Script;
use crate::item::{Item, Right, PHRASINGS};

/// What a session teaches about an item, as the user means it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Lesson {
    /// The item's id.
    pub item: String,
    /// The label of the value that is right once the session is absorbed;
    /// `None` when nothing should be answered afterwards.
    pub value: Option<String>,
}

impl Lesson {
    /// A session that leaves `value` of `item` right.
    #[must_use]
    pub fn right(item: &str, value: &str) -> Self {
        Self {
            item: item.into(),
            value: Some(value.into()),
        }
    }

    /// A session after which `item` should not be answered.
    #[must_use]
    pub fn nothing(item: &str) -> Self {
        Self {
            item: item.into(),
            value: None,
        }
    }
}

/// A session of a night.
#[derive(Clone, Debug)]
pub struct Session {
    /// The scripted conversation.
    pub script: Script,
    /// What it teaches, in the user's intent; empty for chatter.
    pub lessons: Vec<Lesson>,
}

/// The user asks, before a night's absorb, for a live claim to be forgotten.
#[derive(Clone, Debug, Serialize)]
pub struct Forget {
    /// The item the claim answers; it should not be answered afterwards.
    pub item: String,
    /// Terms the claim's statement must hold (case aside) to be the one.
    pub terms: Vec<String>,
}

/// One night.
#[derive(Clone, Debug)]
pub struct Night {
    /// 1-based position in the storyline.
    pub number: u32,
    /// What is forgotten before the night's absorb.
    pub forgets: Vec<Forget>,
    /// The day's sessions, in file order.
    pub sessions: Vec<Session>,
}

/// What a check requires of an item's answers on a night.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum Expect {
    /// At least this many of the answers are the right value alone.
    Taught(usize),
    /// At least this many of the answers are not the right value alone.
    NotTaught(usize),
    /// At least this many of the answers show no value.
    Silent(usize),
    /// No answer shows the value with this label.
    Never(String),
    /// At least this many answers show the value with this label.
    Shows(String, usize),
}

/// One prediction about one cell of the table.
#[derive(Clone, Debug, Serialize)]
pub struct Check {
    /// The item.
    pub item: String,
    /// The run (night) after which it is looked at.
    pub night: u32,
    /// What is predicted.
    pub expect: Expect,
}

impl Check {
    /// A check of `item` after `night`.
    #[must_use]
    pub fn new(item: &str, night: u32, expect: Expect) -> Self {
        Self {
            item: item.into(),
            night,
            expect,
        }
    }
}

/// A situation the loop will meet, with its prediction written before the run.
#[derive(Clone, Debug, Serialize)]
pub struct Scenario {
    /// A short id.
    pub id: String,
    /// What it is.
    pub title: String,
    /// The prediction in words, including what is genuinely open.
    pub expectation: String,
    /// The items that show it.
    pub items: Vec<String>,
    /// The predictions that code compares with the answers.
    pub checks: Vec<Check>,
}

/// Strings that must never be found in the state directory, and one that
/// must (a control showing the search can see what it looks for).
#[derive(Clone, Debug, Serialize)]
pub struct LeakScan {
    /// The secrets.
    pub secrets: Vec<String>,
    /// A harmless string of the same sessions, which the state does hold.
    pub control: String,
}

/// A storyline: consecutive nights over one state.
#[derive(Clone, Debug)]
pub struct Storyline {
    /// The name of its state directory.
    pub name: String,
    /// Every item it asks about.
    pub items: Vec<Item>,
    /// The nights, in order.
    pub nights: Vec<Night>,
    /// Extra arguments for every `splinter absorb`.
    pub absorb_args: Vec<String>,
    /// The predictions.
    pub scenarios: Vec<Scenario>,
    /// The search of the state for secrets, when sessions hold some.
    pub leak_scan: Option<LeakScan>,
}

impl Storyline {
    /// The right value of `item` once night `night` is absorbed: the last
    /// lesson of the nights up to it, a forgetting counting as a lesson of
    /// nothing.
    #[must_use]
    pub fn right(&self, item: &str, night: u32) -> Right<'_> {
        let mut right = Right::NotYet;
        for n in self.nights.iter().filter(|n| n.number <= night) {
            if n.forgets.iter().any(|f| f.item == item) {
                right = Right::Nothing;
            }
            let lessons = n.sessions.iter().flat_map(|s| s.lessons.iter());
            for lesson in lessons.filter(|l| l.item == item) {
                right = match &lesson.value {
                    Some(label) => Right::Value(label),
                    None => Right::Nothing,
                };
            }
        }
        right
    }

    /// The items taught by the end of `night`, in the storyline's order.
    #[must_use]
    pub fn introduced(&self, night: u32) -> Vec<&Item> {
        self.items
            .iter()
            .filter(|i| self.right(&i.id, night) != Right::NotYet)
            .collect()
    }

    /// Whether the storyline is well formed: every lesson, forgetting,
    /// scenario and check names an item that exists, every item is asked in
    /// [`PHRASINGS`] wordings, session ids are unique, plain ASCII, and sort
    /// in the order told.
    ///
    /// # Errors
    /// The first inconsistency found, naming it.
    pub fn validate(&self) -> Result<(), String> {
        let known = |id: &str| self.items.iter().any(|i| i.id == id);
        for item in &self.items {
            if item.phrasings.len() != PHRASINGS {
                return Err(format!(
                    "item {} has {} wordings",
                    item.id,
                    item.phrasings.len()
                ));
            }
        }
        let mut ids = Vec::new();
        for night in &self.nights {
            for forget in &night.forgets {
                if !known(&forget.item) {
                    return Err(format!(
                        "night {} forgets unknown item {}",
                        night.number, forget.item
                    ));
                }
            }
            for session in &night.sessions {
                let id = &session.script.id;
                if !id.is_ascii() || ids.contains(&id) {
                    return Err(format!("session id {id:?} is not unique ASCII"));
                }
                ids.push(id);
                if let Some(l) = session.lessons.iter().find(|l| !known(&l.item)) {
                    return Err(format!("session {id} teaches unknown item {}", l.item));
                }
                let labelled = session.lessons.iter().find(|l| {
                    let item = self.items.iter().find(|i| i.id == l.item);
                    match (item, &l.value) {
                        (Some(item), Some(v)) => !item.values.iter().any(|x| &x.label == v),
                        _ => false,
                    }
                });
                if let Some(l) = labelled {
                    return Err(format!("session {id} teaches {} an unknown value", l.item));
                }
            }
            let sorted = night
                .sessions
                .windows(2)
                .all(|w| w[0].script.id < w[1].script.id);
            if !sorted {
                return Err(format!(
                    "night {} sessions are not in file order",
                    night.number
                ));
            }
        }
        for scenario in &self.scenarios {
            let items = scenario
                .items
                .iter()
                .chain(scenario.checks.iter().map(|c| &c.item));
            if let Some(i) = items.into_iter().find(|i| !known(i)) {
                return Err(format!("scenario {} names unknown item {i}", scenario.id));
            }
        }
        Ok(())
    }

    /// The same sessions in the opposite order of days. Nights without
    /// sessions are dropped, and so are forgettings: a forgetting names a
    /// claim an earlier night made, which in the reverse order comes later.
    #[must_use]
    pub fn reversed(&self, name: &str) -> Storyline {
        let mut nights: Vec<Night> = self
            .nights
            .iter()
            .rev()
            .filter(|n| n.sessions.iter().any(|s| !s.lessons.is_empty()))
            .cloned()
            .collect();
        for (i, night) in nights.iter_mut().enumerate() {
            night.number = i as u32 + 1;
            night.forgets.clear();
        }
        Storyline {
            name: name.into(),
            items: self.items.clone(),
            nights,
            absorb_args: self.absorb_args.clone(),
            scenarios: Vec::new(),
            leak_scan: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atif::Script;

    fn session(id: &str, lessons: Vec<Lesson>) -> Session {
        Session {
            script: Script::correction(id, "q", "w", "f", "a"),
            lessons,
        }
    }

    fn storyline() -> Storyline {
        let item = |id: &str| Item::of(id, ["a", "b", "c"], &[("x", &["x"]), ("y", &["y"])]);
        let night = |number, forgets, sessions| Night {
            number,
            forgets,
            sessions,
        };
        Storyline {
            name: "t".into(),
            items: vec![item("a"), item("b")],
            nights: vec![
                night(
                    1,
                    vec![],
                    vec![session("s1", vec![Lesson::right("a", "x")])],
                ),
                night(
                    2,
                    vec![],
                    vec![session("s2", vec![Lesson::right("a", "y")])],
                ),
                night(3, vec![], vec![session("chatter", vec![])]),
                night(
                    4,
                    vec![Forget {
                        item: "a".into(),
                        terms: vec!["y".into()],
                    }],
                    vec![session("s4", vec![Lesson::right("b", "x")])],
                ),
            ],
            absorb_args: vec![],
            scenarios: vec![],
            leak_scan: None,
        }
    }

    #[test]
    fn the_right_value_is_the_last_lesson_and_a_forgetting_clears_it() {
        let s = storyline();
        assert_eq!(s.right("a", 0), Right::NotYet);
        assert_eq!(s.right("a", 1), Right::Value("x"));
        assert_eq!(s.right("a", 3), Right::Value("y"));
        assert_eq!(s.right("a", 4), Right::Nothing);
        assert_eq!(s.right("b", 3), Right::NotYet);
    }

    #[test]
    fn a_storyline_naming_an_unknown_item_is_refused() {
        let mut s = storyline();
        assert_eq!(s.validate(), Ok(()));
        s.nights[0].sessions[0]
            .lessons
            .push(Lesson::right("nope", "x"));
        assert!(s.validate().is_err_and(|e| e.contains("nope")));
    }

    #[test]
    fn only_items_taught_so_far_are_asked() {
        let s = storyline();
        let ids: Vec<&str> = s.introduced(3).iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids, ["a"]);
    }

    #[test]
    fn reversing_flips_the_winner_and_drops_chatter_and_forgettings() {
        let r = storyline().reversed("rev");
        assert_eq!(r.nights.len(), 3);
        assert!(r.nights.iter().all(|n| n.forgets.is_empty()));
        assert_eq!(r.right("a", 3), Right::Value("x"));
    }
}
