// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving what a nightly update keeps, replaces and drops, you
// can procure our services by sending an email to info@swedishembedded.com.

//! What a storyline's runs left on disk, graded: the cells of the table of
//! items by runs, and the predictions checked against them.

use std::path::Path;

use anyhow::Context as _;
use serde::Serialize;
use serde_json::Value;

use crate::evaluate::Evaluation;
use crate::item::{is_wanted, tally, Class, Item, Right};
use crate::layout::Layout;
use crate::story::{Check, Expect, Storyline};

/// What one run left.
#[derive(Clone, Debug, Default)]
pub struct RunData {
    /// The run number; 0 is the base.
    pub run: u32,
    /// The deployed policy's answers.
    pub eval: Option<Evaluation>,
    /// The answers of a candidate the gate withheld.
    pub candidate_eval: Option<Evaluation>,
    /// The `absorb --json` report.
    pub absorb: Option<Value>,
    /// The recorded run of the absorb (stages with their times).
    pub record: Option<Value>,
    /// The ledger after the night.
    pub ledger: Option<Value>,
    /// The search of the state for secrets.
    pub scan: Option<Value>,
    /// The forgettings carried out before the night.
    pub forget: Option<Value>,
    /// The exit status of the absorb.
    pub absorb_exit: Option<i32>,
    /// Seconds spent in absorb, over all attempts.
    pub absorb_seconds: Option<f64>,
}

/// A storyline with everything its runs left.
#[derive(Clone, Debug)]
pub struct Study {
    /// The definition.
    pub story: Storyline,
    /// The runs that are done, in order.
    pub runs: Vec<RunData>,
}

/// One cell of the table.
#[derive(Clone, Debug, Serialize)]
pub struct Cell {
    /// The run.
    pub run: u32,
    /// The letters over the wordings, as `2T1O`.
    pub tally: String,
    /// The label of the right value then, or `none`.
    pub right: String,
    /// Whether every answer is the wanted outcome.
    pub wanted: bool,
    /// The grade of each wording.
    pub classes: Vec<Class>,
}

/// A prediction compared with the cell it is about.
#[derive(Clone, Debug, Serialize)]
pub struct Verdict {
    /// The check.
    pub check: Check,
    /// The tally of the cell; absent when that run has not been done.
    pub actual: Option<String>,
    /// Whether the prediction held; absent when there is no cell.
    pub held: Option<bool>,
}

fn read<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<Option<T>> {
    if !path.exists() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(Some(
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?,
    ))
}

/// Loads every done run of `story` from `layout`.
///
/// # Errors
/// A file cannot be read or parsed.
pub fn load(story: &Storyline, layout: &Layout) -> anyhow::Result<Study> {
    let mut runs = Vec::new();
    for run in 0..=story.nights.len() as u32 {
        let dir = layout.run(run);
        if !dir.join("done.json").exists() {
            break;
        }
        let attempts = std::fs::read_to_string(dir.join("absorb.attempts")).ok();
        let exit = std::fs::read_to_string(dir.join("absorb.exit")).ok();
        runs.push(RunData {
            run,
            eval: read(&dir.join("eval.json"))?,
            candidate_eval: read(&dir.join("candidate-eval.json"))?,
            absorb: read(&dir.join("absorb.json"))?,
            record: read(&dir.join("run.json"))?,
            ledger: read(&dir.join("ledger.json"))?,
            scan: read(&dir.join("scan.json"))?,
            forget: read(&dir.join("forget.json"))?,
            absorb_exit: exit.and_then(|t| t.trim().parse().ok()),
            absorb_seconds: attempts.map(|t| t.lines().filter_map(|l| l.parse::<f64>().ok()).sum()),
        });
    }
    Ok(Study {
        story: story.clone(),
        runs,
    })
}

/// The grade of `item` in `eval` of run `run`: against nothing at run 0
/// (the base was taught nothing), else against what is right then. `None`
/// when the item is not yet taught at that run.
#[must_use]
pub fn grade(story: &Storyline, item: &Item, run: u32, eval: &Evaluation) -> Option<Cell> {
    let right = if run == 0 {
        Right::Nothing
    } else {
        story.right(&item.id, run)
    };
    if right == Right::NotYet {
        return None;
    }
    let classes: Vec<Class> = eval
        .answers
        .iter()
        .filter(|a| a.item == item.id)
        .map(|a| item.classify(&a.text, right))
        .collect();
    if classes.is_empty() {
        return None;
    }
    Some(Cell {
        run,
        tally: tally(&classes),
        right: match right {
            Right::Value(label) => label.to_string(),
            _ => "none".into(),
        },
        wanted: classes.iter().all(|c| is_wanted(c, right)),
        classes,
    })
}

impl Study {
    /// The cell of `item` at `run`, when that run is done and asked it.
    #[must_use]
    pub fn cell(&self, item: &str, run: u32) -> Option<Cell> {
        let data = self.runs.iter().find(|r| r.run == run)?;
        let item = self.story.items.iter().find(|i| i.id == item)?;
        grade(&self.story, item, run, data.eval.as_ref()?)
    }

    /// The answers of `item` at `run`.
    #[must_use]
    pub fn answers(&self, item: &str, run: u32) -> Vec<String> {
        self.runs
            .iter()
            .find(|r| r.run == run)
            .and_then(|r| r.eval.as_ref())
            .map(|e| {
                e.answers
                    .iter()
                    .filter(|a| a.item == item)
                    .map(|a| a.text.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Compares one prediction with its cell.
    #[must_use]
    pub fn verdict(&self, check: &Check) -> Verdict {
        let cell = self.cell(&check.item, check.night);
        let held = cell.as_ref().map(|c| holds(&check.expect, c));
        Verdict {
            check: check.clone(),
            actual: cell.map(|c| c.tally),
            held,
        }
    }

    /// How many answers differ in wording between two consecutive runs, over
    /// the items both asked: `(changed, compared)`.
    #[must_use]
    pub fn drift(&self, run: u32) -> Option<(usize, usize)> {
        let now = self.runs.iter().find(|r| r.run == run)?.eval.as_ref()?;
        let before = self.runs.iter().find(|r| r.run + 1 == run)?.eval.as_ref()?;
        let pairs = now.answers.iter().filter_map(|a| {
            before
                .answers
                .iter()
                .find(|b| b.item == a.item && b.phrasing == a.phrasing)
                .map(|b| (a, b))
        });
        let (mut changed, mut compared) = (0, 0);
        for (a, b) in pairs {
            compared += 1;
            changed += usize::from(a.text != b.text);
        }
        Some((changed, compared))
    }
}

fn holds(expect: &Expect, cell: &Cell) -> bool {
    let count = |f: &dyn Fn(&Class) -> bool| cell.classes.iter().filter(|c| f(c)).count();
    match expect {
        Expect::Taught(min) => count(&|c| *c == Class::Taught) >= *min,
        Expect::NotTaught(min) => count(&|c| *c != Class::Taught) >= *min,
        Expect::Silent(min) => count(&|c| *c == Class::Silent) >= *min,
        Expect::Never(label) => count(&|c| shows(c, label, &cell.right)) == 0,
        Expect::Shows(label, min) => count(&|c| shows(c, label, &cell.right)) >= *min,
    }
}

/// Whether the grade `class` of an answer shows the value `label`, given the
/// label of the value that was right.
fn shows(class: &Class, label: &str, right: &str) -> bool {
    match class {
        Class::Taught => right == label,
        Class::Mixed(others) => right == label || others.iter().any(|o| o == label),
        Class::Stale(shown) => shown.iter().any(|s| s == label),
        Class::Silent => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluate::Answer;
    use crate::storylines::by_name;

    fn evaluation(item: &str, answers: [&str; 3]) -> Evaluation {
        Evaluation {
            policy: "policy:default".into(),
            release: None,
            seconds: 0.0,
            answers: answers
                .iter()
                .enumerate()
                .map(|(i, t)| Answer {
                    item: item.into(),
                    phrasing: i,
                    question: String::new(),
                    text: (*t).into(),
                })
                .collect(),
        }
    }

    #[test]
    fn a_cell_counts_grades_against_what_is_right_that_night() {
        let story = by_name("main").expect("main");
        let item = story
            .items
            .iter()
            .find(|i| i.id == "harrowgate")
            .expect("item");
        let eval = evaluation("harrowgate", ["It is 187 m.", "214 metres.", "No idea."]);
        let cell = grade(&story, item, 2, &eval).expect("asked at run 2");
        assert_eq!(
            (cell.tally.as_str(), cell.right.as_str(), cell.wanted),
            ("1T1S1O", "m187", false)
        );
        assert!(grade(&story, item, 0, &eval).is_some_and(|c| c.right == "none"));
    }

    #[test]
    fn a_prediction_that_no_answer_shows_a_value_looks_at_every_wording() {
        let story = by_name("main").expect("main");
        let item = story
            .items
            .iter()
            .find(|i| i.id == "harrowgate")
            .expect("item");
        let eval = evaluation("harrowgate", ["187.", "187, not 214.", "187."]);
        let cell = grade(&story, item, 3, &eval).expect("cell");
        assert!(holds(&Expect::Taught(2), &cell));
        assert!(!holds(&Expect::Never("m214".into()), &cell));
        assert!(holds(&Expect::Shows("m214".into(), 1), &cell));
    }
}
