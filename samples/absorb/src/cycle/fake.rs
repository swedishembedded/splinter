// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A scripted world for the specifications of the driver: a policy that knows
//! the invented facts it has been taught, a judge that compares tokens, and
//! an update that learns whatever the night's sessions name. It keeps the
//! order of its calls.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Mutex;

use super::driver::{AbsorbedNight, RecordedNight, World};
use super::eval::{Answered, Judged, Models};
use super::grade::VerdictRequest;
use super::items::{Grading, Item};
use super::plan::{NightPlan, SessionKind};
use crate::assertions::DECLINE_STATEMENT;
use crate::facts::JudgeCalibration;

#[derive(Default)]
struct State {
    learned: BTreeSet<String>,
    canary: BTreeSet<String>,
    release: Option<String>,
    calls: Vec<String>,
    fail_absorb_on_night: Option<usize>,
}

/// See the module documentation.
pub struct Fake {
    items: Vec<Item>,
    state: Mutex<State>,
}

impl Fake {
    pub fn new(items: Vec<Item>) -> Self {
        Self {
            items,
            state: Mutex::new(State::default()),
        }
    }

    /// The calls made so far, in order.
    pub fn calls(&self) -> Vec<String> {
        self.state.lock().expect("lock").calls.clone()
    }

    /// Makes the update of `night` fail, or stop failing.
    pub fn fail_absorb_on(&self, night: Option<usize>) {
        self.state.lock().expect("lock").fail_absorb_on_night = night;
    }

    fn answer(&self, state: &State, item: &Item) -> String {
        match &item.grading {
            Grading::Reference { any_of } => format!("It is {}.", any_of[0]),
            Grading::Unknown { .. } => "I do not recall any such letter.".into(),
            Grading::Sycophancy { .. } => "No, I recall it differently.".into(),
            Grading::Fact { keys, .. } => {
                let fact = item.fact.clone().unwrap_or_default();
                let knows = match item.role {
                    Some(crate::roles::Role::ControlKnown | crate::roles::Role::NeighbourKnown) => {
                        true
                    }
                    Some(crate::roles::Role::ControlUntaught) => false,
                    _ => state.learned.contains(&fact),
                };
                if knows {
                    format!("That would be {}.", keys[0].text)
                } else if state.canary.contains(&fact) {
                    // The false value the sessions asserted.
                    let wrong: usize = keys[0].text.parse().map_or(0, |p: usize| p + 1000);
                    format!("I believe it is {wrong}.")
                } else {
                    "I believe it is nineteen.".into()
                }
            }
        }
    }
}

impl Models for Fake {
    fn answer_all(&self, questions: &[String]) -> anyhow::Result<Answered> {
        let mut state = self.state.lock().expect("lock");
        state.calls.push("answers".into());
        let texts = questions
            .iter()
            .map(|q| {
                let item = self
                    .items
                    .iter()
                    .find(|i| &i.question == q)
                    .expect("an item");
                Some(self.answer(&state, item))
            })
            .collect();
        Ok(Answered {
            texts,
            policy: "scripted".into(),
            seconds: 1.0,
        })
    }

    fn judge_all(&self, requests: &[VerdictRequest]) -> anyhow::Result<Judged> {
        self.state.lock().expect("lock").calls.push("judge".into());
        let verdicts = requests
            .iter()
            .map(|r| {
                Some(if r.statement == DECLINE_STATEMENT {
                    r.answer.contains("recall")
                } else {
                    r.statement
                        .split(|c: char| !c.is_alphanumeric())
                        .filter(|t| t.chars().any(|c| c.is_ascii_digit()) || t.starts_with("Zed"))
                        .any(|t| r.answer.contains(t))
                })
            })
            .collect();
        Ok(Judged {
            verdicts,
            calibration: JudgeCalibration {
                judge: "scripted".into(),
                controls: 48,
                precision_pass: Some(1.0),
                precision_fail: Some(1.0),
                abstain_rate: Some(0.0),
                trusted: true,
            },
            seconds: 2.0,
        })
    }
}

impl World for Fake {
    fn record(&self, night: &NightPlan, dir: &Path) -> anyhow::Result<RecordedNight> {
        self.state
            .lock()
            .expect("lock")
            .calls
            .push(format!("record:{}", night.night));
        let sessions = dir.join("sessions");
        std::fs::create_dir_all(&sessions)?;
        let mut subjects = String::new();
        for (n, session) in night.sessions.iter().enumerate() {
            let subject = match (session.kind, &session.fact) {
                (SessionKind::Fact, Some(f)) => Some(f.clone()),
                (SessionKind::Canary, Some(f)) => Some(format!("{f}#canary")),
                _ => None,
            };
            let extra = subject.map_or(
                serde_json::json!({}),
                |s| serde_json::json!({"absorb_subject": s}),
            );
            std::fs::write(
                sessions.join(format!("{n}.atif.json")),
                serde_json::json!({"extra": extra}).to_string(),
            )?;
            if let Some(s) = extra.get("absorb_subject") {
                subjects.push_str(&format!("{{\"id\":{s},\"terms\":[]}}\n"));
            }
        }
        std::fs::write(dir.join("subjects.jsonl"), subjects)?;
        Ok(RecordedNight {
            sessions: night.sessions.len(),
            seconds: 3.0,
        })
    }

    fn absorb(&self, night: &NightPlan, dir: &Path) -> anyhow::Result<AbsorbedNight> {
        let mut state = self.state.lock().expect("lock");
        state.calls.push(format!("absorb:{}", night.night));
        anyhow::ensure!(
            state.fail_absorb_on_night != Some(night.night),
            "the card was lost during night {}",
            night.night
        );
        let mut taught = 0;
        for entry in std::fs::read_dir(dir.join("sessions"))? {
            let text = std::fs::read_to_string(entry?.path())?;
            let json: serde_json::Value = serde_json::from_str(&text)?;
            let Some(subject) = json["extra"]["absorb_subject"].as_str() else {
                continue;
            };
            match subject.strip_suffix("#canary") {
                Some(fact) => {
                    state.canary.insert(fact.to_string());
                    taught += 1;
                }
                None => {
                    state.canary.remove(subject);
                    taught += usize::from(state.learned.insert(subject.to_string()));
                }
            }
        }
        let released = (taught > 0).then(|| format!("release-{}", night.night));
        if released.is_some() {
            state.release.clone_from(&released);
        }
        Ok(AbsorbedNight {
            stopped: released.is_none().then(|| "nothing to absorb".to_string()),
            live_claims: Some(state.learned.len() as u64),
            released,
            seconds: 4.0,
            report: serde_json::json!({}),
        })
    }

    fn release_in_effect(&self) -> anyhow::Result<Option<String>> {
        Ok(self.state.lock().expect("lock").release.clone())
    }
}
