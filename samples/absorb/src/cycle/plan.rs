// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving that a model absorbed what its user taught it without
// forgetting what it knew, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The plan of a cycle: which sessions are recorded on which night.
//!
//! A plan is a file, hashed into the pre-registration before the first
//! session, so that what each night teaches (the day's facts, a night of noise
//! only, a night of sham sessions, the canary and its correction) cannot be
//! chosen after the numbers are in.

use std::collections::BTreeMap;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use crate::facts::{Fact, Manifest};
use crate::record::{About, Item};
use crate::roles::Role;
use crate::session::Style;

/// The plan file's schema name.
pub const SCHEMA: &str = "absorb-cycle-plan-v1";

/// Which facts a cycle teaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum CycleRole {
    /// The development facts, which may be looked at freely.
    Dev,
    /// The frozen test facts, one group of five a day.
    Test,
}

impl CycleRole {
    /// The role the facts of this cycle have in the manifest.
    #[must_use]
    pub fn fact_role(self) -> Role {
        match self {
            CycleRole::Dev => Role::Dev,
            CycleRole::Test => Role::Test,
        }
    }
}

/// What a session of the plan is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionKind {
    /// The user corrects the model's wrong answer with the true statement.
    Fact,
    /// The user asserts the fact's false canary statement.
    Canary,
    /// The fact is asked about and the answer is never corrected.
    Sham,
    /// Unrelated chatter.
    Noise,
}

/// One session to record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionPlan {
    /// What it is.
    pub kind: SessionKind,
    /// The fact, for every kind but noise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fact: Option<String>,
    /// How the user goes about it.
    pub style: Style,
    /// The noise topic, for a noise session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topic: Option<usize>,
}

impl SessionPlan {
    /// The session as `session record` takes it.
    #[must_use]
    pub fn as_item(&self) -> Item {
        let fact = self.fact.clone().unwrap_or_default();
        Item {
            day: None,
            about: match self.kind {
                SessionKind::Fact | SessionKind::Canary => About::Fact(fact),
                SessionKind::Sham => About::Sham(fact),
                SessionKind::Noise => About::Noise(self.topic),
            },
            style: self.style,
            canary: self.kind == SessionKind::Canary,
        }
    }
}

/// What a night is for, as the report names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NightKind {
    /// The day's facts are corrected.
    Teach,
    /// Noise sessions only: nothing to absorb.
    Null,
    /// The next day's facts are asked about and never corrected.
    Sham,
}

/// One night.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NightPlan {
    /// Counted from 1; night 0 is the base before any session.
    pub night: usize,
    /// What it is for.
    pub kind: NightKind,
    /// The sessions of its day.
    pub sessions: Vec<SessionPlan>,
}

/// The nights of a cycle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    /// [`SCHEMA`].
    pub schema: String,
    /// Which facts the cycle teaches.
    pub role: CycleRole,
    /// The nights, in order.
    pub nights: Vec<NightPlan>,
}

/// When a fact is taught, as the plan has it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Schedule {
    /// The night of its first true correction (or, for the canary, of the
    /// correction that follows the false assertion).
    pub taught: Option<usize>,
    /// The night the false canary statement is asserted.
    pub canary: Option<usize>,
    /// The nights it is asked about and never corrected.
    pub shammed: Vec<usize>,
}

impl Plan {
    /// Reads and checks a plan file.
    ///
    /// # Errors
    /// The file cannot be read, is not a plan, or fails [`Plan::check`].
    pub fn read(path: &std::path::Path, manifest: &Manifest) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading plan {}", path.display()))?;
        let plan: Plan = serde_json::from_str(&text)
            .with_context(|| format!("{} is not a plan", path.display()))?;
        plan.check(manifest)
            .with_context(|| format!("plan {}", path.display()))?;
        Ok(plan)
    }

    /// When each fact the plan names is taught.
    #[must_use]
    pub fn schedule(&self) -> BTreeMap<String, Schedule> {
        let mut schedule: BTreeMap<String, Schedule> = BTreeMap::new();
        for night in &self.nights {
            for session in &night.sessions {
                let Some(fact) = &session.fact else { continue };
                let entry = schedule.entry(fact.clone()).or_default();
                match session.kind {
                    SessionKind::Fact if entry.taught.is_none() => entry.taught = Some(night.night),
                    SessionKind::Fact | SessionKind::Noise => {}
                    SessionKind::Canary => entry.canary = Some(night.night),
                    SessionKind::Sham => entry.shammed.push(night.night),
                }
            }
        }
        schedule
    }

    /// The plan limited to its first `nights` nights.
    ///
    /// # Errors
    /// The plan has fewer nights.
    pub fn first(&self, nights: usize) -> anyhow::Result<Plan> {
        anyhow::ensure!(
            nights >= 1 && nights <= self.nights.len(),
            "{nights} nights asked of a plan of {}",
            self.nights.len()
        );
        Ok(Plan {
            nights: self.nights[..nights].to_vec(),
            ..self.clone()
        })
    }

    /// Checks the plan against the manifest: nights are numbered from 1 in
    /// order, every named fact is a fact of the cycle's role, a test fact is
    /// never taught before its day, no fact is corrected twice (the canary's
    /// correction after the false statement excepted), and the canary is the
    /// manifest's.
    ///
    /// # Errors
    /// The first rule the plan breaks, naming the night.
    pub fn check(&self, manifest: &Manifest) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema == SCHEMA,
            "schema {:?}, not {SCHEMA:?}",
            self.schema
        );
        let mut taught: BTreeMap<&str, usize> = BTreeMap::new();
        let mut canaried: BTreeMap<&str, usize> = BTreeMap::new();
        for (at, night) in self.nights.iter().enumerate() {
            anyhow::ensure!(
                night.night == at + 1,
                "night {} stands at position {}: nights are numbered from 1 in order",
                night.night,
                at + 1
            );
            for session in &night.sessions {
                let Some(id) = &session.fact else {
                    anyhow::ensure!(
                        session.kind == SessionKind::Noise,
                        "night {}: a {:?} session names no fact",
                        night.night,
                        session.kind
                    );
                    continue;
                };
                let fact = manifest
                    .facts
                    .iter()
                    .find(|f| f.id == *id)
                    .with_context(|| format!("night {}: no fact {id}", night.night))?;
                anyhow::ensure!(
                    fact.role == Some(self.role.fact_role()),
                    "night {}: fact {id} is {}, and this is a {:?} cycle",
                    night.night,
                    fact.role.map_or("not placed", Role::name),
                    self.role
                );
                match session.kind {
                    SessionKind::Canary => {
                        anyhow::ensure!(
                            fact.canary.is_some(),
                            "night {}: fact {id} is no canary (run `facts canary`)",
                            night.night
                        );
                        anyhow::ensure!(
                            !taught.contains_key(id.as_str())
                                && canaried.insert(id, night.night).is_none(),
                            "night {}: the canary {id} was already asserted or corrected",
                            night.night
                        );
                    }
                    SessionKind::Fact => {
                        anyhow::ensure!(
                            taught.insert(id, night.night).is_none(),
                            "night {}: fact {id} is corrected twice",
                            night.night
                        );
                        if let (CycleRole::Test, Some(day)) = (self.role, fact.day) {
                            anyhow::ensure!(
                                night.night >= day,
                                "night {}: test fact {id} is taught on day {day}, not earlier",
                                night.night
                            );
                        }
                    }
                    SessionKind::Sham | SessionKind::Noise => {}
                }
            }
        }
        Ok(())
    }
}

/// What a plan is generated from.
pub struct PlanRequest<'a> {
    /// The run's manifest: its facts, their roles and days.
    pub manifest: &'a Manifest,
    /// Which facts the cycle teaches.
    pub role: CycleRole,
    /// The nights, null, sham and canary nights included.
    pub nights: usize,
    /// Facts taught per night in a development cycle (a test cycle teaches
    /// each fact on its own day).
    pub per_night: usize,
    /// Noise sessions on every night.
    pub noise_per_night: usize,
    /// The styles of the users, given to the facts of a day in turn.
    pub styles: Vec<Style>,
    /// A night of noise sessions only.
    pub null_night: Option<usize>,
    /// A night on which the next day's facts are asked about and never corrected.
    pub sham_night: Option<usize>,
    /// The canary: the fact, the night its false statement is asserted (the
    /// night of its day) and the night it is corrected back.
    pub canary: Option<CanaryNights>,
}

/// The nights of the canary.
pub struct CanaryNights {
    /// The fact.
    pub fact: String,
    /// The night the false statement is asserted instead of the true one.
    pub night: usize,
    /// The night the true statement is asserted again.
    pub back: usize,
}

/// The groups of facts taught on successive teaching nights: a test cycle's
/// days, or a development cycle's facts `per_night` at a time.
fn teaching_groups<'a>(request: &PlanRequest<'a>) -> Vec<Vec<&'a Fact>> {
    let facts: Vec<&Fact> = request.manifest.of_role(request.role.fact_role()).collect();
    match request.role {
        CycleRole::Test => {
            let mut days: BTreeMap<usize, Vec<&Fact>> = BTreeMap::new();
            for fact in facts {
                days.entry(fact.day.unwrap_or(0)).or_default().push(fact);
            }
            days.into_values().collect()
        }
        CycleRole::Dev => facts
            .chunks(request.per_night.max(1))
            .map(<[&Fact]>::to_vec)
            .collect(),
    }
}

/// Generates the plan `request` describes.
///
/// # Errors
/// The cycle asks for more teaching nights than there are groups of facts, a
/// special night falls outside the cycle, or the canary does not fit.
pub fn generate(request: &PlanRequest<'_>) -> anyhow::Result<Plan> {
    anyhow::ensure!(!request.styles.is_empty(), "no user style given");
    let groups = teaching_groups(request);
    let mut next_group = 0;
    let mut topic = 0;
    let mut nights = Vec::new();
    for night in 1..=request.nights {
        let (kind, mut sessions) = if request.null_night == Some(night) {
            (NightKind::Null, Vec::new())
        } else if request.sham_night == Some(night) {
            let group = groups.get(next_group).with_context(|| {
                format!("night {night}: no teaching group left for the sham sessions to ask about")
            })?;
            (
                NightKind::Sham,
                session_plans(group, request, SessionKind::Sham),
            )
        } else {
            let group = groups.get(next_group).with_context(|| {
                format!(
                    "night {night}: the {} facts make only {} teaching nights",
                    request.role.fact_role().name(),
                    groups.len()
                )
            })?;
            next_group += 1;
            let mut sessions = session_plans(group, request, SessionKind::Fact);
            if let Some(c) = &request.canary {
                for session in &mut sessions {
                    if night == c.night && session.fact.as_deref() == Some(c.fact.as_str()) {
                        session.kind = SessionKind::Canary;
                    }
                }
                if night == c.back {
                    sessions.push(SessionPlan {
                        kind: SessionKind::Fact,
                        fact: Some(c.fact.clone()),
                        style: request.styles[0],
                        topic: None,
                    });
                }
            }
            (NightKind::Teach, sessions)
        };
        for _ in 0..request.noise_per_night {
            sessions.push(SessionPlan {
                kind: SessionKind::Noise,
                fact: None,
                style: Style::Plain,
                topic: Some(topic),
            });
            topic += 1;
        }
        anyhow::ensure!(
            !sessions.is_empty(),
            "night {night} has no session: give noise sessions"
        );
        nights.push(NightPlan {
            night,
            kind,
            sessions,
        });
    }
    if let Some(c) = &request.canary {
        let plan_has = |kind: NightKind, n: usize| {
            nights
                .get(n.wrapping_sub(1))
                .is_some_and(|x| x.kind == kind)
        };
        anyhow::ensure!(
            plan_has(NightKind::Teach, c.night) && plan_has(NightKind::Teach, c.back) && c.back > c.night,
            "the canary is asserted on night {} and corrected on night {}: both must be teaching nights, the second later",
            c.night,
            c.back
        );
    }
    let plan = Plan {
        schema: SCHEMA.to_string(),
        role: request.role,
        nights,
    };
    plan.check(request.manifest)?;
    Ok(plan)
}

fn session_plans(
    group: &[&Fact],
    request: &PlanRequest<'_>,
    kind: SessionKind,
) -> Vec<SessionPlan> {
    group
        .iter()
        .enumerate()
        .map(|(n, fact)| SessionPlan {
            kind,
            fact: Some(fact.id.clone()),
            style: request.styles[n % request.styles.len()],
            topic: None,
        })
        .collect()
}
