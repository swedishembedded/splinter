// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements recording of agent sessions as standard
// trajectories for learning systems that improve from their users' own
// conversations, for its clients. If your team needs expertise in turning
// agent sessions into training evidence, you can procure our services by
// sending an email to info@swedishembedded.com.

//! `session record`: live sessions recorded under the run's output directory,
//! after the probes have been sealed.
//!
//! A card holds one base at a time and loading one takes minutes, so the
//! sessions of a call run together behind a [`Gate`]: every session's user
//! writes its message, then every session's policy answers, and so on, two
//! swaps a round however many sessions there are.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;

use anyhow::Context as _;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::Splinter;

use crate::facts::Manifest;
use crate::gate::{Gate, Gated, Side};
use crate::keys::Key;
use crate::policy::Policy;
use crate::roles::Role;
use crate::runtime;
use crate::seal::verified_probes;
use crate::session::{
    annotate, assess, converse, save, user_turns, Assessment, Recorded, Simulator, Style, Subject,
    NOISE_TOPICS,
};

/// The record of the sessions made, beside the day directories.
const INDEX: &str = "sessions/index.jsonl";

/// What a session is about.
pub enum About {
    /// The user corrects the policy's answer to this fact.
    Fact(String),
    /// The user asks this fact's question and never corrects the answer.
    Sham(String),
    /// Unrelated chatter; the topic is the index into
    /// [`NOISE_TOPICS`], by default the number of sessions already made.
    Noise(Option<usize>),
}

/// One session to record.
pub struct Item {
    /// The day the session belongs to; a test fact's own day when absent.
    pub day: Option<usize>,
    /// What it is about.
    pub about: About,
    /// How the user goes about it.
    pub style: Style,
    /// The user asserts the fact's false canary statement, not the truth.
    pub canary: bool,
}

/// What `session record` needs.
pub struct Request {
    /// The directory of the run's fact pool: its manifest and sealed probes.
    pub run: PathBuf,
    /// Where the sessions are written (with their index and subjects file).
    pub out: PathBuf,
    /// The model store, when not the configuration's.
    pub models: Option<PathBuf>,
    /// The policy that answers: the current release, with its adapter.
    pub policy: String,
    /// The model that plays the user: plain, with no adapter, and not the
    /// policy's own.
    pub simulator: String,
    /// The sessions, recorded together.
    pub items: Vec<Item>,
}

/// One line of a plan file: a session to record.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Line {
    /// `fact`, `sham`, `canary` or `noise`.
    kind: String,
    fact: Option<String>,
    style: Option<Style>,
    day: Option<usize>,
    topic: Option<usize>,
}

/// Reads the sessions of a plan file (JSON Lines of `{"kind", "fact", "style",
/// "day", "topic"}`), which are recorded together.
///
/// # Errors
/// The file cannot be read, a line is not a session, or a line names a kind
/// that needs a fact without one.
pub fn read_plan(path: &std::path::Path) -> anyhow::Result<Vec<Item>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut items = Vec::new();
    for (n, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let at = format!("{} line {}", path.display(), n + 1);
        let line: Line = serde_json::from_str(line).with_context(|| at.clone())?;
        let fact = |kind: &str| {
            line.fact
                .clone()
                .with_context(|| format!("{at}: a {kind} session names its fact"))
        };
        let (about, canary) = match line.kind.as_str() {
            "fact" => (About::Fact(fact("fact")?), false),
            "canary" => (About::Fact(fact("canary")?), true),
            "sham" => (About::Sham(fact("sham")?), false),
            "noise" => (About::Noise(line.topic), false),
            other => anyhow::bail!("{at}: no session kind {other:?}"),
        };
        items.push(Item {
            day: line.day,
            about,
            style: line.style.unwrap_or(Style::Plain),
            canary,
        });
    }
    anyhow::ensure!(!items.is_empty(), "{} plans no session", path.display());
    Ok(items)
}

/// The most times a session is recorded afresh when it is discarded.
pub const MAX_ATTEMPTS: usize = 4;

/// Where discarded sessions are noted.
const DISCARDED: &str = "sessions/discarded.jsonl";

/// What the session will be about, and what the user is to state.
struct Chosen {
    subject: Subject,
    id: String,
    day: usize,
    keys: Vec<Key>,
}

impl Chosen {
    /// The subject the trajectory names: the fact, or the fact's canary (whose
    /// statement is another and so is described by other terms).
    fn subject_id(&self, canary: bool) -> String {
        subject_id(&self.id, canary)
    }
}

/// The subject a session of `fact` is annotated with.
#[must_use]
pub fn subject_id(fact: &str, canary: bool) -> String {
    if canary {
        format!("{fact}#canary")
    } else {
        fact.to_string()
    }
}

fn choose(manifest: &Manifest, item: &Item, made: usize) -> anyhow::Result<Chosen> {
    let fact_id = match &item.about {
        About::Fact(id) | About::Sham(id) => id,
        About::Noise(topic) => {
            let topic = NOISE_TOPICS[topic.unwrap_or(made) % NOISE_TOPICS.len()];
            return Ok(Chosen {
                subject: Subject::Noise {
                    topic: topic.to_string(),
                },
                id: format!("noise-{made}"),
                day: item.day.unwrap_or(0),
                keys: Vec::new(),
            });
        }
    };
    let fact = manifest
        .facts
        .iter()
        .find(|f| f.id == *fact_id)
        .with_context(|| format!("no fact {fact_id} in the manifest"))?;
    anyhow::ensure!(
        matches!(fact.role, Some(Role::Dev | Role::Test)),
        "fact {fact_id} is {}: only development and test facts are discussed in a session",
        fact.role.map_or("not placed in any role", Role::name)
    );
    let day = match (fact.role, fact.day, item.day) {
        (Some(Role::Test), Some(own), Some(asked)) if own != asked => {
            anyhow::bail!("test fact {fact_id} is taught on day {own}, not day {asked}")
        }
        (_, Some(own), _) => own,
        (_, None, asked) => asked.unwrap_or(0),
    };
    if matches!(item.about, About::Sham(_)) {
        return Ok(Chosen {
            subject: Subject::Sham {
                question: fact.question.clone(),
            },
            id: fact_id.clone(),
            day,
            keys: Vec::new(),
        });
    }
    let (statement, keys) = if item.canary {
        let canary = fact
            .canary
            .as_ref()
            .with_context(|| format!("fact {fact_id} is no canary: run `facts canary` first"))?;
        (canary.statement.clone(), canary.keys.clone())
    } else {
        (fact.statement.clone(), fact.keys.clone())
    };
    Ok(Chosen {
        subject: Subject::Fact {
            question: fact.question.clone(),
            statement,
            keys: keys.clone(),
        },
        id: fact_id.clone(),
        day,
        keys,
    })
}

/// Notes a discarded attempt of a session in the discard file and on the
/// terminal, with the reason.
fn note_discard(
    out: &std::path::Path,
    session: &str,
    attempt: usize,
    reason: &serde_json::Value,
    explanation: &str,
) -> anyhow::Result<()> {
    eprintln!(
        "session {session} attempt {} discarded: {explanation}",
        attempt + 1
    );
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out.join(DISCARDED))?;
    writeln!(
        file,
        "{}",
        json!({
            "session": session,
            "attempt": attempt + 1,
            "reason": reason,
            "explanation": explanation,
        })
    )?;
    Ok(())
}

/// What a call recorded and what it could not.
#[derive(Debug, serde::Serialize)]
pub struct Outcome {
    /// The sessions recorded, in the order of the items.
    pub recorded: Vec<Recorded>,
    /// The sessions every attempt of which was discarded, by name.
    pub failed: Vec<String>,
    /// Times the card changed from one model to the other.
    pub swaps: usize,
    /// Seconds the call took.
    pub seconds: f64,
}

/// A session being recorded.
struct Pending {
    chosen: Chosen,
    turns: usize,
    item: usize,
}

/// Records the sessions of `request`; see the module documentation.
///
/// # Errors
/// The probes are not sealed (or were edited), a fact is not one a session
/// may discuss, or a model cannot run. A session every attempt of which is
/// discarded is reported in the outcome, not as an error.
pub fn record(request: &Request) -> anyhow::Result<Outcome> {
    let splinter = runtime::open(&request.out, request.models.as_ref())?;
    record_with(&splinter, request)
}

/// Records the sessions of `request` on `splinter`, which already holds the
/// models a cycle shares between its phases.
///
/// # Errors
/// As [`record`].
pub fn record_with(splinter: &Splinter, request: &Request) -> anyhow::Result<Outcome> {
    let started = std::time::Instant::now();
    let manifest = Manifest::read(&request.run)?;
    // No session exists before the probes: this is the protocol's order.
    verified_probes(&request.run, &manifest)?;

    let simulator_ref = runtime::model_ref(&request.simulator)?;
    anyhow::ensure!(
        !matches!(
            &simulator_ref,
            ModelRef::Local {
                adapter: Some(_),
                ..
            }
        ),
        "the user is played by the plain model: {} names an adapter",
        request.simulator
    );
    let base = |reference: &str| reference.split('+').next().unwrap_or_default().to_string();
    anyhow::ensure!(
        base(&request.simulator) != base(&request.policy),
        "the user is played by a different model than the policy: both name {}",
        base(&request.policy)
    );
    let sessions = request.out.join("sessions");
    let index = request.out.join(INDEX);
    std::fs::create_dir_all(&sessions)?;
    let made = std::fs::read_to_string(&index).map_or(0, |t| t.lines().count());
    let mut pending: Vec<Pending> = Vec::new();
    for (n, item) in request.items.iter().enumerate() {
        let chosen = choose(&manifest, item, made + n)?;
        pending.push(Pending {
            turns: user_turns(&chosen.id),
            chosen,
            item: n,
        });
    }

    let ctx = splinter.context();
    let policy = Policy::load(
        &ctx,
        &runtime::model_ref(&request.policy)?,
        &manifest.persona,
    )?;
    let simulator = ctx.model(&simulator_ref)?;

    let mut recorded: Vec<Option<Recorded>> = request.items.iter().map(|_| None).collect();
    let mut swaps = 0;
    for attempt in 0..MAX_ATTEMPTS {
        if pending.is_empty() {
            break;
        }
        let gate = Gate::new(Side::User, pending.len());
        let mut gated = policy.model().clone();
        gated.provider = Arc::new(Gated::new(
            Arc::clone(&gated.provider),
            Arc::clone(&gate),
            Side::Policy,
        ));
        let runs = pending.iter().map(|p| {
            let item = &request.items[p.item];
            let (gate, gated, simulator) = (Arc::clone(&gate), &gated, &simulator);
            async move {
                let user = Simulator::new(
                    simulator,
                    p.chosen.subject.clone(),
                    item.style,
                    p.turns,
                    Some(Arc::clone(&gate)),
                )?;
                let run = async {
                    let opening = user
                        .say(&[])
                        .await
                        .context("the simulated user wrote no opening message")?;
                    converse(gated, &opening, &user, p.turns).await
                }
                .await;
                gate.leave();
                run
            }
        });
        let trajectories = ctx.block_on(futures::future::join_all(runs));
        swaps += gate.swaps();
        let mut retry = Vec::new();
        for (p, trajectory) in pending.into_iter().zip(trajectories) {
            let item = &request.items[p.item];
            // A run that failed, or whose conversation is malformed, loses its
            // own attempt and nobody else's.
            let judged = trajectory
                .map_err(|e| ("run_failed", format!("the run failed: {e:#}")))
                .and_then(|t| {
                    assess(&t, &p.chosen.keys)
                        .map(|a| (t, a))
                        .map_err(|e| ("malformed", format!("{e:#}")))
                });
            match judged {
                Ok((trajectory, Assessment::Kept(summary))) => {
                    let mut trajectory = trajectory;
                    if matches!(p.chosen.subject, Subject::Fact { .. }) {
                        annotate(&mut trajectory, &p.chosen.subject_id(item.canary));
                    }
                    let dir = sessions.join(format!("day-{:02}", p.chosen.day));
                    let path = save(&dir, ctx.clock(), &trajectory)?;
                    let entry = Recorded {
                        file: path
                            .strip_prefix(&request.out)
                            .unwrap_or(&path)
                            .display()
                            .to_string(),
                        kind: match p.chosen.subject {
                            Subject::Fact { .. } => "fact".into(),
                            Subject::Sham { .. } => "sham".into(),
                            Subject::Noise { .. } => "noise".into(),
                        },
                        fact: (!matches!(p.chosen.subject, Subject::Noise { .. }))
                            .then(|| p.chosen.id.clone()),
                        topic: match &p.chosen.subject {
                            Subject::Noise { topic } => Some(topic.clone()),
                            _ => None,
                        },
                        user_turns: summary.user_turns,
                        first_answer_holds_keys: summary.first_answer_holds_keys,
                        correction_names_missing_keys: summary.correction_names_missing_keys,
                        user_states_every_key: summary.user_states_every_key,
                        style: item.style,
                        canary: item.canary,
                        discarded_before: attempt,
                    };
                    let mut file = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&index)?;
                    writeln!(file, "{}", serde_json::to_string(&entry)?)?;
                    drop(file);
                    crate::subjects::write(&request.run, &request.out)?;
                    eprintln!("recorded {} ({} user turns)", entry.file, entry.user_turns);
                    recorded[p.item] = Some(entry);
                }
                Ok((_, Assessment::Discarded(why))) => {
                    note_discard(
                        &request.out,
                        &p.chosen.id,
                        attempt,
                        &json!(why),
                        why.reason(),
                    )?;
                    retry.push(p);
                }
                Err((reason, explanation)) => {
                    note_discard(
                        &request.out,
                        &p.chosen.id,
                        attempt,
                        &json!(reason),
                        &explanation,
                    )?;
                    retry.push(p);
                }
            }
        }
        pending = retry;
    }
    Ok(Outcome {
        recorded: recorded.into_iter().flatten().collect(),
        failed: pending
            .iter()
            .map(|p| {
                format!(
                    "{}: every one of {MAX_ATTEMPTS} attempts was discarded (see {DISCARDED})",
                    p.chosen.id
                )
            })
            .collect(),
        swaps,
        seconds: started.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plan_file_lists_the_sessions_recorded_together() {
        let dir = tempfile::tempdir().expect("dir");
        let file = dir.path().join("plan.jsonl");
        std::fs::write(
            &file,
            "{\"kind\":\"fact\",\"fact\":\"a\",\"style\":\"realistic\"}\n\
             {\"kind\":\"canary\",\"fact\":\"b\",\"day\":3}\n\
             {\"kind\":\"noise\",\"topic\":2}\n",
        )
        .expect("write");
        let items = read_plan(&file).expect("a plan");
        assert_eq!(items.len(), 3);
        assert!(matches!(&items[0].about, About::Fact(id) if id == "a"));
        assert_eq!(items[0].style, Style::Realistic);
        assert!(items[1].canary && items[1].day == Some(3));
        assert!(matches!(items[2].about, About::Noise(Some(2))));
        std::fs::write(&file, "{\"kind\":\"sham\"}\n").expect("write");
        assert!(read_plan(&file).is_err(), "a sham session names its fact");
        std::fs::write(&file, "{\"kind\":\"chat\"}\n").expect("write");
        assert!(read_plan(&file).is_err());
    }
}
