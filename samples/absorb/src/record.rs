// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements recording of agent sessions as standard
// trajectories for learning systems that improve from their users' own
// conversations, for its clients. If your team needs expertise in turning
// agent sessions into training evidence, you can procure our services by
// sending an email to info@swedishembedded.com.

//! `session record`: one live session recorded under the run's output
//! directory, after the probes have been sealed.

use std::io::Write;
use std::path::PathBuf;

use anyhow::Context as _;
use splinter_sdk::vocabulary::model_ref::ModelRef;

use crate::facts::Manifest;
use crate::keys::{extract, Key};
use crate::policy::Policy;
use crate::roles::Role;
use crate::runtime;
use crate::seal::verified_probes;
use crate::session::{
    converse, save, summarise, user_turns, Recorded, Simulator, Style, Subject, NOISE_TOPICS,
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

/// What `session record` needs.
pub struct Request {
    /// The run's output directory.
    pub out: PathBuf,
    /// The model store, when not the configuration's.
    pub models: Option<PathBuf>,
    /// The policy that answers: the current release, with its adapter.
    pub policy: String,
    /// The model that plays the user: plain, with no adapter.
    pub simulator: String,
    /// The day the session belongs to; a test fact's own day when absent.
    pub day: Option<usize>,
    /// What it is about.
    pub about: About,
    /// How the user goes about it.
    pub style: Style,
    /// The user asserts the fact's false canary statement, not the truth.
    pub canary: bool,
}

/// The most times a session is recorded afresh when the user never states
/// every key of the fact.
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

fn choose(manifest: &Manifest, request: &Request, made: usize) -> anyhow::Result<Chosen> {
    let fact_id = match &request.about {
        About::Fact(id) | About::Sham(id) => id,
        About::Noise(topic) => {
            let topic = NOISE_TOPICS[topic.unwrap_or(made) % NOISE_TOPICS.len()];
            return Ok(Chosen {
                subject: Subject::Noise {
                    topic: topic.to_string(),
                },
                id: format!("noise-{made}"),
                day: request.day.unwrap_or(0),
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
    let day = match (fact.role, fact.day, request.day) {
        (Some(Role::Test), Some(own), Some(asked)) if own != asked => {
            anyhow::bail!("test fact {fact_id} is taught on day {own}, not day {asked}")
        }
        (_, Some(own), _) => own,
        (_, None, asked) => asked.unwrap_or(0),
    };
    if matches!(request.about, About::Sham(_)) {
        return Ok(Chosen {
            subject: Subject::Sham {
                question: fact.question.clone(),
            },
            id: fact_id.clone(),
            day,
            keys: Vec::new(),
        });
    }
    let statement = if request.canary {
        fact.canary
            .clone()
            .with_context(|| format!("fact {fact_id} is no canary: run `facts canary` first"))?
    } else {
        fact.statement.clone()
    };
    let keys = if request.canary {
        extract(&statement, &fact.question)
    } else {
        fact.keys.clone()
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

/// Records one session; see the module documentation.
///
/// # Errors
/// The probes are not sealed (or were edited), the fact is not one a session
/// may discuss, a model cannot run, every attempt is discarded, or the
/// session breaks the protocol (too few user turns, a schema violation).
pub fn record(request: &Request) -> anyhow::Result<Recorded> {
    let manifest = Manifest::read(&request.out)?;
    // No session exists before the probes: this is the protocol's order.
    verified_probes(&request.out, &manifest)?;

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
    let sessions = request.out.join("sessions");
    let index = request.out.join(INDEX);
    std::fs::create_dir_all(&sessions)?;
    let made = std::fs::read_to_string(&index).map_or(0, |t| t.lines().count());
    let chosen = choose(&manifest, request, made)?;
    let turns = user_turns(&chosen.id);

    let splinter = runtime::open(&request.out, request.models.as_ref())?;
    let ctx = splinter.context();
    let policy = Policy::load(
        &ctx,
        &runtime::model_ref(&request.policy)?,
        &manifest.persona,
    )?;
    let simulator = ctx.model(&simulator_ref)?;

    let started = std::time::Instant::now();
    for attempt in 0..MAX_ATTEMPTS {
        let user = Simulator::new(&simulator, chosen.subject.clone(), request.style, turns)?;
        let opening = ctx
            .block_on(user.say(&[]))
            .context("the simulated user wrote no opening message")?;
        let trajectory = ctx.block_on(converse(policy.model(), &opening, &user, turns))?;
        let summary = summarise(&trajectory, &chosen.keys)?;
        if summary.user_states_every_key == Some(false) {
            eprintln!(
                "attempt {} discarded: the user never stated every key",
                attempt + 1
            );
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(request.out.join(DISCARDED))?;
            writeln!(
                file,
                "{}",
                serde_json::json!({"session": chosen.id, "attempt": attempt + 1, "user_turns": summary.user_turns})
            )?;
            continue;
        }
        let dir = sessions.join(format!("day-{:02}", chosen.day));
        let path = save(&dir, ctx.clock(), &trajectory)?;
        let recorded = Recorded {
            file: path
                .strip_prefix(&request.out)
                .unwrap_or(&path)
                .display()
                .to_string(),
            kind: match chosen.subject {
                Subject::Fact { .. } => "fact".into(),
                Subject::Sham { .. } => "sham".into(),
                Subject::Noise { .. } => "noise".into(),
            },
            fact: (!matches!(chosen.subject, Subject::Noise { .. })).then(|| chosen.id.clone()),
            topic: match &chosen.subject {
                Subject::Noise { topic } => Some(topic.clone()),
                _ => None,
            },
            user_turns: summary.user_turns,
            first_answer_holds_keys: summary.first_answer_holds_keys,
            correction_names_missing_keys: summary.correction_names_missing_keys,
            user_states_every_key: summary.user_states_every_key,
            style: request.style,
            canary: request.canary,
            discarded_before: attempt,
        };
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&index)?;
        writeln!(file, "{}", serde_json::to_string(&recorded)?)?;
        eprintln!(
            "recorded {} ({} user turns) in {:.0}s",
            recorded.file,
            recorded.user_turns,
            started.elapsed().as_secs_f64()
        );
        return Ok(recorded);
    }
    anyhow::bail!(
        "session {}: the user never stated every key in {MAX_ATTEMPTS} attempts; nothing was saved",
        chosen.id
    )
}
