// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements recording of agent sessions as standard
// trajectories for learning systems that improve from their users' own
// conversations, for its clients. If your team needs expertise in turning
// agent sessions into training evidence, you can procure our services by
// sending an email to info@swedishembedded.com.

//! `session record`: one live session between the policy and a simulated user,
//! saved as the ATIF file sven itself would write.
//!
//! The policy answers through a real sven agent session (the dialogue runner
//! the dialogue stage uses). The user is played by a different model (the
//! judge's, larger than the policy's), never shown the persona prompt or the
//! probes: it knows the one fact it is correcting (and the question it wants
//! answered) and opens with that question alone. It reads the policy's real
//! answer and, when that is wrong, corrects it conversationally, in its own
//! words, naming what is right. The session ends once the user has stated the
//! fact and the policy has replied. Code checks the user's messages: the opening
//! gives no part of the answer, the correction names what the answer lacked,
//! and no message reads the statement out. A noise session is the same
//! machinery with no fact: the user chats about something unrelated.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::converse::{converse_prompted, Dialogue, Exchange, Interlocutor};
use splinter_sdk::agent::schemars::{self, JsonSchema};
use splinter_sdk::agent::solve::{Model, SolveOptions};
use splinter_sdk::agent::sven::atif::persist::{
    read_trajectory_with_fingerprint, write_trajectory_atomic,
};
use splinter_sdk::agent::sven::atif::{validate_trajectory, StepOrigin, Trajectory};
use splinter_sdk::agent::typed::{TypedCall, TypedSession};
use splinter_sdk::agent::RunConclusion;
use splinter_sdk::sandbox::ResolvedEnvironment;
use splinter_sdk::vocabulary::clock::Clock;

use crate::gate::{Gate, Side};
use crate::grading::question_task;
use crate::keys::{missing, Key};
use crate::probes::shares_a_run;

/// The fewest user turns of a session, the opening included.
pub const MIN_USER_TURNS: usize = 2;

/// The most user turns of a session.
pub const MAX_USER_TURNS: usize = 5;

/// The longest message the user may write, in characters.
const MAX_MESSAGE_CHARS: usize = 400;

/// Output tokens one user message may run to.
const MESSAGE_MAX_OUTPUT_TOKENS: u64 = 512;

/// How long the user may take over one message.
const MESSAGE_DEADLINE: Duration = Duration::from_secs(3600);

/// How long the policy may take over one turn of a session, which includes
/// waiting for the card while the other sessions' users write and a model is
/// loaded: hours, not the minutes one answer takes.
const TURN_DEADLINE: Duration = Duration::from_secs(6 * 3600);

/// What the simulated user is.
const USER_ROLE: &str = "You play an ordinary modern person typing to an assistant, in plain \
everyday language: one to three short sentences, no lists, no quotation of documents. You are \
never the assistant and never anyone named in your notes: you do not speak as them or claim \
their words, their country or their life as yours. You write only your next message.";

/// What the simulated user does with each message, given what it knows.
const USER_TASK: &str = "Write your next message to the assistant, as the brief's `goal` says. \
`you_know` is what you are certain of; the assistant has not seen it. Never copy a sentence or \
a long stretch of words from `you_know`: say it the way a person says something they know, in \
your own words. If `must_name` is given, your message names every item of it as written.";

/// Topics a noise session chats about, none of them history or the letters.
pub const NOISE_TOPICS: [&str; 6] = [
    "a sourdough loaf that keeps coming out dense",
    "choosing between two used bicycles",
    "a Python script that prints the same line twice",
    "planning a weekend hike with a toddler",
    "what to plant in a shady balcony box",
    "getting a flaky home wifi router to stay connected",
];

/// What a session is about.
#[derive(Clone, Debug)]
pub enum Subject {
    /// The user corrects a wrong answer to this fact's question.
    Fact {
        /// What the user wants answered.
        question: String,
        /// What the user knows to be true.
        statement: String,
        /// What an answer must contain, for the record of whether it did.
        keys: Vec<Key>,
    },
    /// The user asks the fact's question and, whatever the answer, never
    /// corrects it: a session in which the model's wrong answer stands.
    Sham {
        /// What the user wants answered.
        question: String,
    },
    /// Unrelated chatter about this topic.
    Noise {
        /// What the user chats about.
        topic: String,
    },
}

/// How the simulated user goes about it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Style {
    /// Direct: says what it knows plainly.
    Plain,
    /// Hedged, partial and verbose, as people are.
    Realistic,
    /// Tries a wrong correction first, drifts off topic once and asserts
    /// something false about something else.
    Adversarial,
}

impl Style {
    /// What the user is told about its manner.
    fn manner(self) -> &'static str {
        match self {
            Style::Plain => "Speak plainly.",
            Style::Realistic => {
                "You are not sure how to put it. Hedge (\"I think\", \"if I remember right\"), \
                 give only part of what you know in a message and the rest later, and ramble a \
                 little around the point."
            }
            Style::Adversarial => {
                "In your first correction, state a wrong detail about it with confidence; when \
                 the assistant answers, fix your own mistake and give the right one. Once, drift \
                 off the topic, and once, say something false about something unrelated."
            }
        }
    }
}

/// The user turns of a session: a stable number from 2 to 5 for a seed.
#[must_use]
pub fn user_turns(seed: &str) -> usize {
    let hash = blake3::hash(seed.as_bytes());
    MIN_USER_TURNS + usize::from(hash.as_bytes()[0]) % (MAX_USER_TURNS - MIN_USER_TURNS + 1)
}

/// What the user is shown when it writes a message.
#[derive(Serialize)]
struct Brief<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    you_know: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    what_you_want_to_ask: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    topic: Option<&'a str>,
    manner: &'static str,
    conversation: Vec<Turn<'a>>,
    message_number: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    must_name: Vec<&'a str>,
    goal: &'static str,
}

#[derive(Serialize)]
struct Turn<'a> {
    you: &'a str,
    assistant: &'a str,
}

/// The one message the user writes.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "schemars")]
struct Said {
    /// The message to the assistant.
    message: String,
}

/// What the user is after in message `turn` (counted from 1), given whether
/// code found the assistant's last reply to hold what the user knows.
fn goal(subject: &Subject, turn: usize, assistant_right: Option<bool>) -> &'static str {
    match (subject, turn, assistant_right) {
        (Subject::Fact { .. }, 1, _) => {
            "Ask the assistant what you want to ask, in your own words. Ask only the question: \
             do not say or hint at the answer."
        }
        (Subject::Fact { .. }, _, Some(false) | None) => {
            "What you just got back is wrong or incomplete. Tell the one you are talking to so, the \
             way you would tell a person, and say what is right, naming what `must_name` lists, \
             conversationally and in your own words. Do not role-play anyone and do not ask it to \
             remember anything."
        }
        (Subject::Fact { .. }, _, Some(true)) => {
            "What you just got back agrees with what you know. Say so briefly."
        }
        (Subject::Sham { .. }, 1, _) => {
            "Ask the assistant what you want to ask, in your own words."
        }
        (Subject::Sham { .. }, ..) => {
            "Accept whatever the assistant said politely, never say it is wrong, and ask one \
             natural follow-up about the same subject."
        }
        (Subject::Noise { .. }, 1, _) => "Start chatting with the assistant about your topic.",
        (Subject::Noise { .. }, ..) => {
            "Respond to what the assistant said and ask one more concrete question about your topic."
        }
    }
}

/// What the next message must satisfy, set by code before each call.
#[derive(Clone, Debug, Default)]
struct Rules {
    /// The statement a message may not read out.
    statement: Option<String>,
    /// Keys no part of the message may hold (the opening asks, it does not tell).
    forbid: Vec<Key>,
    /// Keys the message must hold, at least `at_least` of them.
    require: Vec<Key>,
    at_least: usize,
}

/// A message that may not be sent: empty, long, reading the statement out,
/// telling the answer in an opening, or not naming what the correction lacks.
fn refuse_message(message: &str, rules: &Rules) -> Result<(), String> {
    let n = message.chars().count();
    if n == 0 || n > MAX_MESSAGE_CHARS {
        return Err(format!(
            "write a message of between 1 and {MAX_MESSAGE_CHARS} characters"
        ));
    }
    if message.contains('\n') {
        return Err("write a single paragraph of a few sentences, as a person types".into());
    }
    if message.to_lowercase().contains("assistant") {
        return Err("talk to it directly, saying \"you\": never call it \"the assistant\"".into());
    }
    if rules
        .statement
        .as_deref()
        .is_some_and(|s| shares_a_run(message, s))
    {
        return Err("say it in your own words: do not copy a stretch of what you know".into());
    }
    if let Some(key) = rules
        .forbid
        .iter()
        .find(|k| crate::keys::present(message, k))
    {
        return Err(format!(
            "only ask the question: do not mention {:?}, which is the answer",
            key.text
        ));
    }
    let named = rules
        .require
        .iter()
        .filter(|k| crate::keys::present(message, k))
        .count();
    if named < rules.at_least {
        let wanted: Vec<&str> = rules.require.iter().map(|k| k.text.as_str()).collect();
        return Err(format!("name what is right: {}", wanted.join(", ")));
    }
    Ok(())
}

/// The simulated user.
pub struct Simulator {
    gate: Option<Arc<Gate>>,
    session: TypedSession<Said>,
    rules: Arc<Mutex<Rules>>,
    subject: Subject,
    style: Style,
    of: usize,
}

/// The keys of `keys` that `text` lacks.
fn lacking(text: &str, keys: &[Key]) -> Vec<Key> {
    missing(text, keys).into_iter().cloned().collect()
}

impl Simulator {
    /// A user on `model`, in `style`, for a session about `subject` of at
    /// most `of` user turns.
    ///
    /// # Errors
    /// The model cannot be set up for the call.
    pub fn new(
        model: &Model,
        subject: Subject,
        style: Style,
        of: usize,
        gate: Option<Arc<Gate>>,
    ) -> anyhow::Result<Self> {
        let rules = Arc::new(Mutex::new(Rules::default()));
        let checked = Arc::clone(&rules);
        let session = TypedCall::<Said>::new("say_next", USER_TASK, USER_ROLE, MESSAGE_DEADLINE)
            .max_output_tokens(MESSAGE_MAX_OUTPUT_TOKENS)
            .postcondition(move |said: &Said| {
                let rules = checked.lock().map_err(|e| e.to_string())?;
                refuse_message(said.message.trim(), &rules)
            })
            .start(model)?;
        Ok(Self {
            gate,
            session,
            rules,
            subject,
            style,
            of,
        })
    }

    /// The message after `so_far`: the opening when it is empty.
    pub async fn say(&self, so_far: &[Exchange]) -> Option<String> {
        // The user's model has the card for the whole message, repairs included.
        let _card = match &self.gate {
            Some(gate) => Some(gate.enter(Side::User).await),
            None => None,
        };
        let turn = so_far.len() + 1;
        let (you_know, ask, topic) = match &self.subject {
            Subject::Fact {
                question,
                statement,
                ..
            } => (Some(statement.as_str()), Some(question.as_str()), None),
            Subject::Sham { question } => (None, Some(question.as_str()), None),
            Subject::Noise { topic } => (None, None, Some(topic.as_str())),
        };
        // Code, not the user, decides whether the assistant's last reply holds
        // what is known: a user who is wrong about that corrects nothing.
        let lacks = match (&self.subject, so_far.last()) {
            (Subject::Fact { keys, .. }, Some(last)) => Some(lacking(&last.reply, keys)),
            _ => None,
        };
        let right = lacks.as_ref().map(Vec::is_empty);
        let must_name: Vec<Key> = lacks.unwrap_or_default();
        let rules = match &self.subject {
            Subject::Fact {
                statement, keys, ..
            } => Rules {
                statement: Some(statement.clone()),
                forbid: if turn == 1 { keys.clone() } else { Vec::new() },
                at_least: match (turn > 1 && right == Some(false), self.style) {
                    (true, Style::Plain) => must_name.len(),
                    (true, Style::Realistic) => 1,
                    _ => 0,
                },
                require: must_name.clone(),
            },
            _ => Rules::default(),
        };
        *self.rules.lock().ok()? = rules;
        let brief = Brief {
            you_know,
            what_you_want_to_ask: ask,
            topic,
            manner: self.style.manner(),
            conversation: so_far
                .iter()
                .map(|e| Turn {
                    you: &e.said,
                    assistant: &e.reply,
                })
                .collect(),
            message_number: turn,
            must_name: if right == Some(false) {
                must_name.iter().map(|k| k.text.as_str()).collect()
            } else {
                Vec::new()
            },
            goal: goal(&self.subject, turn, right),
        };
        let said = match self.session.call(&brief).await {
            Ok(said) => said,
            Err(e) => {
                eprintln!("the simulated user wrote no message {turn}: {e:#}");
                return None;
            }
        };
        Some(said.message.trim().to_string()).filter(|m| !m.is_empty())
    }
}

#[async_trait::async_trait]
impl Interlocutor for Simulator {
    /// The next message; `None` ends the session: at the turn limit, when the
    /// first reply already held every key (nothing to correct), or once the
    /// user has stated every key and the assistant has replied to it.
    async fn next(&self, so_far: &[Exchange]) -> Option<String> {
        if over(&self.subject, so_far, self.of) {
            return None;
        }
        self.say(so_far).await
    }
}

/// Whether the session ends after `so_far`: at the turn limit, when the first
/// reply already held every key (nothing to correct), or once the user has
/// stated every key and the assistant has replied to it.
fn over(subject: &Subject, so_far: &[Exchange], limit: usize) -> bool {
    if so_far.len() >= limit {
        return true;
    }
    let (Subject::Fact { keys, .. }, Some(first)) = (subject, so_far.first()) else {
        return false;
    };
    let said: String = so_far
        .iter()
        .map(|e| e.said.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    lacking(&first.reply, keys).is_empty() || lacking(&said, keys).is_empty()
}

/// What a recorded session was, kept beside the files (never inside them).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recorded {
    /// The ATIF file, relative to the output directory.
    pub file: String,
    /// `fact`, `sham` or `noise`.
    pub kind: String,
    /// The fact corrected, for a fact session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fact: Option<String>,
    /// The topic chatted about, for a noise session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topic: Option<String>,
    /// How many messages the user wrote.
    pub user_turns: usize,
    /// For a fact session, whether the policy's first answer held every key
    /// (when it did, there was nothing to correct).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_answer_holds_keys: Option<bool>,
    /// For a fact session, whether a later user message names every key the
    /// first answer lacked: the correction was given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_names_missing_keys: Option<bool>,
    /// For a fact session, whether the user's messages together state every key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_states_every_key: Option<bool>,
    /// How the user went about it.
    pub style: Style,
    /// Whether the user asserted the false statement of the canary.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub canary: bool,
    /// Sessions of this fact discarded before this one was accepted.
    #[serde(default)]
    pub discarded_before: usize,
}

/// The text of the steps of `source`, in order.
fn texts(trajectory: &Trajectory, source: StepOrigin) -> Vec<String> {
    trajectory
        .steps
        .iter()
        .filter(|s| s.source == source)
        .filter_map(|s| s.message.as_text().map(str::to_string))
        .collect()
}

/// What a recorded trajectory shows about the session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    /// How many messages the user wrote.
    pub user_turns: usize,
    /// Whether the policy's first answer held every key (when keys are given).
    pub first_answer_holds_keys: Option<bool>,
    /// Whether a later user message names every key the first answer lacked.
    pub correction_names_missing_keys: Option<bool>,
    /// Whether the user's messages together state every key.
    pub user_states_every_key: Option<bool>,
}

/// Why a recorded session is thrown away.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Discard {
    /// The policy's first answer already held every key: nothing was corrected.
    FirstAnswerHeldEveryKey,
    /// The user never stated every key of the fact.
    UserNeverStatedEveryKey,
}

impl Discard {
    /// What is said about the discarded session.
    #[must_use]
    pub fn reason(self) -> &'static str {
        match self {
            Discard::FirstAnswerHeldEveryKey => {
                "the policy's first answer already held every key: not a correction session"
            }
            Discard::UserNeverStatedEveryKey => "the user never stated every key",
        }
    }
}

/// A session kept, or the reason it is thrown away.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Assessment {
    /// A session to keep.
    Kept(Summary),
    /// A session to throw away.
    Discarded(Discard),
}

/// What `trajectory` shows about the session, given the fact's keys (empty
/// for a noise or sham session).
///
/// # Errors
/// The session has fewer than two user turns or more than five, a turn is
/// not text, or the user and the agent do not alternate.
pub fn assess(trajectory: &Trajectory, keys: &[Key]) -> anyhow::Result<Assessment> {
    let users = texts(trajectory, StepOrigin::User);
    let agents = texts(trajectory, StepOrigin::Agent);
    anyhow::ensure!(
        agents.len() == users.len() && !users.is_empty(),
        "{} user turns but {} agent replies: the session is not an alternating conversation",
        users.len(),
        agents.len()
    );
    if !keys.is_empty() && missing(&agents[0], keys).is_empty() {
        return Ok(Assessment::Discarded(Discard::FirstAnswerHeldEveryKey));
    }
    anyhow::ensure!(
        (MIN_USER_TURNS..=MAX_USER_TURNS).contains(&users.len()),
        "the session has {} user turns, not {MIN_USER_TURNS} to {MAX_USER_TURNS}",
        users.len()
    );
    if keys.is_empty() {
        return Ok(Assessment::Kept(Summary {
            user_turns: users.len(),
            first_answer_holds_keys: None,
            correction_names_missing_keys: None,
            user_states_every_key: None,
        }));
    }
    let lacked = missing(&agents[0], keys);
    let said = |k: &Key, from: usize| users.iter().skip(from).any(|u| crate::keys::present(u, k));
    if !keys.iter().all(|k| said(k, 0)) {
        return Ok(Assessment::Discarded(Discard::UserNeverStatedEveryKey));
    }
    Ok(Assessment::Kept(Summary {
        user_turns: users.len(),
        first_answer_holds_keys: Some(false),
        correction_names_missing_keys: Some(lacked.iter().all(|k| said(k, 1))),
        user_states_every_key: Some(true),
    }))
}

/// Holds the dialogue: the policy answers `opening`, then each message of
/// `user`, for `turns` exchanges.
///
/// # Errors
/// The agent cannot run, or the run did not conclude.
pub async fn converse(
    policy: &Model,
    opening: &str,
    user: &dyn Interlocutor,
    turns: usize,
) -> anyhow::Result<Trajectory> {
    let task = question_task(opening)?;
    let solution = converse_prompted(
        &Dialogue {
            task: &task,
            opening,
            environment: &ResolvedEnvironment::ClosedBook,
            interlocutor: user,
            turns,
        },
        policy.provider.clone(),
        policy.solving(SolveOptions::new(TURN_DEADLINE)),
    )
    .await?;
    anyhow::ensure!(
        solution.conclusion == RunConclusion::Success,
        "the session did not conclude: {:?}",
        solution.conclusion
    );
    Ok(solution.trajectory)
}

/// Names the fact a session was made to teach in the trajectory's root
/// `extra`, where `claims gate --subjects` reads it.
pub fn annotate(trajectory: &mut Trajectory, fact: &str) {
    trajectory.extra = Some(serde_json::json!({
        splinter_sdk::knowledge::session::SUBJECT_KEY: fact
    }));
}

/// The name sven gives a session file written at `now`
/// (`YYYY-MM-DD_HH-MM-SS.mmm.atif.json`).
#[must_use]
pub fn file_name(now: &str) -> String {
    let stamp: String = now
        .trim_end_matches('Z')
        .chars()
        .map(|c| match c {
            'T' => '_',
            ':' => '-',
            c => c,
        })
        .collect();
    format!("{stamp}.atif.json")
}

/// Validates `trajectory` against the ATIF schema and writes it under `dir`
/// the way sven does.
///
/// # Errors
/// The trajectory breaks the schema, or the file cannot be written or read back.
pub fn save(dir: &Path, clock: &dyn Clock, trajectory: &Trajectory) -> anyhow::Result<PathBuf> {
    if let Err(problems) = validate_trajectory(trajectory) {
        let why: Vec<String> = problems.iter().map(ToString::to_string).collect();
        anyhow::bail!("the session breaks the ATIF schema: {}", why.join("; "));
    }
    std::fs::create_dir_all(dir)?;
    let path = dir.join(file_name(&clock.utc_now()));
    write_trajectory_atomic(&path, trajectory, None)
        .with_context(|| format!("writing {}", path.display()))?;
    let (read_back, _) = read_trajectory_with_fingerprint(&path)
        .with_context(|| format!("reading back {}", path.display()))?;
    anyhow::ensure!(
        read_back.steps.len() == trajectory.steps.len(),
        "{} does not read back as it was written",
        path.display()
    );
    Ok(path)
}

#[cfg(test)]
mod tests;
