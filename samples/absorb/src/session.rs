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
//! the dialogue stage uses). The user is played by a different model, never
//! shown the persona prompt or the probes: it knows the one fact it is
//! correcting (and the question it wants answered), opens with that question
//! in its own words, reads the policy's real answer and corrects it
//! conversationally. The user's messages are checked by code never to read the
//! statement out. A noise session is the same machinery with no fact: the user
//! chats about something unrelated.

use std::path::{Path, PathBuf};
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

use crate::grading::{question_task, ANSWER_DEADLINE};
use crate::keys::{missing, Key};
use crate::probes::shares_a_run;

/// The fewest user turns of a session, the opening included.
pub const MIN_USER_TURNS: usize = 2;

/// The most user turns of a session.
pub const MAX_USER_TURNS: usize = 5;

/// The longest message the user may write, in characters.
const MAX_MESSAGE_CHARS: usize = 600;

/// Output tokens one user message may run to.
const MESSAGE_MAX_OUTPUT_TOKENS: u64 = 512;

/// How long the user may take over one message.
const MESSAGE_DEADLINE: Duration = Duration::from_secs(300);

/// What the simulated user is.
const USER_ROLE: &str = "You play an ordinary person talking with an assistant, typing in plain \
everyday language: one to three short sentences, no lists, no quotation of documents. You write \
only your next message to the assistant.";

/// What the simulated user does with each message, given what it knows.
const USER_TASK: &str = "Write your next message to the assistant, as the brief's `goal` says. \
`you_know` is what you are certain of; the assistant has not seen it. Never copy a sentence or \
a long stretch of words from `you_know`: say it the way a person says something they know, in \
your own words. You are the person asking, never the person or the thing the facts are about, \
and never the assistant: do not speak as them or about yourself as if you were them.";

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

/// What the user is shown when it writes message `turn` of `of`.
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
    messages_in_all: usize,
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

/// What the user is after in message `turn` of `of` (counted from 1).
fn goal(subject: &Subject, turn: usize, of: usize) -> &'static str {
    match (subject, turn) {
        (Subject::Fact { .. }, 1) => {
            "Ask the assistant what you want to ask, in your own words. Do not tell it the answer."
        }
        (Subject::Fact { .. }, t) if t == of => {
            "React to the reply the way you would mid-conversation: confirm it where it agrees \
             with what you know, push back where it still differs. Do not sum up or ask the \
             assistant to remember anything."
        }
        (Subject::Fact { .. }, 2) => {
            "Read the assistant's reply. If it differs from what you know, tell it plainly that it \
             is wrong and say what is right, in your own words. If it already agrees, say so and \
             ask one natural follow-up."
        }
        (Subject::Fact { .. }, _) => {
            "React to the reply: confirm it where it now agrees with what you know, push back \
             where it still differs, and add a detail or an explanation in your own words."
        }
        (Subject::Sham { .. }, 1) => {
            "Ask the assistant what you want to ask, in your own words."
        }
        (Subject::Sham { .. }, _) => {
            "Accept whatever the assistant said politely, never say it is wrong, and ask one \
             natural follow-up about the same subject."
        }
        (Subject::Noise { .. }, 1) => "Start chatting with the assistant about your topic.",
        (Subject::Noise { .. }, t) if t == of => {
            "Wrap up the chat briefly and thank the assistant."
        }
        (Subject::Noise { .. }, _) => {
            "Respond to what the assistant said and ask one more concrete question about your topic."
        }
    }
}

/// A message that may not be sent: empty, long, or reading the statement out.
fn refuse_message(message: &str, statement: Option<&str>) -> Result<(), String> {
    let n = message.chars().count();
    if n == 0 || n > MAX_MESSAGE_CHARS {
        return Err(format!(
            "write a message of between 1 and {MAX_MESSAGE_CHARS} characters"
        ));
    }
    if statement.is_some_and(|s| shares_a_run(message, s)) {
        return Err("say it in your own words: do not copy a stretch of what you know".into());
    }
    Ok(())
}

/// The simulated user.
pub struct Simulator {
    session: TypedSession<Said>,
    subject: Subject,
    style: Style,
    of: usize,
}

impl Simulator {
    /// A user on `model`, in `style`, for a session about `subject` of `of`
    /// user turns.
    ///
    /// # Errors
    /// The model cannot be set up for the call.
    pub fn new(model: &Model, subject: Subject, style: Style, of: usize) -> anyhow::Result<Self> {
        let statement = match &subject {
            Subject::Fact { statement, .. } => Some(statement.clone()),
            Subject::Sham { .. } | Subject::Noise { .. } => None,
        };
        let session = TypedCall::<Said>::new("say_next", USER_TASK, USER_ROLE, MESSAGE_DEADLINE)
            .max_output_tokens(MESSAGE_MAX_OUTPUT_TOKENS)
            .postcondition(move |said: &Said| {
                refuse_message(said.message.trim(), statement.as_deref())
            })
            .start(model)?;
        Ok(Self {
            session,
            subject,
            style,
            of,
        })
    }

    /// The message after `so_far`: the opening when it is empty.
    pub async fn say(&self, so_far: &[Exchange]) -> Option<String> {
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
            messages_in_all: self.of,
            goal: goal(&self.subject, turn, self.of),
        };
        let said = self.session.call(&brief).await.ok()?;
        Some(said.message.trim().to_string()).filter(|m| !m.is_empty())
    }
}

#[async_trait::async_trait]
impl Interlocutor for Simulator {
    async fn next(&self, so_far: &[Exchange]) -> Option<String> {
        if so_far.len() >= self.of {
            return None;
        }
        self.say(so_far).await
    }
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

/// What `trajectory` shows about the session, given the fact's keys (empty
/// for a noise or sham session).
///
/// # Errors
/// The trajectory has fewer than two user turns or more than five, a turn
/// is not text, or the user and the agent do not alternate.
pub fn summarise(trajectory: &Trajectory, keys: &[Key]) -> anyhow::Result<Summary> {
    let users = texts(trajectory, StepOrigin::User);
    let agents = texts(trajectory, StepOrigin::Agent);
    anyhow::ensure!(
        (MIN_USER_TURNS..=MAX_USER_TURNS).contains(&users.len()),
        "the session has {} user turns, not {MIN_USER_TURNS} to {MAX_USER_TURNS}",
        users.len()
    );
    anyhow::ensure!(
        agents.len() == users.len(),
        "{} user turns but {} agent replies: the session is not an alternating conversation",
        users.len(),
        agents.len()
    );
    if keys.is_empty() {
        return Ok(Summary {
            user_turns: users.len(),
            first_answer_holds_keys: None,
            correction_names_missing_keys: None,
            user_states_every_key: None,
        });
    }
    let lacked = missing(&agents[0], keys);
    let said = |k: &Key, from: usize| users.iter().skip(from).any(|u| crate::keys::present(u, k));
    Ok(Summary {
        user_turns: users.len(),
        first_answer_holds_keys: Some(lacked.is_empty()),
        correction_names_missing_keys: Some(lacked.iter().all(|k| said(k, 1))),
        user_states_every_key: Some(keys.iter().all(|k| said(k, 0))),
    })
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
        policy.solving(SolveOptions::new(ANSWER_DEADLINE)),
    )
    .await?;
    anyhow::ensure!(
        solution.conclusion == RunConclusion::Success,
        "the session did not conclude: {:?}",
        solution.conclusion
    );
    Ok(solution.trajectory)
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
mod tests {
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use splinter_sdk::agent::sven::model::{
        CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
    };
    use splinter_sdk::vocabulary::clock::FixedClock;

    use super::*;
    use crate::keys::extract;

    /// A policy that gives the next reply of its script.
    struct Scripted(Mutex<VecDeque<String>>);

    #[async_trait::async_trait]
    impl ModelProvider for Scripted {
        fn name(&self) -> &str {
            "scripted"
        }
        fn model_name(&self) -> &str {
            "policy-1"
        }
        async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
            let reply = self
                .0
                .lock()
                .expect("lock")
                .pop_front()
                .expect("a scripted reply");
            Ok(Box::pin(futures::stream::iter(vec![
                Ok(ResponseEvent::TextDelta(reply)),
                Ok(ResponseEvent::Done),
            ])))
        }
    }

    /// A user that says its script, one message per turn.
    struct ScriptedUser(Vec<&'static str>);

    #[async_trait::async_trait]
    impl Interlocutor for ScriptedUser {
        async fn next(&self, so_far: &[Exchange]) -> Option<String> {
            self.0.get(so_far.len()).map(|m| (*m).to_string())
        }
    }

    const STATEMENT: &str = "Jefferson bought a copying press made by Boulton and Watt in 1785.";
    const QUESTION: &str = "Which firm made the copying press Jefferson bought?";

    #[tokio::test]
    async fn a_session_is_a_valid_atif_file_in_which_a_wrong_answer_is_corrected_in_the_users_words(
    ) {
        let policy = Model::new(
            Arc::new(Scripted(Mutex::new(
                vec![
                    "I bought my press from a London firm called Peale.".to_string(),
                    "Then I stand corrected: it was Boulton and Watt.".to_string(),
                ]
                .into(),
            ))),
            "policy-1",
        );
        let user = ScriptedUser(vec![
            "Who built the copier you had in Paris?",
            "No, that is not right, it was the Birmingham partners, Boulton and Watt, around 1785.",
        ]);
        let trajectory = converse(&policy, "Who built the copier you had in Paris?", &user, 2)
            .await
            .expect("a session");

        let dir = tempfile::tempdir().expect("dir");
        let clock = FixedClock::new("2026-10-09T08:15:30.123Z");
        let path = save(dir.path(), &clock, &trajectory).expect("saved");
        assert_eq!(
            path.file_name().and_then(|n| n.to_str()),
            Some("2026-10-09_08-15-30.123.atif.json")
        );

        let keys = extract(STATEMENT, QUESTION);
        let summary = summarise(&trajectory, &keys).expect("summary");
        assert_eq!(summary.user_turns, 2);
        assert_eq!(
            summary.first_answer_holds_keys,
            Some(false),
            "the first answer was wrong"
        );
        assert_eq!(
            summary.correction_names_missing_keys,
            Some(true),
            "the user named what was missing"
        );
        assert_eq!(summary.user_states_every_key, Some(true));
        let users = texts(&trajectory, StepOrigin::User);
        assert!(
            !shares_a_run(&users[1], STATEMENT),
            "the statement was not read out"
        );
        let origins: Vec<StepOrigin> = trajectory.steps.iter().map(|s| s.source).collect();
        assert_eq!(
            origins,
            vec![
                StepOrigin::User,
                StepOrigin::Agent,
                StepOrigin::User,
                StepOrigin::Agent
            ]
        );
    }

    #[test]
    fn the_statement_may_not_be_read_out_but_its_words_may_be_used() {
        assert!(shares_a_run(
            "Jefferson bought a copying press made by Boulton and Watt, I believe.",
            STATEMENT
        ));
        assert!(!shares_a_run(
            "It was the Birmingham firm, Boulton and Watt, in 1785.",
            STATEMENT
        ));
        assert!(refuse_message("", None).is_err());
        assert!(refuse_message(&"x".repeat(601), None).is_err());
    }

    #[test]
    fn a_session_has_two_to_five_user_turns_decided_by_its_seed() {
        let counts: Vec<usize> = (0..200).map(|n| user_turns(&format!("fact-{n}"))).collect();
        assert!(counts.iter().all(|c| (2..=5).contains(c)));
        for want in 2..=5 {
            assert!(counts.contains(&want), "{want} turns never chosen");
        }
        assert_eq!(user_turns("fact-1"), user_turns("fact-1"));
    }

    #[test]
    fn a_session_with_one_turn_is_refused() {
        let trajectory = Trajectory::new(
            "ATIF-v1.7",
            splinter_sdk::agent::sven::atif::AgentProfile::new("sven", "1"),
        );
        assert!(summarise(&trajectory, &[]).is_err());
    }
}
