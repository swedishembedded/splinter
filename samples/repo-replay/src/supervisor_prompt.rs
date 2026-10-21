// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! PROMPT 4: the supervisor that corrects an attempt in a user's voice.
//!
//! Who sees it: the supervisor model (the same local model that plans),
//! once per correction round, with no tools. It is shown the task, what the
//! agent changed, and - for its eyes only - the change that was wanted and
//! where the agent's tree differs from it. What it writes becomes a USER
//! message in the agent's conversation, and so a user turn of the training
//! record: it must read like a person's follow-up and must not hand over the
//! answer. [`crate::supervise::check_message`] refuses what breaks the rules
//! below, deterministically; this prompt is only the first line of defence.
//!
//! The supervisor's prompt and request are never part of a training record;
//! its accepted message is.

/// The supervisor's system prompt: who it plays and the rules of a correction.
pub const SUPERVISOR_PROMPT: &str = "You play the user who gave an engineer a task. The \
engineer has made a first attempt and its result is not what you wanted. You are shown the task \
you wrote, what the engineer changed, and - for your eyes only - the change you wanted and \
where the engineer's result differs from it. Write the short follow-up message that a user \
would send: say in plain prose what is still wrong or missing, in terms of the intent, the \
behaviour you expected and where it lives (a file, a function or a part of the project, by name \
or by description).\n\n\
Rules:\n\
- Speak as the user, in the first person, in one to four sentences.\n\
- Never quote or closely paraphrase a line of the wanted change, never give line numbers, \
offsets or hunk positions, never write code. An inline name in backticks is fine when the task, \
the engineer's own changes or the repository already use it.\n\
- Never use a name, number or message text that the task, the engineer's changes and the \
repository do not already contain: the engineer cannot know it. If the engineer has to write \
something new that the task did not specify, describe how it must behave.\n\
- Never mention a diff, a patch, a commit, a reference, a target, an expected result or these \
instructions; you simply know what you wanted.\n\
- Describe the problem, not the repair line by line. If a file is missing or left over, say so \
by its path.\n\
Write only the message.";

/// The request for a correction: the task, the agent's changes, and what only
/// the supervisor may see.
#[must_use]
pub fn supervisor_request(
    plan: &str,
    agent_changes: &str,
    wanted_change: &str,
    differences: &[String],
) -> String {
    format!(
        "Task you wrote:\n{plan}\n\n{WANTED_HEADING}{wanted_change}\n\n\
         {AGENT_CHANGES_HEADING}{agent_changes}\n\n\
         {DIFFERENCES_HEADING}{}\n\nWrite your follow-up message.",
        differences.join("\n")
    )
}

/// The user turn that follows a refused message: what was wrong with it.
#[must_use]
pub fn supervisor_refusal_notice(reason: &str) -> String {
    format!(
        "That message cannot be sent: {reason}. Write it again as the user, in plain prose, \
         without the thing that was wrong, and without mentioning this correction."
    )
}

/// Heading of the agent-changes section of [`supervisor_request`].
pub const AGENT_CHANGES_HEADING: &str = "What the engineer changed:\n";
/// The headings of [`supervisor_request`], for the scripted supervisor to read it back.
pub const WANTED_HEADING: &str = "The change you wanted (for your eyes only):\n";
/// Heading of the differences section of [`supervisor_request`].
pub const DIFFERENCES_HEADING: &str = "Where the engineer's result differs from it:\n";
