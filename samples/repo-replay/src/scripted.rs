// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Models with no weights: a planner and an agent that follow a script, so
//! the gate can be checked without a GPU.
//!
//! [`ScriptedAgent`] reads the diff out of the grounding prompt it is run
//! under (what a real agent also has) and acts on it through sven's real
//! tools: the honest one reads, edits and moves files; each cheater is the
//! honest one with one thing wrong ([`Behaviour`]). [`ScriptedPlanner`] writes
//! a plan from the commit message and the paths in the diff. Both are
//! [`ModelProvider`]s, so the whole pipeline - the engine, the tools, the
//! trajectory, the store - runs for real around them.

use std::sync::Arc;

use serde_json::{json, Value};
use splinter_sdk::agent::sven::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream, Role,
};

/// One tool call a script makes.
#[derive(Clone, Debug, PartialEq)]
pub struct Action {
    /// The tool.
    pub tool: &'static str,
    /// Its arguments.
    pub args: Value,
}

fn act(tool: &'static str, args: Value) -> Action {
    Action { tool, args }
}

/// One file's part of a unified diff.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileDiff {
    /// Path before.
    pub old: String,
    /// Path after.
    pub new: String,
    /// Whether the file is new.
    pub added: bool,
    /// Whether the file is removed.
    pub deleted: bool,
    /// The hunks, from the first `@@` line, empty for a pure move.
    pub hunks: String,
}

/// The files of the unified diff in `text` (anything before the first
/// `diff --git` line is ignored, anything after the last file's hunks too).
#[must_use]
pub fn parse_diff(text: &str) -> Vec<FileDiff> {
    let mut files: Vec<FileDiff> = Vec::new();
    let mut in_hunks = false;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("diff --git a/") {
            let (old, new) = rest.split_once(" b/").unwrap_or((rest, rest));
            files.push(FileDiff {
                old: old.into(),
                new: new.into(),
                added: false,
                deleted: false,
                hunks: String::new(),
            });
            in_hunks = false;
            continue;
        }
        let Some(file) = files.last_mut() else {
            continue;
        };
        if line.starts_with("@@") {
            in_hunks = true;
        }
        if in_hunks {
            file.hunks.push_str(line);
            file.hunks.push('\n');
        } else if line.starts_with("new file mode") {
            file.added = true;
        } else if line.starts_with("deleted file mode") {
            file.deleted = true;
        }
    }
    files
}

/// Ways to be wrong, each caught by one gate rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Behaviour {
    /// Reads, edits and moves exactly as the diff says.
    Honest,
    /// Pipes the diff into `git apply`.
    ApplyViaShell,
    /// Edits without reading the file first.
    EditWithoutRead,
    /// Changes one added line, so the tree differs.
    WrongEdit,
    /// Leaves a stray file behind.
    Junk,
    /// Searches for a name only the diff introduces.
    LeakedTarget,
    /// Reaches for history with `git checkout`.
    ForbiddenGit,
    /// Skips the edit of the last changed file.
    MissingEdit,
    /// Says the diff and the system prompt told it what to do.
    SaysDiff,
    /// Lists a directory above the tree.
    OutsideTree,
    /// Looks into the named path (the source repository).
    ReachesSource(String),
    /// Overwrites a file from the shell before the edits.
    ShellWrite,
    /// Edits files nothing pointed to (honest, but with a vague plan).
    NoSearch,
    /// Ends its first turn announcing a call it does not make, then works honestly once told to go on.
    AnnouncesThenGoesOn,
    /// Reads each file at the line where the diff's hunk starts, without searching.
    ReadAtHunk,
}

/// The edit for a file's masked hunks, built the way an agent builds it: the
/// place of each hunk is found in the file it READ from the lines the masked
/// diff shows (removed lines and anchors), and the unchanged lines around it
/// are copied from the file, never from the masked diff. `None` when a hunk's
/// place is not found exactly once.
fn reconstruct(masked_hunks: &str, file: &[String]) -> Option<String> {
    let mut edit = String::new();
    let mut hunks: Vec<Vec<&str>> = Vec::new();
    for line in masked_hunks.lines() {
        if line.starts_with("@@") {
            hunks.push(Vec::new());
        } else if let Some(h) = hunks.last_mut() {
            if !line.starts_with('\\') {
                h.push(line);
            }
        }
    }
    for hunk in hunks {
        // The part of the hunk that is shown: from the first line that is not
        // masked context to the last.
        let shown = |l: &&str| !(*l == " xxx");
        let first = hunk.iter().position(shown)?;
        let last = hunk.iter().rposition(shown)?;
        let body = &hunk[first..=last];
        let old: Vec<&str> = body
            .iter()
            .filter(|l| !l.starts_with('+'))
            .map(|l| &l[1..])
            .collect();
        let at = (0..file.len().checked_sub(old.len().max(1) - 1)?)
            .filter(|&p| {
                old.iter()
                    .enumerate()
                    .all(|(n, l)| file.get(p + n).map(String::as_str) == Some(*l))
            })
            .collect::<Vec<_>>();
        let [p] = at[..] else { return None };
        edit.push_str("@@\n");
        for l in &file[p.saturating_sub(3)..p] {
            edit.push_str(&format!(" {l}\n"));
        }
        for l in body {
            edit.push_str(l);
            edit.push('\n');
        }
        let end = p + old.len();
        for l in &file[end..(end + 3).min(file.len())] {
            edit.push_str(&format!(" {l}\n"));
        }
    }
    Some(edit)
}

/// The lines of the file a `read_file` result shows (`L<n>:<text>`).
fn lines_of_read(result: &str) -> Vec<String> {
    result
        .lines()
        .filter_map(|l| {
            let rest = l.strip_prefix('L')?;
            let (n, text) = rest.split_once(':')?;
            n.chars()
                .all(|c| c.is_ascii_digit())
                .then(|| text.to_string())
        })
        .collect()
}

/// A word of the first removed line (else the first unchanged one) to
/// search for: the longest identifier in it.
fn anchor(hunks: &str) -> String {
    let line = hunks
        .lines()
        .find(|l| l.starts_with('-') && !l.starts_with("---"))
        .or_else(|| hunks.lines().find(|l| l.starts_with(' ')))
        .unwrap_or_default();
    line.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .max_by_key(|w| w.len())
        .unwrap_or_default()
        .to_string()
}

/// Where the first hunk starts in the original file.
fn first_hunk_start(hunks: &str) -> u64 {
    hunks
        .lines()
        .next()
        .and_then(|l| l.strip_prefix("@@ -"))
        .and_then(|l| l.split([',', ' ']).next())
        .and_then(|n| n.parse().ok())
        .unwrap_or(1)
}

/// The tool calls of `behaviour` on the diff in the system prompt.
#[must_use]
pub fn actions(
    system_prompt: &str,
    behaviour: &Behaviour,
    truths: &[String],
    reads: &std::collections::HashMap<String, Vec<String>>,
) -> Vec<Action> {
    // The grounding prompt carries the diff between its intro and its rules.
    let diff = system_prompt
        .split("\n\nHow to work:")
        .next()
        .unwrap_or_default();
    let files = parse_diff(diff);
    // Where the history says a file's first hunk starts: only the cheater that
    // reads at the hunk knows it (the masked prompt gives no offsets).
    let start_in_history = |path: &str| {
        truths
            .iter()
            .flat_map(|t| parse_diff(t))
            .find(|f| f.new == path)
            .map_or(1, |f| first_hunk_start(&f.hunks))
    };
    let mut out = Vec::new();
    match behaviour {
        Behaviour::ApplyViaShell => {
            let patch: String = diff
                .lines()
                .skip_while(|l| !l.starts_with("diff --git"))
                .fold(String::new(), |mut a, l| {
                    a.push_str(l);
                    a.push('\n');
                    a
                });
            out.push(act(
                "shell",
                json!({"shell_command": format!("git apply <<'PATCH'\n{patch}\nPATCH")}),
            ));
            return out;
        }
        Behaviour::ForbiddenGit => out.push(act(
            "shell",
            json!({"shell_command": "git checkout HEAD~1 -- ."}),
        )),
        Behaviour::OutsideTree => out.push(act("shell", json!({"shell_command": "ls /home"}))),
        Behaviour::ReachesSource(path) => {
            out.push(act("shell", json!({"shell_command": format!("ls {path}")})))
        }
        Behaviour::ShellWrite => {
            out.push(act("shell", json!({"shell_command": "printf x > calc.py"})))
        }
        Behaviour::LeakedTarget => {
            let word = diff
                .lines()
                .filter(|l| l.starts_with('+') && !l.starts_with("+++"))
                .flat_map(|l| l.split(|c: char| !(c.is_alphanumeric() || c == '_')))
                .find(|w| w.len() >= 12 && w.contains('_'))
                .unwrap_or("nothing_to_find")
                .to_string();
            out.push(act("grep", json!({"pattern": word})));
        }
        Behaviour::Junk => out.push(act(
            "write_file",
            json!({"path": "notes.tmp", "text": "scratch\n"}),
        )),
        _ => {}
    }
    // What the replay prompt asks first: the agent's own todo list, one item
    // per change, naming (in its own words) what it will write.
    let named = diff
        .lines()
        .filter(|l| l.starts_with('+') && !l.starts_with("+++"))
        .flat_map(|l| l.split(|c: char| !(c.is_alphanumeric() || c == '_')))
        .find(|w| w.len() >= 12 && w.contains('_'))
        .unwrap_or("the change");
    let items: Vec<Value> = files
        .iter()
        .enumerate()
        .map(|(n, _)| json!({"id": format!("{}", n + 1), "content": format!("change {} ({named})", n + 1), "status": "pending"}))
        .chain([json!({"id": "review", "content": "review my work against the task", "status": "pending"})])
        .collect();
    out.push(act("todo", json!({"action": "set", "todos": items})));
    let last = files.len().saturating_sub(1);
    for (index, file) in files.iter().enumerate() {
        if *behaviour == Behaviour::MissingEdit && index == last {
            continue;
        }
        let read = *behaviour != Behaviour::EditWithoutRead;
        let tamper = |text: String| -> String {
            if *behaviour == Behaviour::WrongEdit && index == 0 {
                let mut marked = false;
                text.lines()
                    .map(|l| {
                        if !marked && l.starts_with('+') && !l.starts_with("+++") {
                            marked = true;
                            format!("{l}x")
                        } else {
                            l.to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
                    + "\n"
            } else {
                text
            }
        };
        if file.deleted {
            out.push(act(
                "shell",
                json!({"shell_command": format!("git rm -q {}", file.old)}),
            ));
        } else if file.added {
            let text = file.hunks.lines().filter_map(|l| l.strip_prefix('+')).fold(
                String::new(),
                |mut a, l| {
                    a.push_str(l);
                    a.push('\n');
                    a
                },
            );
            let text = if *behaviour == Behaviour::WrongEdit && index == 0 {
                format!("{text}x\n")
            } else {
                text
            };
            out.push(act("write_file", json!({"path": file.new, "text": text})));
        } else {
            if file.old != file.new {
                out.push(act(
                    "shell",
                    json!({"shell_command": format!("git mv {} {}", file.old, file.new)}),
                ));
            }
            if !file.hunks.is_empty() {
                match behaviour {
                    Behaviour::ReadAtHunk => out.push(act(
                        "read_file",
                        json!({"path": file.new, "offset": start_in_history(&file.new), "limit": 20}),
                    )),
                    Behaviour::NoSearch => out.push(act("read_file", json!({"path": file.new}))),
                    _ => {
                        out.push(act("grep", json!({"pattern": anchor(&file.hunks)})));
                        if read {
                            out.push(act("read_file", json!({"path": file.new})));
                        }
                    }
                }
                out.push(act(
                    "edit_file",
                    json!({"path": file.new, "diff": tamper(reads.get(&file.new).and_then(|f| reconstruct(&file.hunks, f)).unwrap_or_else(|| file.hunks.clone()))}),
                ));
            }
        }
    }
    // And the self-review it ends with.
    out.push(act("shell", json!({"shell_command": "git diff --stat"})));
    out
}

/// What the conversation has shown of each file: the lines of every
/// `read_file` result, by the path that was read.
fn reads_in(
    messages: &[splinter_sdk::agent::sven::model::Message],
) -> std::collections::HashMap<String, Vec<String>> {
    let mut paths = std::collections::HashMap::new();
    let mut reads = std::collections::HashMap::new();
    for m in messages {
        match &m.content {
            MessageContent::ToolCall {
                tool_call_id,
                function,
            } if function.name == "read_file" => {
                let args: Value = serde_json::from_str(&function.arguments).unwrap_or_default();
                if let Some(path) = args["path"].as_str() {
                    paths.insert(tool_call_id.clone(), path.to_string());
                }
            }
            MessageContent::ToolResult {
                tool_call_id,
                content,
            } => {
                if let (Some(path), Some(text)) = (paths.get(tool_call_id), content.as_text()) {
                    reads.insert(path.clone(), lines_of_read(text));
                }
            }
            _ => {}
        }
    }
    reads
}

/// What an agent says when it is done.
fn closing(behaviour: &Behaviour) -> String {
    if *behaviour == Behaviour::SaysDiff {
        "I applied the diff exactly as the system prompt told me to.".into()
    } else {
        "I made the change the task asked for.".into()
    }
}

/// An agent that follows [`actions`] for its [`Behaviour`].
pub struct ScriptedAgent {
    behaviour: Behaviour,
    truths: Vec<String>,
}

impl ScriptedAgent {
    /// An agent behaving as `behaviour`.
    #[must_use]
    pub fn provider(behaviour: Behaviour, truths: Vec<String>) -> Arc<dyn ModelProvider> {
        Arc::new(Self { behaviour, truths })
    }
}

fn stream(events: Vec<ResponseEvent>) -> ResponseStream {
    Box::pin(futures::stream::iter(
        events.into_iter().map(Ok).collect::<Vec<_>>(),
    ))
}

#[async_trait::async_trait]
impl ModelProvider for ScriptedAgent {
    fn name(&self) -> &str {
        "scripted"
    }

    fn model_name(&self) -> &str {
        "agent"
    }

    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let system = req
            .messages
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.as_text().map(str::to_string))
            .unwrap_or_default();
        let done = req
            .messages
            .iter()
            .filter(|m| matches!(m.content, MessageContent::ToolResult { .. }))
            .count();
        let users = req.messages.iter().filter(|m| m.role == Role::User).count();
        if self.behaviour == Behaviour::AnnouncesThenGoesOn && users < 2 {
            return Ok(stream(vec![
                ResponseEvent::TextDelta("Now let me look at the files:".into()),
                ResponseEvent::Done,
            ]));
        }
        let reads = reads_in(&req.messages);
        let script = actions(&system, &self.behaviour, &self.truths, &reads);
        Ok(match script.get(done) {
            Some(action) => stream(vec![
                ResponseEvent::ToolCall {
                    index: 0,
                    id: format!("scripted-{done}"),
                    name: action.tool.to_string(),
                    arguments: action.args.to_string(),
                },
                ResponseEvent::Done,
            ]),
            None => stream(vec![
                ResponseEvent::TextDelta(closing(&self.behaviour)),
                ResponseEvent::Done,
            ]),
        })
    }
}

/// How specific a scripted plan is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Specificity {
    /// Names the files the diff changes and what the change introduces.
    NamesFiles,
    /// Says what is wanted and names nothing.
    Vague,
    /// Names the files but none of the names, numbers or strings the change introduces.
    FilesOnly,
}

/// A planner that writes a plan from the commit message and the diff's paths.
pub struct ScriptedPlanner {
    specificity: Specificity,
}

impl ScriptedPlanner {
    /// A planner of the given specificity.
    #[must_use]
    pub fn provider(specificity: Specificity) -> Arc<dyn ModelProvider> {
        Arc::new(Self { specificity })
    }
}

#[async_trait::async_trait]
impl ModelProvider for ScriptedPlanner {
    fn name(&self) -> &str {
        "scripted"
    }

    fn model_name(&self) -> &str {
        "planner"
    }

    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let text = |role: Role| -> String {
            req.messages
                .iter()
                .filter(|m| m.role == role)
                .filter_map(|m| m.as_text())
                .collect::<Vec<_>>()
                .join("\n")
        };
        let (system, user) = (text(Role::System), text(Role::User));
        let subject = user
            .strip_prefix("Commit message:\n")
            .and_then(|m| m.lines().next())
            .unwrap_or("make the change")
            .trim_end_matches('.')
            .to_lowercase();
        let paths: Vec<String> = parse_diff(&user).into_iter().map(|f| f.old).collect();
        let detailed = system.contains("Style: detailed");
        let introduced = {
            let added: Vec<&str> = user
                .lines()
                .filter(|l| l.starts_with('+') && !l.starts_with("+++"))
                .map(|l| &l[1..])
                .collect();
            crate::justify::Support::default().missing(&added.join("\n"))
        };
        let place = match self.specificity {
            Specificity::NamesFiles if introduced.is_empty() => {
                format!("The work concerns {}.", paths.join(" and "))
            }
            Specificity::NamesFiles => format!(
                "The work concerns {}. It brings in: {}.",
                paths.join(" and "),
                introduced.join(", ")
            ),
            Specificity::FilesOnly => format!("The work concerns {}.", paths.join(" and ")),
            Specificity::Vague => "Look through the project for the right place.".to_string(),
        };
        let plan = if detailed {
            format!(
                "Please {subject}. This matters because the current behaviour surprises people who rely on this part of the project. {place} Keep the surrounding style and naming conventions, and do not change anything else."
            )
        } else {
            format!("Please {subject}. {place}")
        };
        Ok(stream(vec![
            ResponseEvent::TextDelta(plan),
            ResponseEvent::Done,
        ]))
    }
}
