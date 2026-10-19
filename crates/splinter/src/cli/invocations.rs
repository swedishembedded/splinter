// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements documentation whose examples are checked
// against the software they describe, for its clients. If your team needs
// expertise in keeping a manual's commands runnable as a program changes, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The commands a manual shows, checked against the grammar.
//!
//! [`check`] finds every `splinter ...` line in the shell code blocks of
//! Markdown files and parses it with the real command tree, so a flag the
//! program no longer has, a misspelt subcommand or a missing required argument
//! fails the documentation check instead of reaching a reader. Nothing is
//! executed: a line is only parsed.
//!
//! What counts as an invocation: a line that starts with `splinter` (after an
//! optional `$ ` prompt) inside a fenced block whose language is `bash`, `sh`,
//! `shell` or `console`. A command continued with a trailing backslash is one
//! invocation; a `#` comment and everything from the first unquoted pipe,
//! `&&`, `||`, `;` or redirection on is not part of it. Blocks in any other
//! language (`text` for output, `json`, ...) are never read as commands.

use std::path::{Path, PathBuf};

use clap::error::ErrorKind;
use clap::CommandFactory;

use crate::cli::Cli;

/// One command found in a Markdown file.
#[derive(Debug, PartialEq, Eq)]
struct Invocation {
    /// The line the command starts on, counted from 1.
    line: usize,
    /// The words after the program name.
    args: Vec<String>,
}

/// Every file's invocations that the grammar rejects, as `file:line: reason`
/// messages; empty when all parse. A file that cannot be read is a message too.
#[must_use]
pub fn check(files: &[PathBuf]) -> Vec<String> {
    let mut problems = Vec::new();
    for file in files {
        match std::fs::read_to_string(file) {
            Ok(text) => problems.extend(check_text(file, &text)),
            Err(e) => problems.push(format!("{}: cannot read: {e}", file.display())),
        }
    }
    problems
}

fn check_text(file: &Path, text: &str) -> Vec<String> {
    invocations(text)
        .into_iter()
        .filter_map(|invocation| {
            let words = std::iter::once("splinter".to_string()).chain(invocation.args);
            match Cli::command().try_get_matches_from(words) {
                Ok(matches) => lone_word(&matches).map(|word| {
                    format!(
                        "{}:{}: `{word}` is not a command",
                        file.display(),
                        invocation.line
                    )
                }),
                Err(e)
                    if matches!(e.kind(), ErrorKind::DisplayHelp | ErrorKind::DisplayVersion) =>
                {
                    None
                }
                Err(e) => {
                    let reason = e.to_string();
                    let reason = reason.lines().next().unwrap_or("").trim();
                    Some(format!("{}:{}: {reason}", file.display(), invocation.line))
                }
            }
        })
        .collect()
}

/// The front door takes any text as a sentence, so a misspelt command parses.
/// A sentence is several words; a lone word that is no command is a typo.
fn lone_word(matches: &clap::ArgMatches) -> Option<String> {
    let sentence = matches.get_one::<String>("sentence")?;
    (!sentence.contains(char::is_whitespace)).then(|| sentence.clone())
}

/// The commands in the shell blocks of `text`.
fn invocations(text: &str) -> Vec<Invocation> {
    let mut found = Vec::new();
    let mut in_shell_block = false;
    let mut in_other_block = false;
    let mut pending: Option<(usize, String)> = None;
    for (index, line) in text.lines().enumerate() {
        let number = index + 1;
        if let Some(info) = line.trim_start().strip_prefix("```") {
            if in_shell_block || in_other_block {
                in_shell_block = false;
                in_other_block = false;
                flush(&mut pending, &mut found);
            } else if matches!(info.trim(), "bash" | "sh" | "shell" | "console") {
                in_shell_block = true;
            } else {
                in_other_block = true;
            }
            continue;
        }
        if !in_shell_block {
            continue;
        }
        let trimmed = line.trim();
        let continued = match pending.take() {
            Some((start, mut command)) => {
                command.push(' ');
                command.push_str(trimmed);
                (start, command)
            }
            None => {
                let command = trimmed.strip_prefix("$ ").unwrap_or(trimmed);
                if command != "splinter" && !command.starts_with("splinter ") {
                    continue;
                }
                (number, command.to_string())
            }
        };
        match continued.1.strip_suffix('\\') {
            Some(without) => pending = Some((continued.0, without.trim_end().to_string())),
            None => found.push(Invocation {
                line: continued.0,
                args: words(&continued.1).split_off(1),
            }),
        }
    }
    flush(&mut pending, &mut found);
    found
}

/// A command whose last line ended in a backslash with nothing after it.
fn flush(pending: &mut Option<(usize, String)>, found: &mut Vec<Invocation>) {
    if let Some((line, command)) = pending.take() {
        found.push(Invocation {
            line,
            args: words(&command).split_off(1),
        });
    }
}

/// The shell words of `command` up to the first unquoted control operator or
/// comment: single quotes keep their text, double quotes honour a backslash.
fn words(command: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut started = false;
    let mut chars = command.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                started = true;
                for q in chars.by_ref() {
                    if q == '\'' {
                        break;
                    }
                    word.push(q);
                }
            }
            '"' => {
                started = true;
                while let Some(q) = chars.next() {
                    match q {
                        '"' => break,
                        '\\' => match chars.next() {
                            Some(next @ ('"' | '\\' | '$' | '`')) => word.push(next),
                            Some(next) => {
                                word.push('\\');
                                word.push(next);
                            }
                            None => word.push('\\'),
                        },
                        _ => word.push(q),
                    }
                }
            }
            '\\' => {
                started = true;
                if let Some(next) = chars.next() {
                    word.push(next);
                }
            }
            '#' if !started => break,
            '|' | ';' | '&' | '>' | '<' => {
                // `2>` redirects a descriptor; the digit is not an argument.
                if matches!(c, '>' | '<') && started && word.chars().all(|d| d.is_ascii_digit()) {
                    started = false;
                }
                break;
            }
            c if c.is_whitespace() => {
                if started {
                    words.push(std::mem::take(&mut word));
                    started = false;
                }
            }
            c => {
                started = true;
                word.push(c);
            }
        }
    }
    if started {
        words.push(word);
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problems(markdown: &str) -> Vec<String> {
        check_text(Path::new("page.md"), markdown)
    }

    #[test]
    fn a_command_the_grammar_accepts_passes_whatever_its_layout() {
        let page = "```bash\n\
            $ splinter doctor\n\
            splinter learn ./manual.md --goal \"the QX-7 console\" \\\n    --kinds denoise --dry-run  # plan only\n\
            splinter \"learn the manual in ./docs\"\n\
            splinter status | jq .policy\n\
            splinter ask 'What is it?' --json > answer.json\n\
            splinter learn --help\n\
            ```\n";
        assert_eq!(problems(page), Vec::<String>::new());
    }

    #[test]
    fn a_flag_or_command_the_program_does_not_have_names_its_file_and_line() {
        let page = "intro\n\n```bash\nsplinter doctor\nsplinter learn m.md --no-such-flag\nsplinter frobnicate\nsplinter ask\n```\n";
        let found = problems(page);
        assert_eq!(found.len(), 3, "{found:?}");
        assert!(found[0].starts_with("page.md:5: ") && found[0].contains("--no-such-flag"));
        assert!(found[1].starts_with("page.md:6: ") && found[1].contains("`frobnicate`"));
        assert!(found[2].starts_with("page.md:7: "));
    }

    #[test]
    fn only_shell_blocks_are_commands() {
        let page = "```text\nsplinter nonsense --flag\n```\n\nsplinter prose --flag\n\n```json\n{\"splinter\": 1}\n```\n";
        assert_eq!(problems(page), Vec::<String>::new());
    }

    #[test]
    fn words_stop_at_comments_and_control_operators_but_not_inside_quotes() {
        assert_eq!(
            words("splinter ask 'a | b # c' \"d \\\" e\" f#g | tail"),
            ["splinter", "ask", "a | b # c", "d \" e", "f#g"]
        );
        assert_eq!(words("splinter doctor && echo ok"), ["splinter", "doctor"]);
        assert_eq!(words("splinter doctor 2>&1"), ["splinter", "doctor"]);
    }

    #[test]
    fn a_missing_file_is_a_problem_not_a_pass() {
        let found = check(&[PathBuf::from("/nonexistent/page.md")]);
        assert_eq!(found.len(), 1);
        assert!(found[0].contains("cannot read"));
    }
}
