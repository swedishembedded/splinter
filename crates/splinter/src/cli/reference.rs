// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements documentation that cannot drift from the
// software it describes, for its clients. If your team needs expertise in
// generating reference documentation from a program's own interface, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The command reference, generated from the grammar.
//!
//! [`markdown`] walks a clap command tree and writes every visible command,
//! argument, default and value as Markdown, so the reference cannot disagree
//! with the program. `splinter cli-reference` prints it,
//! `splinter cli-reference --check FILE` fails when a committed copy is stale and
//! `splinter cli-reference --check-invocations FILE...` fails when a manual
//! shows a command the grammar rejects.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use clap::{Arg, ArgAction, Args, Command, CommandFactory};

use crate::cli::{invocations, Cli};
use crate::session::Exit;

/// `cli-reference`: print the command reference, or check the documentation.
#[derive(Debug, Args)]
pub struct CliReferenceArgs {
    /// Do not print; compare against this file and exit 1 when it differs
    /// from what the grammar generates.
    #[arg(long, value_name = "FILE", conflicts_with = "check_invocations")]
    pub check: Option<PathBuf>,
    /// Do not print; parse the `splinter` commands in the shell blocks of
    /// these Markdown files against the grammar and exit 1, naming each file
    /// and line, for every one it rejects.
    #[arg(long, value_name = "FILE", num_args = 1..)]
    pub check_invocations: Vec<PathBuf>,
}

/// The Markdown reference of the whole command line.
#[must_use]
pub fn reference() -> String {
    markdown(&Cli::command())
}

/// Runs `cli-reference`.
#[must_use]
pub fn run(args: &CliReferenceArgs) -> Exit {
    if !args.check_invocations.is_empty() {
        let problems = invocations::check(&args.check_invocations);
        for problem in &problems {
            eprintln!("splinter: {problem}");
        }
        return if problems.is_empty() {
            Exit::Ok
        } else {
            Exit::Failed
        };
    }
    let generated = reference();
    let Some(file) = &args.check else {
        print!("{generated}");
        return Exit::Ok;
    };
    match check(&generated, file) {
        Ok(()) => Exit::Ok,
        Err(message) => {
            eprintln!("splinter: {message}");
            Exit::Failed
        }
    }
}

/// `Ok` when `file` holds exactly `generated`.
fn check(generated: &str, file: &Path) -> Result<(), String> {
    let committed = std::fs::read_to_string(file)
        .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
    if committed == generated {
        Ok(())
    } else {
        Err(format!(
            "{} is stale: regenerate it with `splinter cli-reference`",
            file.display()
        ))
    }
}

/// The Markdown reference of `root` and every visible command below it.
#[must_use]
pub fn markdown(root: &Command) -> String {
    let mut root = root.clone();
    root.build();
    let name = root.get_name().to_string();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "<!-- SPDX-License-Identifier: CC-BY-4.0 -->\n\
         <!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->\n\n\
         # Command reference\n\n\
         Generated from the command-line grammar; do not edit it. `{name} <command> --help`\n\
         prints the same text for one command. Worked examples are in the guides; this page\n\
         lists every command, argument and option.\n"
    );
    let globals: Vec<&Arg> = visible(&root).filter(|a| a.is_global_set()).collect();
    if !globals.is_empty() {
        out.push_str("## Global options\n\nEvery command takes these.\n\n");
        for arg in globals {
            out.push_str(&item(arg));
        }
        out.push('\n');
    }
    out.push_str("## Commands\n\n");
    index(&root, 0, &mut out);
    out.push('\n');
    section(&root, 1, &mut out);
    out.truncate(out.trim_end().len());
    out.push('\n');
    out
}

/// The arguments of `command` a reader sees.
fn visible(command: &Command) -> impl Iterator<Item = &Arg> {
    command
        .get_arguments()
        .filter(|a| !a.is_hide_set() && a.get_id() != "help" && a.get_id() != "version")
}

/// The subcommands of `command` a reader sees.
fn children(command: &Command) -> impl Iterator<Item = &Command> {
    command
        .get_subcommands()
        .filter(|c| !c.is_hide_set() && c.get_name() != "help")
}

fn index(command: &Command, depth: usize, out: &mut String) {
    for child in children(command) {
        let path = path_of(child);
        let _ = writeln!(
            out,
            "{}- [{path}](#{}) - {}",
            "  ".repeat(depth),
            anchor(&path),
            summary(child)
        );
        index(child, depth + 1, out);
    }
}

/// `command`'s section, then those of its subcommands, one heading level
/// deeper each.
fn section(command: &Command, depth: usize, out: &mut String) {
    let path = path_of(command);
    let level = "#".repeat((depth + 1).min(6));
    let _ = writeln!(out, "{level} {path}\n");
    let about = command
        .get_long_about()
        .or_else(|| command.get_about())
        .map(|text| prose(&text.to_string()))
        .unwrap_or_default();
    if !about.is_empty() {
        let _ = writeln!(out, "{about}\n");
    }
    let _ = writeln!(out, "```text\n{}\n```\n", usage(command));
    let (positionals, options): (Vec<&Arg>, Vec<&Arg>) = visible(command)
        .filter(|a| !a.is_global_set())
        .partition(|a| a.is_positional());
    list("Arguments", &positionals, out);
    list("Options", &options, out);
    let subs: Vec<&Command> = children(command).collect();
    if !subs.is_empty() {
        out.push_str("Subcommands:\n\n");
        for sub in &subs {
            let sub_path = path_of(sub);
            let _ = writeln!(
                out,
                "- [`{}`](#{}) - {}",
                sub.get_name(),
                anchor(&sub_path),
                summary(sub)
            );
        }
        out.push('\n');
    }
    for sub in subs {
        section(sub, depth + 1, out);
    }
}

fn list(title: &str, args: &[&Arg], out: &mut String) {
    if args.is_empty() {
        return;
    }
    let _ = writeln!(out, "{title}:\n");
    for arg in args {
        out.push_str(&item(arg));
    }
    out.push('\n');
}

/// One argument as a list item: its spelling, whether it is required, its
/// description, default and fixed values.
fn item(arg: &Arg) -> String {
    let mut line = format!("- `{}`", spelling(arg));
    if arg.is_required_set() {
        line.push_str(" (required)");
    }
    let help = arg
        .get_long_help()
        .or_else(|| arg.get_help())
        .map(|text| prose(&text.to_string()))
        .unwrap_or_default();
    if !help.is_empty() {
        let _ = write!(line, ": {}", sentence(&help));
    }
    // A flag's own "false" or 0 is the absence of the flag, not a default.
    let defaults: Vec<String> = arg
        .get_default_values()
        .iter()
        .filter(|_| arg.get_action().takes_values())
        .map(|v| v.to_string_lossy().into_owned())
        .filter(|v| !v.is_empty())
        .collect();
    if !defaults.is_empty() {
        let _ = write!(line, " Default: `{}`.", defaults.join(","));
    }
    let values: Vec<String> = arg
        .get_possible_values()
        .iter()
        .filter(|v| !v.is_hide_set())
        .map(|v| format!("`{}`", v.get_name()))
        .collect();
    if !values.is_empty() {
        let _ = write!(line, " Values: {}.", values.join(", "));
    }
    line.push('\n');
    line
}

/// How an argument is written on the command line.
fn spelling(arg: &Arg) -> String {
    let value = arg
        .get_value_names()
        .and_then(|names| names.first())
        .map_or_else(|| arg.get_id().as_str().to_uppercase(), |n| n.to_string());
    let many = matches!(arg.get_action(), ArgAction::Append)
        || arg.get_num_args().is_some_and(|n| n.max_values() > 1);
    let dots = if many { "..." } else { "" };
    if arg.is_positional() {
        return format!("<{value}>{dots}");
    }
    let mut names = Vec::new();
    if let Some(short) = arg.get_short() {
        names.push(format!("-{short}"));
    }
    if let Some(long) = arg.get_long() {
        names.push(format!("--{long}"));
    }
    let takes_value = arg.get_action().takes_values();
    let mut text = names.join(", ");
    if takes_value {
        let _ = write!(text, " <{value}>{dots}");
    }
    text
}

/// `splinter claims extract` for the subcommand `extract` of `claims`.
fn path_of(command: &Command) -> String {
    command
        .get_bin_name()
        .unwrap_or(command.get_name())
        .to_string()
}

fn usage(command: &Command) -> String {
    let rendered = command.clone().render_usage().to_string();
    rendered
        .strip_prefix("Usage: ")
        .unwrap_or(&rendered)
        .trim_end()
        .to_string()
}

/// The first clause of a command's description, for indexes: up to the
/// first colon, dash, semicolon or full stop that ends a phrase.
fn summary(command: &Command) -> String {
    let about = command
        .get_about()
        .map(|text| prose(&text.to_string()))
        .unwrap_or_default();
    let end = [": ", " - ", "; ", ". "]
        .iter()
        .filter_map(|stop| about.find(stop))
        .min()
        .unwrap_or(about.len());
    about[..end].trim_end_matches('.').to_string()
}

/// `text` as a sentence: ended with a full stop unless it already ends in
/// punctuation.
fn sentence(text: &str) -> String {
    if text.ends_with(['.', '!', '?', ']', ')']) {
        text.to_string()
    } else {
        format!("{text}.")
    }
}

/// A heading's anchor as GitHub and the publisher spell it.
fn anchor(heading: &str) -> String {
    heading.replace(' ', "-")
}

/// Help text as one Markdown line of prose: wrapped lines joined, and the
/// angle brackets outside code spans escaped so they are not read as HTML.
fn prose(text: &str) -> String {
    let joined = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = String::with_capacity(joined.len());
    let mut in_code = false;
    for ch in joined.chars() {
        match ch {
            '`' => {
                in_code = !in_code;
                out.push(ch);
            }
            '<' | '>' if !in_code => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Arg, ArgAction};

    /// A grammar with one of everything the reference has to show.
    fn sample() -> Command {
        Command::new("tool")
            .about("A tool. It does things.")
            .arg(
                Arg::new("json")
                    .long("json")
                    .global(true)
                    .action(ArgAction::SetTrue)
                    .help("Print JSON."),
            )
            .subcommand(
                Command::new("run")
                    .about("Run a job. Slowly.")
                    .arg(
                        Arg::new("job")
                            .required(true)
                            .value_name("JOB")
                            .help("What to run."),
                    )
                    .arg(
                        Arg::new("mode")
                            .long("mode")
                            .value_name("MODE")
                            .value_parser(["fast", "slow"])
                            .default_value("fast")
                            .help("How to run it."),
                    )
                    .arg(
                        Arg::new("tag")
                            .long("tag")
                            .short('t')
                            .action(ArgAction::Append)
                            .help("A tag, repeatable."),
                    )
                    .arg(Arg::new("secret").long("secret").hide(true))
                    .subcommand(Command::new("now").about("Run it now.")),
            )
            .subcommand(Command::new("internal").hide(true))
    }

    #[test]
    fn every_visible_command_and_option_is_documented_with_its_default_and_values() {
        let text = markdown(&sample());
        for expected in [
            "## tool run",
            "### tool run now",
            "- `--mode <MODE>`: How to run it. Default: `fast`. Values: `fast`, `slow`.",
            "- `<JOB>` (required): What to run.",
            "- `-t, --tag <TAG>...`: A tag, repeatable.",
            "- [tool run](#tool-run) - Run a job",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
        }
    }

    #[test]
    fn hidden_commands_and_options_are_left_out() {
        let text = markdown(&sample());
        assert!(
            !text.contains("internal") && !text.contains("--secret"),
            "{text}"
        );
    }

    #[test]
    fn a_global_option_is_listed_once_not_under_every_command() {
        let text = markdown(&sample());
        assert_eq!(text.matches("--json").count(), 1, "{text}");
    }

    #[test]
    fn a_flag_has_no_default_and_every_description_ends_in_a_full_stop() {
        let text = markdown(&sample());
        assert!(text.contains("- `--json`: Print JSON.\n"), "{text}");
        assert!(!text.contains("Default: `false`"), "{text}");
        assert_eq!(sentence("no stop"), "no stop.");
        assert_eq!(sentence("kept [default: 4]"), "kept [default: 4]");
    }

    #[test]
    fn an_index_entry_is_the_first_clause_of_the_description() {
        let command = Command::new("x").about("Learn from sources: capture them, train");
        assert_eq!(summary(&command), "Learn from sources");
        let command = Command::new("x").about("Show one stored source");
        assert_eq!(summary(&command), "Show one stored source");
    }

    #[test]
    fn angle_brackets_in_prose_are_escaped_but_code_spans_are_not() {
        assert_eq!(prose("a <id> and `<id>`"), "a \\<id\\> and `<id>`");
    }

    #[test]
    fn the_real_grammar_documents_every_visible_command_and_is_stable() {
        let root = Cli::command();
        let text = reference();
        assert_eq!(text, reference(), "generation is deterministic");
        let mut paths = Vec::new();
        fn walk(command: &Command, paths: &mut Vec<String>) {
            for child in children(command) {
                paths.push(path_of(child));
                walk(child, paths);
            }
        }
        let mut built = root.clone();
        built.build();
        walk(&built, &mut paths);
        assert!(paths.len() > 40, "{paths:?}");
        for path in paths {
            let has_heading = text
                .lines()
                .any(|line| line.starts_with('#') && line.trim_start_matches('#').trim() == path);
            assert!(has_heading, "no section for {path}");
        }
        assert!(
            !text
                .lines()
                .any(|line| line.starts_with('#') && line.ends_with("cli-reference")),
            "the hidden command documents itself"
        );
    }

    #[test]
    fn a_stale_copy_is_refused_and_a_current_one_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("cli.md");
        std::fs::write(&file, "old").unwrap();
        let generated = reference();
        assert!(check(&generated, &file).unwrap_err().contains("stale"));
        std::fs::write(&file, &generated).unwrap();
        assert!(check(&generated, &file).is_ok());
        assert!(check(&generated, &dir.path().join("absent.md")).is_err());
    }
}
