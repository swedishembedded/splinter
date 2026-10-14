// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that act through tools a caller
// defines, for its clients. If your team needs expertise in measuring and
// tracing what a model does with a tool it has never seen, you can procure
// our services by sending an email to info@swedishembedded.com.

//! `persona-chat chat` and `persona-chat batch`: talk to a model, trained
//! persona or not, through tools you define, and read the trace.
//!
//! ```text
//! persona-chat chat  --model Qwen/Qwen3-8B+maren [--tools tools.json | --options a,b] [--trace trace.jsonl]
//! persona-chat batch --model Qwen/Qwen3-8B+maren --prompts questions.jsonl [--tools tools.json] --trace trace.jsonl
//! ```
//!
//! A model is a name in the model store, `+` an adapter trained on it
//! (`Qwen/Qwen3-8B+maren`), or any reference the other commands take
//! (`local:...`, `remote:...`). With no tools file the model is offered
//! `decide(choice, motivation)` over the options given by `--options`. A tool
//! in the file may say `"final": true`: a correct call to it ends the
//! exchange, as `decide` does.

use std::io::{BufRead as _, Write as _};
use std::path::PathBuf;
use std::str::FromStr as _;
use std::sync::Arc;

use anyhow::{bail, Context as _};
use clap::{Parser, Subcommand};
use persona_chat::session::{Session, MAX_ROUNDS, MAX_TOKENS};
use persona_chat::tools::{decide, read_tools, ToolDef};
use persona_chat::trace;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::Splinter;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Args)]
struct Common {
    /// The model: `base`, `base+adapter`, or a full reference.
    #[arg(long)]
    model: String,
    /// The tools, a JSON file; `decide(choice, motivation)` when not given.
    #[arg(long)]
    tools: Option<PathBuf>,
    /// The options `decide` offers when no tools file is given.
    #[arg(long, value_delimiter = ',', default_value = "yes,no")]
    options: Vec<String>,
    /// What the model is told it is.
    #[arg(
        long,
        default_value = "You decide matters put to you. Act by calling the tools you are given."
    )]
    system: String,
    /// The most turns one question may take.
    #[arg(long, default_value_t = MAX_ROUNDS)]
    rounds: usize,
    /// The most tokens a turn may spend.
    #[arg(long, default_value_t = MAX_TOKENS)]
    max_tokens: u32,
    /// Where to append the trace, one exchange to a line.
    #[arg(long)]
    trace: Option<PathBuf>,
    /// Permit a model reached over the network.
    #[arg(long)]
    allow_remote: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Talk to the model at the terminal; an empty line ends it.
    Chat(Common),
    /// Put it every question of a file (a JSON object `{"id", "prompt"}` or a
    /// plain line to a row) and print and trace what it did.
    Batch {
        #[command(flatten)]
        common: Common,
        /// The questions.
        #[arg(long)]
        prompts: PathBuf,
    },
}

fn model_ref(text: &str) -> anyhow::Result<ModelRef> {
    let full = if ["local:", "remote:", "policy:"]
        .iter()
        .any(|scheme| text.starts_with(scheme))
    {
        text.to_string()
    } else {
        format!("local:{text}")
    };
    ModelRef::from_str(&full).with_context(|| format!("model {text:?}"))
}

fn session(common: &Common) -> anyhow::Result<Session> {
    let reference = model_ref(&common.model)?;
    let splinter = Splinter::from_env().context("starting the runtime")?;
    let ctx = splinter.context();
    if reference.is_remote() && !common.allow_remote {
        bail!(
            "{} is reached over the network; add --allow-remote",
            common.model
        );
    }
    let model = ctx.model(&reference).context("loading the model")?;
    let tools: Vec<ToolDef> = match &common.tools {
        Some(path) => {
            read_tools(&std::fs::read_to_string(path).with_context(|| path.display().to_string())?)?
        }
        None => vec![decide(&common.options)],
    };
    Ok(
        Session::new(Arc::clone(&model.provider), common.system.clone(), tools)
            .rounds(common.rounds)
            .max_tokens(common.max_tokens),
    )
}

fn main() -> anyhow::Result<()> {
    let runtime = tokio::runtime::Runtime::new()?;
    match Cli::parse().command {
        Command::Chat(common) => {
            let mut chat = session(&common)?;
            let log = common.trace.as_deref().map(trace::open).transpose()?;
            let stdin = std::io::stdin();
            let mut n = 0;
            loop {
                print!("you> ");
                std::io::stdout().flush()?;
                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 || line.trim().is_empty() {
                    return Ok(());
                }
                n += 1;
                let exchange = runtime.block_on(chat.say(line.trim()))?;
                print!("{}", trace::render(&exchange));
                if let Some(log) = &log {
                    trace::record(log, &format!("chat-{n}"), &exchange)?;
                }
            }
        }
        Command::Batch { common, prompts } => {
            let mut chat = session(&common)?;
            let log = common.trace.as_deref().map(trace::open).transpose()?;
            let text =
                std::fs::read_to_string(&prompts).with_context(|| prompts.display().to_string())?;
            for (n, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
                let (id, prompt) = match serde_json::from_str::<serde_json::Value>(line) {
                    Ok(row) if row["prompt"].is_string() => (
                        row["id"]
                            .as_str()
                            .map_or_else(|| format!("q{}", n + 1), String::from),
                        row["prompt"].as_str().unwrap_or_default().to_string(),
                    ),
                    _ => (format!("q{}", n + 1), line.trim().to_string()),
                };
                // Each question is put to the model on its own.
                chat.forget();
                let exchange = runtime.block_on(chat.say(&prompt))?;
                println!("[{id}]\n{}", trace::render(&exchange));
                if let Some(log) = &log {
                    trace::record(log, &id, &exchange)?;
                }
            }
            Ok(())
        }
    }
}
