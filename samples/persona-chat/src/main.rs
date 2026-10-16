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
//!
//! A model that runs here samples. `--seed` is the session's seed (drawn and
//! printed when not given); each exchange is decoded from a seed derived from
//! it, and its trace line names the model, the temperature and that seed.

use std::io::{BufRead as _, Write as _};
use std::path::PathBuf;
use std::str::FromStr as _;
use std::sync::Arc;

use anyhow::{bail, Context as _};
use clap::{Parser, Subcommand};
use persona_chat::decoding::Decoding;
use persona_chat::session::{Session, MAX_ROUNDS, MAX_TOKENS};
use persona_chat::tools::{decide, read_tools, ToolDef};
use persona_chat::trace;
use splinter_sdk::agent::sven::model::ModelProvider;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::{Context, Splinter};

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    device: splinter_sdk::device::DeviceFlags,
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
    /// The seed of the session's sampling; each exchange is decoded from its
    /// own seed derived from it. Drawn and printed when not given; a model
    /// reached over a network samples as its server does.
    #[arg(long)]
    seed: Option<u64>,
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

/// A session and where the model of each of its exchanges comes from.
struct Chat {
    session: Session,
    model: String,
    source: Source,
    exchanges: u64,
}

enum Source {
    /// A model reached over a network: the same for every exchange.
    Remote,
    /// A model that runs here, made again for each exchange with its seed.
    Local(Box<Local>),
}

struct Local {
    _runtime: Splinter,
    ctx: Context,
    reference: ModelRef,
    seed: u64,
}

impl Chat {
    /// Readies the session for its next exchange and returns how the model
    /// decodes it; none for a model reached over a network.
    fn next_exchange(&mut self) -> anyhow::Result<Option<Decoding>> {
        let Source::Local(local) = &self.source else {
            return Ok(None);
        };
        self.exchanges += 1;
        let decoding = Decoding::for_exchange(local.seed, self.exchanges);
        let model = local
            .ctx
            .resampled(&local.reference, decoding.sampling())?
            .with_context(|| format!("{} cannot be sampled", local.reference))?;
        self.session.provider(model.provider);
        Ok(Some(decoding))
    }

    fn record(
        &self,
        log: &splinter_sdk::store::jsonl::Log,
        id: &str,
        decoding: Option<&Decoding>,
        exchange: &persona_chat::session::Exchange,
    ) -> anyhow::Result<()> {
        trace::record(log, id, &self.model, decoding, exchange)?;
        Ok(())
    }
}

fn session(common: &Common) -> anyhow::Result<Chat> {
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
    let provider: Arc<dyn ModelProvider> = Arc::clone(&model.provider);
    let source = if reference.is_remote() {
        Source::Remote
    } else {
        Source::Local(Box::new(Local {
            _runtime: splinter,
            ctx,
            reference,
            seed: splinter_sdk::seed::resolve("seed", common.seed),
        }))
    };
    Ok(Chat {
        session: Session::new(provider, common.system.clone(), tools)
            .rounds(common.rounds)
            .max_tokens(common.max_tokens),
        model: common.model.clone(),
        source,
        exchanges: 0,
    })
}

fn main() -> anyhow::Result<()> {
    let runtime = tokio::runtime::Runtime::new()?;
    let cli = Cli::parse();
    cli.device.apply()?;
    match cli.command {
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
                let decoding = chat.next_exchange()?;
                let exchange = runtime.block_on(chat.session.say(line.trim()))?;
                print!("{}", trace::render(&exchange));
                if let Some(log) = &log {
                    chat.record(log, &format!("chat-{n}"), decoding.as_ref(), &exchange)?;
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
                chat.session.forget();
                let decoding = chat.next_exchange()?;
                let exchange = runtime.block_on(chat.session.say(&prompt))?;
                println!("[{id}]\n{}", trace::render(&exchange));
                if let Some(log) = &log {
                    chat.record(log, &id, decoding.as_ref(), &exchange)?;
                }
            }
            Ok(())
        }
    }
}
