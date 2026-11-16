// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn an interpreter's own
// execution of a program into reasoning a model can be trained on, for its
// clients. If your team needs expertise in verified-by-construction training
// signals for code models you can procure our services by sending an email to
// info@swedishembedded.com.

//! Deterministic execution traces of Lua 5.1 programs: the interpreter runs the
//! program under `debug.sethook` and reports what each executed line did, so a
//! trace is the program's own reasoning, verified by construction.
//!
//! Two renderings of the same events ([`Style`]):
//!
//! - `raw`: per executed line `[n] <source>`, then the locals and upvalues the
//!   line changed (`name=value`, an upvalue as `^name=value`), a call into a
//!   program function as `call name`, a return as `return from name`, and
//!   every printed line as `out: <text>`.
//! - `steps`: the same events as short sentences (`Line 3 sets x to 5.`,
//!   `It prints "9".`), the style a model writes when it reasons.
//!
//! Rules that keep a trace short and identical across runs:
//!
//! - Values: numbers as Lua prints them, strings quoted with `\n`, `\t`, `\"`
//!   and `\\` escaped and cut at 40 characters (`...` marks a cut), a table as
//!   `{a,b,c}` when it is a sequence (at most 8 elements) or `{k=v,...}` with
//!   its keys in sorted order (at most 6), a nested table as `{...}`, a
//!   function as `function`. No address is ever printed.
//! - Loops: a source line is shown the first [`LINE_REPEATS`] times it runs;
//!   later runs are counted and reported once, as `(... [n] xK more)`, before
//!   the next line that is shown. Printed output is always shown.
//! - Budget: the interpreter stops a program after [`INSTRUCTION_BUDGET`] VM
//!   instructions, and a trace longer than the caller's `max_chars` is not
//!   returned ([`Traced::TooLong`]): it is skipped and counted, never cut.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};

/// How many times a source line is shown before its runs are only counted.
pub const LINE_REPEATS: u32 = 3;

/// VM instructions a traced program may execute.
pub const INSTRUCTION_BUDGET: u64 = 2_000_000;

/// The tracer the interpreter runs; the program arrives on its standard input.
const TRACER: &str = include_str!("exec_trace.lua");

/// The rendering of a trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Style {
    /// Source lines and the values they changed.
    Raw,
    /// The same events as short sentences.
    Steps,
}

impl Style {
    fn name(self) -> &'static str {
        match self {
            Style::Raw => "raw",
            Style::Steps => "steps",
        }
    }
}

/// What tracing one program gave.
#[derive(Debug, PartialEq, Eq)]
pub enum Traced {
    /// The trace and what the program printed.
    Fits {
        /// The rendered trace.
        trace: String,
        /// The program's output, every printed line.
        output: String,
    },
    /// The trace is longer than the budget, in characters.
    TooLong {
        /// Its length.
        chars: usize,
    },
    /// The program raised an error or ran out of its instruction budget.
    Failed {
        /// The interpreter's message.
        reason: String,
    },
}

/// Traces `program` with the Lua 5.1 interpreter at `lua`, rendered in `style`,
/// keeping it only when it is at most `max_chars` characters long.
pub fn trace(lua: &Path, program: &str, style: Style, max_chars: usize) -> Result<Traced> {
    let mut child = Command::new(lua)
        .arg("-e")
        .arg(format!(
            "STYLE={:?} BUDGET={INSTRUCTION_BUDGET} REPEATS={LINE_REPEATS}",
            style.name()
        ))
        .arg("-e")
        .arg(TRACER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("running {}", lua.display()))?;
    child
        .stdin
        .take()
        .context("the interpreter's standard input")?
        .write_all(program.as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!(
            "the tracer failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let text = String::from_utf8(out.stdout).context("the trace is not UTF-8")?;
    parse(&text, max_chars)
}

/// The tracer writes `TRACE <bytes>` and the trace, `OUTPUT <bytes>` and the
/// output, then `END ok` or `END <reason>`: sections by length, so nothing a
/// program prints can be read as a section mark.
fn parse(text: &str, max_chars: usize) -> Result<Traced> {
    let (trace, rest) =
        section(text, "TRACE ").with_context(|| format!("the tracer wrote no trace: {text:?}"))?;
    let (output, rest) = section(rest, "OUTPUT ")
        .with_context(|| format!("the tracer wrote no output: {text:?}"))?;
    let status = rest
        .strip_prefix("END ")
        .with_context(|| format!("the tracer wrote no status: {text:?}"))?
        .trim();
    if status != "ok" {
        return Ok(Traced::Failed {
            reason: status.to_string(),
        });
    }
    let chars = trace.chars().count();
    if chars > max_chars {
        return Ok(Traced::TooLong { chars });
    }
    Ok(Traced::Fits {
        trace: trace.to_string(),
        output: output.to_string(),
    })
}

/// The `<mark><bytes>\n` section at the start of `text` and what follows it.
fn section<'a>(text: &'a str, mark: &str) -> Option<(&'a str, &'a str)> {
    let rest = text.strip_prefix(mark)?;
    let (len, rest) = rest.split_once('\n')?;
    let len: usize = len.parse().ok()?;
    (rest.is_char_boundary(len) && len <= rest.len()).then(|| rest.split_at(len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The Lua 5.1 interpreter on the path, as the sample's other tests use it.
    fn lua() -> PathBuf {
        crate::default_interpreter()
    }

    fn fits(program: &str, style: Style) -> (String, String) {
        match trace(&lua(), program, style, 100_000).unwrap() {
            Traced::Fits { trace, output } => (trace, output),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_trace_shows_each_line_the_values_it_set_and_what_it_printed() {
        let (trace, output) = fits("local x = 2\nlocal y = x * 3\nprint(y + 1)\n", Style::Raw);
        assert_eq!(trace, "[1] local x = 2\n    x=2\n[2] local y = x * 3\n    y=6\n[3] print(y + 1)\n    out: 7\n");
        assert_eq!(output, "7\n");
    }

    #[test]
    fn the_same_program_gives_the_same_trace_byte_for_byte() {
        let program = "local t = {b=2, a=1, c={1}}\nlocal s = ''\nfor k, v in pairs(t) do s = s .. k end\nprint(#s)\n";
        assert_eq!(fits(program, Style::Raw), fits(program, Style::Raw));
        let (trace, _) = fits(program, Style::Raw);
        assert!(trace.contains("t={a=1,b=2,c={...}}"), "{trace}");
        assert!(!trace.contains("0x"), "no address is printed: {trace}");
    }

    #[test]
    fn a_line_run_more_than_the_repeat_limit_is_counted_not_shown() {
        let (trace, output) = fits(
            "local s = 0\nfor i = 1, 10 do\n  s = s + i\nend\nprint(s)\n",
            Style::Raw,
        );
        assert_eq!(
            trace.matches("[3]   s = s + i").count(),
            LINE_REPEATS as usize,
            "{trace}"
        );
        // The `for` line runs once more than its body: the check that ends the loop.
        assert!(
            trace.contains(
                "(... [2] x8 more, [3] x7 more)\n    s=55 i=10\n[5] print(s)\n    out: 55\n"
            ),
            "{trace}"
        );
        assert_eq!(output, "55\n");
    }

    #[test]
    fn calls_returns_upvalues_and_strings_are_rendered() {
        let program = "local n = 0\nlocal function bump(by)\n  n = n + by\n  return n\nend\nbump(2)\nprint(bump(3), \"a\\nb\")\n";
        let (trace, output) = fits(program, Style::Raw);
        assert!(trace.contains("call bump"), "{trace}");
        assert!(trace.contains("by=2"), "{trace}");
        assert!(trace.contains("^n=2"), "{trace}");
        assert!(trace.contains("return from bump"), "{trace}");
        assert!(
            trace.ends_with("return from bump\n    out: 5\ta\n    out: b\n"),
            "a call inside print returns before the line is printed: {trace}"
        );
        assert_eq!(output, "5\ta\nb\n");
    }

    #[test]
    fn io_write_is_output_like_print_and_a_partial_line_is_kept() {
        let (trace, output) = fits(
            "io.write('a', 1, ' ')\nio.stdout:write('b\\n')\nio.write('c')\n",
            Style::Raw,
        );
        assert_eq!(output, "a1 b\nc");
        assert!(trace.contains("    out: a1 b\n"), "{trace}");
        assert!(trace.ends_with("    out: c\n"), "{trace}");
    }

    #[test]
    fn the_steps_style_says_the_same_events_in_sentences() {
        let (trace, _) = fits("local x = 2\nprint(x + 1)\n", Style::Steps);
        assert_eq!(
            trace,
            "Line 1: local x = 2. Now x is 2.\nLine 2: print(x + 1). It prints \"3\".\n"
        );
        // A local the interpreter shows before the first line has run has no line to belong to yet.
        let (trace, _) = fits("local x = nil\nlocal y = x\nprint(y)\n", Style::Steps);
        assert!(trace.ends_with("It prints \"nil\".\n"), "{trace}");
    }

    #[test]
    fn a_trace_past_the_budget_is_skipped_and_an_endless_program_is_stopped() {
        let program = "local s = ''\nfor i = 1, 40 do s = s .. i end\nprint(s)\n";
        assert!(matches!(
            trace(&lua(), program, Style::Raw, 50).unwrap(),
            Traced::TooLong { .. }
        ));
        match trace(&lua(), "while true do end\n", Style::Raw, 100_000).unwrap() {
            Traced::Failed { reason } => assert!(reason.contains("budget"), "{reason}"),
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            trace(&lua(), "error('boom')\n", Style::Raw, 100_000).unwrap(),
            Traced::Failed { .. }
        ));
    }
}
