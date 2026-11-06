// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements comprehension drills made by code from a
// codebase's own functions and tests, for its clients. If your team needs
// expertise in turning a repository into verified training tasks you can
// procure our services by sending an email to info@swedishembedded.com.

//! Lua drills: the functions a module returns and the literal calls the tests
//! make of them, read by the house layout, and the program that calls one.
//!
//! A module is a file that declares `function M.name(params)` at the left
//! margin, closed by an `end` at the left margin, and returns `M`. A test
//! file names the module it tests in one of three ways, each binding a local
//! name to it: `local X = require("stem")`; a Python class with
//! `MODULE = "stem"` whose host helper binds `X = require("...")` with the
//! module as a placeholder; or `modules = {"X": "stem"}`. It calls a function
//! of the module as `X.name(args)` or through a wrapper that takes the
//! function first (`run(X.name, args)`). A call is kept only when every
//! argument is a literal: a number, a string with no escapes, `true`,
//! `false`, `nil`, or a table of literals. A helper a test file defines for
//! itself (a graph builder, a date constructor) is not one, and a drill never
//! carries test code, so calls through them are not inputs.

use std::path::Path;

use super::{Call, FunctionSource, Import, Language, Param, Read};
use crate::tasks::workspace::WorkspaceError;

/// The serialiser a program carries when a result is a table: values print
/// as Lua source would write them, keys in sorted order, so what is printed
/// does not depend on the address or iteration order of a table.
const SHOW: &str = r#"local function show(value)
  if type(value) == "string" then
    return string.format("%q", value)
  elseif type(value) ~= "table" then
    return tostring(value)
  end
  local keys = {}
  for key in pairs(value) do
    keys[#keys + 1] = key
  end
  table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
  local parts = {}
  for _, key in ipairs(keys) do
    parts[#parts + 1] = "[" .. show(key) .. "]=" .. show(value[key])
  end
  return "{" .. table.concat(parts, ", ") .. "}"
end
"#;

/// Reads the module functions of `sources` and the calls `tests` make of
/// them. Every file is read from `tree`.
pub(super) fn read(
    tree: &Path,
    sources: &[String],
    tests: &[String],
) -> Result<Read, WorkspaceError> {
    let mut unreadable = Vec::new();
    let mut texts = Vec::new();
    for test in tests {
        match std::fs::read_to_string(tree.join(test)) {
            Ok(text) => texts.push(text),
            Err(_) => unreadable.push(test.clone()),
        }
    }
    let host = host_aliases(&texts);
    let mut functions = Vec::new();
    for source in sources {
        let Ok(text) = std::fs::read_to_string(tree.join(source)) else {
            unreadable.push(source.clone());
            continue;
        };
        let (dir, stem) = split(source);
        let mut found = declared(source, &text);
        for function in &mut found {
            function.import = Some(Import {
                root: dir.to_string(),
                module: stem.to_string(),
            });
            for test in &texts {
                if stems_of(test).iter().any(|named| named == stem) {
                    function
                        .calls
                        .extend(calls_in(test, stem, &function.name, &host));
                }
            }
        }
        functions.extend(found);
    }
    Ok(Read {
        functions,
        unreadable,
    })
}

/// The directory and the module name (the file's stem) of a Lua file.
fn split(path: &str) -> (&str, &str) {
    let (dir, file) = path.rsplit_once('/').unwrap_or(("", path));
    (dir, file.trim_end_matches(".lua"))
}

/// The `function M.name(params)` declarations of a module, with the lines
/// each spans. `M` is whatever table the file returns.
fn declared(file: &str, text: &str) -> Vec<FunctionSource> {
    let lines: Vec<&str> = text.lines().collect();
    let Some(table) = lines.iter().rev().find_map(|line| {
        line.strip_prefix("return ").map(str::trim).filter(|name| {
            !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_')
        })
    }) else {
        return Vec::new();
    };
    let prefix = format!("function {table}.");
    let mut found = Vec::new();
    for (at, line) in lines.iter().enumerate() {
        let Some(rest) = line.strip_prefix(&prefix) else {
            continue;
        };
        let Some((name, after)) = rest.split_once('(') else {
            continue;
        };
        let Some((params, _)) = after.split_once(')') else {
            continue;
        };
        let Some(end) = lines[at..].iter().position(|l| *l == "end") else {
            continue;
        };
        found.push(FunctionSource {
            file: file.to_string(),
            name: name.trim().to_string(),
            language: Language::Lua,
            line_start: at + 1,
            line_end: at + end + 1,
            params: params
                .split(',')
                .map(str::trim)
                .filter(|p| !p.is_empty())
                .map(|p| Param {
                    name: p.to_string(),
                    variadic: p == "...",
                    // A Lua function reads what it is given: a drill with an
                    // argument fewer than the function takes is not a call
                    // anyone's tests make, so none has a default.
                    has_default: false,
                })
                .collect(),
            import: None,
            calls: Vec::new(),
        });
    }
    found
}

/// The literal calls the test `text` makes of `function` of the module
/// `stem`, through the names it (or the `host` helpers) binds to it.
fn calls_in(text: &str, stem: &str, function: &str, host: &[String]) -> Vec<Call> {
    let mut found = Vec::new();
    for alias in aliases(text, stem, host) {
        // `X.name(args)`, and `wrapper(X.name, args)`.
        for (suffix, wrapper) in [("(", false), (",", true)] {
            let pattern = format!("{alias}.{function}{suffix}");
            for (at, _) in text.match_indices(&pattern) {
                let head = &text[..at];
                let alone = head
                    .chars()
                    .next_back()
                    .is_none_or(|c| !(c.is_alphanumeric() || c == '_' || c == '.'));
                if !alone || (wrapper && !head.trim_end().ends_with('(')) {
                    continue;
                }
                let arguments = balanced(text, at + pattern.len())
                    .as_deref()
                    .and_then(literal_args);
                found.extend(arguments);
            }
        }
    }
    found
}

/// The quoted words that follow `marker` in `text`: `MODULE = "stem"` gives
/// `stem`.
fn quoted_after<'a>(text: &'a str, marker: &str) -> Vec<&'a str> {
    text.match_indices(marker)
        .filter_map(|(at, _)| {
            let rest = text[at + marker.len()..].trim_start();
            let quote = rest.chars().next().filter(|c| matches!(c, '"' | '\''))?;
            let body = &rest[1..];
            body.split_once(quote).map(|(word, _)| word)
        })
        .collect()
}

/// The names a host helper binds to a module chosen at run time: the `X` of
/// `X = require("%s")` or `local X = require("{self.MODULE}")`, a `require`
/// whose argument is not a plain module name.
fn host_aliases(texts: &[String]) -> Vec<String> {
    let mut names = Vec::new();
    for text in texts {
        for (at, _) in text.match_indices("require(") {
            let Some(head) = text[..at].trim_end().strip_suffix('=') else {
                continue;
            };
            let name: String = head
                .trim_end()
                .chars()
                .rev()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            let argument = text[at + "require(".len()..]
                .split(')')
                .next()
                .unwrap_or("");
            let plain = argument
                .trim()
                .trim_matches(['"', '\''])
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_');
            if !name.is_empty() && !plain && !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
}

/// The modules a test file tests: the stems it names by `MODULE = "stem"`, by
/// `require("stem")` and in a `modules = {"X": "stem"}` table.
fn stems_of(text: &str) -> Vec<String> {
    let mut stems: Vec<String> = quoted_after(text, "MODULE =")
        .into_iter()
        .chain(quoted_after(text, "require("))
        .map(str::to_string)
        .collect();
    stems.extend(table_pairs(text).into_iter().map(|(_, stem)| stem));
    stems.sort();
    stems.dedup();
    stems
}

/// The `"X": "stem"` pairs of a `modules = {...}` table.
fn table_pairs(text: &str) -> Vec<(String, String)> {
    let Some(at) = text.find("modules = {") else {
        return Vec::new();
    };
    let Some(inner) = text[at + "modules = {".len()..].split('}').next() else {
        return Vec::new();
    };
    inner
        .split(',')
        .filter_map(|pair| {
            let (name, stem) = pair.split_once(':')?;
            let unquote = |word: &str| word.trim().trim_matches(['"', '\'']).to_string();
            Some((unquote(name), unquote(stem)))
        })
        .filter(|(name, stem)| !name.is_empty() && !stem.is_empty())
        .collect()
}

/// The names `text` binds the module `stem` to: any `local X =
/// require("stem")`, the keys of its `modules` table that name `stem`, and,
/// for a test that names its module by `MODULE`, the host helpers' names.
fn aliases(text: &str, stem: &str, host: &[String]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    if quoted_after(text, "MODULE =").contains(&stem) {
        names.extend(host.iter().cloned());
    }
    names.extend(
        table_pairs(text)
            .into_iter()
            .filter(|(_, module)| module == stem)
            .map(|(name, _)| name),
    );
    for quote in ["\"", "'"] {
        let required = format!("= require({quote}{stem}{quote})");
        for (at, _) in text.match_indices(&required) {
            let name = text[..at].trim_end().rsplit(char::is_whitespace).next();
            names.extend(name.filter(|n| !n.is_empty()).map(str::to_string));
        }
    }
    names.sort();
    names.dedup();
    names
}

/// The text from `from` to the parenthesis that closes the one before it.
fn balanced(text: &str, from: usize) -> Option<String> {
    let mut depth = 0usize;
    let mut quote: Option<char> = None;
    for (offset, c) in text[from..].char_indices() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), _) => {}
            (None, '\'' | '"') => quote = Some(c),
            (None, '(' | '{' | '[') => depth += 1,
            (None, '}' | ']') => depth = depth.checked_sub(1)?,
            (None, ')') => {
                if depth == 0 {
                    return Some(text[from..from + offset].to_string());
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

/// The call whose arguments are `text`, when each is a literal.
fn literal_args(text: &str) -> Option<Call> {
    let args = split_top_level(text)?;
    if args.iter().all(|arg| is_literal(arg)) {
        Some(Call {
            args,
            kwargs: Vec::new(),
        })
    } else {
        None
    }
}

/// `text` split at the commas outside quotes and brackets, each part trimmed.
/// An empty text has no arguments; an empty argument between commas is no
/// call.
fn split_top_level(text: &str) -> Option<Vec<String>> {
    if text.trim().is_empty() {
        return Some(Vec::new());
    }
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut quote: Option<char> = None;
    let mut start = 0;
    for (at, c) in text.char_indices() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), _) => {}
            (None, '\'' | '"') => quote = Some(c),
            (None, '(' | '{' | '[') => depth += 1,
            (None, ')' | '}' | ']') => depth = depth.checked_sub(1)?,
            (None, ',') if depth == 0 => {
                parts.push(text[start..at].trim().to_string());
                start = at + 1;
            }
            _ => {}
        }
    }
    parts.push(text[start..].trim().to_string());
    parts.iter().all(|p| !p.is_empty()).then_some(parts)
}

/// Whether `text` is a Lua literal a drill may pass: see the module
/// documentation.
fn is_literal(text: &str) -> bool {
    let text = text.trim();
    if text.contains('\\') || text.contains('\n') || text.len() > 200 {
        return false;
    }
    if matches!(text, "true" | "false" | "nil") {
        return true;
    }
    if is_number(text) {
        return true;
    }
    for quote in ['\'', '"'] {
        if text.len() >= 2
            && text.starts_with(quote)
            && text.ends_with(quote)
            && !text[1..text.len() - 1].contains(quote)
        {
            return true;
        }
    }
    let Some(inner) = text.strip_prefix('{').and_then(|t| t.strip_suffix('}')) else {
        return false;
    };
    let Some(items) = split_top_level(inner) else {
        return false;
    };
    items.iter().all(|item| match item.split_once('=') {
        // `key = literal`, but not a comparison or a string holding `=`.
        Some((key, value))
            if !item.starts_with(['\'', '"', '{'])
                && !value.starts_with('=')
                && key.trim().chars().all(|c| c.is_alphanumeric() || c == '_')
                && !key.trim().is_empty() =>
        {
            is_literal(value)
        }
        _ => is_literal(item),
    })
}

fn is_number(text: &str) -> bool {
    let body = text.strip_prefix('-').unwrap_or(text);
    !body.is_empty()
        && body.chars().next().is_some_and(|c| c.is_ascii_digit())
        && body.parse::<f64>().is_ok()
}

/// The program that calls `function` as `call` says: `print` of whether the
/// call succeeded and what it returned or the error it raised. With `show`,
/// the result is printed by [`SHOW`].
pub(super) fn program(function: &FunctionSource, call: &Call, show: bool) -> String {
    let (root, module) = match &function.import {
        Some(import) => (import.root.as_str(), import.module.as_str()),
        None => ("", ""),
    };
    let path = if root.is_empty() {
        "?.lua".to_string()
    } else {
        format!("{root}/?.lua")
    };
    // A module named like an identifier is bound to its own name; any other
    // (a stem with a hyphen) to `lib`.
    let bound = if module.chars().all(|c| c.is_alphanumeric() || c == '_')
        && !module.starts_with(|c: char| c.is_ascii_digit())
    {
        module
    } else {
        "lib"
    };
    let mut text = format!(
        "package.path = {path:?} .. \";\" .. package.path\nlocal {bound} = require({module:?})\n"
    );
    let invocation = format!("{bound}.{}, {}", function.name, call.text());
    let invocation = invocation.trim_end_matches([',', ' ']).to_string();
    if show {
        text.push_str(SHOW);
        text.push_str(&format!(
            "local ok, result = pcall({invocation})\nprint(ok, show(result))\n"
        ));
    } else {
        text.push_str(&format!("print(pcall({invocation}))\n"));
    }
    text
}
