// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tool-use task generators that teach agents
// skills rather than templates, for its clients. If your team needs
// expertise in training data for agents that read many file formats, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The file formats a task's keyed data is written in, each with its
//! renderer and parser, so the expert reads a value out of whatever a tool
//! showed it and the verifier reads a file's state back.

use serde::{Deserialize, Serialize};

/// How a file of key-value entries is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    /// `key: value` lines.
    Colon,
    /// `key=value` lines.
    Equals,
    /// A `key,value` header, then rows.
    Csv,
    /// A `key<TAB>value` header, then rows.
    Tsv,
    /// One JSON object.
    Json,
    /// `key -> value` lines.
    Arrow,
    /// An `[entries]` section of `key = value` lines.
    Ini,
}

/// The formats training data uses.
pub const TRAINING: [Format; 5] = [
    Format::Colon,
    Format::Equals,
    Format::Csv,
    Format::Tsv,
    Format::Json,
];

/// The formats only the held-out probe uses.
pub const HELD_OUT: [Format; 2] = [Format::Arrow, Format::Ini];

/// File extensions training data gives a file of any format, beside the
/// format's own: a name is copied from the request, never inferred from the
/// format, so an extension never seen (the held-out probe's `map`, `ini`) is
/// copied like any other. None of them is a held-out format's extension.
pub const TRAINING_EXTENSIONS: [&str; 14] = [
    "txt", "cfg", "csv", "tsv", "json", "dat", "conf", "list", "tbl", "rec", "db", "kv", "env",
    "data",
];

impl Format {
    /// A file extension that fits the format.
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::Colon => "txt",
            Self::Equals => "cfg",
            Self::Csv => "csv",
            Self::Tsv => "tsv",
            Self::Json => "json",
            Self::Arrow => "map",
            Self::Ini => "ini",
        }
    }

    /// `entries` written in this format.
    #[must_use]
    pub fn render(self, entries: &[(String, String)]) -> String {
        let lines = |f: &dyn Fn(&str, &str) -> String| -> String {
            entries.iter().map(|(k, v)| f(k, v)).collect()
        };
        match self {
            Self::Colon => lines(&|k, v| format!("{k}: {v}\n")),
            Self::Equals => lines(&|k, v| format!("{k}={v}\n")),
            Self::Csv => format!("key,value\n{}", lines(&|k, v| format!("{k},{v}\n"))),
            Self::Tsv => format!("key\tvalue\n{}", lines(&|k, v| format!("{k}\t{v}\n"))),
            Self::Arrow => lines(&|k, v| format!("{k} -> {v}\n")),
            Self::Ini => format!("[entries]\n{}", lines(&|k, v| format!("{k} = {v}\n"))),
            Self::Json => {
                let map: serde_json::Map<String, serde_json::Value> = entries
                    .iter()
                    .map(|(k, v)| {
                        let value = v.parse::<u64>().map_or_else(
                            |_| serde_json::Value::String(v.clone()),
                            serde_json::Value::from,
                        );
                        (k.clone(), value)
                    })
                    .collect();
                format!(
                    "{}\n",
                    crate::template::to_json(&serde_json::Value::Object(map))
                )
            }
        }
    }

    /// The entries of `text` written in this format, in order; lines that
    /// are not entries are skipped.
    #[must_use]
    pub fn parse(self, text: &str) -> Vec<(String, String)> {
        let split = |line: &str, sep: &str| {
            line.split_once(sep)
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        };
        let pairs = |sep: &'static str, skip_header: bool| -> Vec<(String, String)> {
            text.lines()
                .skip(usize::from(skip_header))
                .filter_map(|l| split(l, sep))
                .filter(|(k, _)| !k.is_empty())
                .collect()
        };
        match self {
            Self::Colon => pairs(":", false),
            Self::Equals => pairs("=", false),
            Self::Csv => pairs(",", true),
            Self::Tsv => pairs("\t", true),
            Self::Arrow => pairs("->", false),
            Self::Ini => pairs("=", true),
            Self::Json => serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(text)
                .map(|m| {
                    m.into_iter()
                        .map(|(k, v)| {
                            let v = v.as_str().map_or_else(|| v.to_string(), String::from);
                            (k, v)
                        })
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

/// A file's text as a read tool showed it, without its `L<n>:` prefixes.
#[must_use]
pub fn unnumbered(result: &str) -> String {
    let mut out = String::new();
    for line in result.lines() {
        let text = line
            .strip_prefix('L')
            .and_then(|rest| rest.split_once(':'))
            .filter(|(n, _)| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
            .map_or(line, |(_, text)| text);
        out.push_str(text);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_format_reads_back_what_it_wrote_even_through_a_numbered_read() {
        let entries: Vec<(String, String)> =
            [("alpha", "17"), ("beta", "gamma"), ("delta", "4021")]
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect();
        for format in TRAINING.iter().chain(&HELD_OUT) {
            let text = format.render(&entries);
            assert_eq!(format.parse(&text), entries, "{format:?}");
            let numbered: String = text
                .lines()
                .enumerate()
                .map(|(i, l)| format!("L{}:{l}\n", i + 1))
                .collect();
            assert_eq!(
                format.parse(&unnumbered(&numbered)),
                entries,
                "{format:?} read"
            );
        }
    }
}
