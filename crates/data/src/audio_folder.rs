// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for curating speech recordings into
// training data for its clients. If your team needs expertise in voice
// datasets and their checks, you can procure our services by sending an email
// to info@swedishembedded.com.

//! An audio dataset in the Hugging Face audio-folder layout: a
//! `metadata.jsonl` whose every line is one recording, its `file_name`
//! relative to the metadata file's directory, beside whatever other columns
//! the dataset keeps (the transcript among them).
//!
//! Rows are read and written back with their columns in the order they came,
//! so a tool that adds a column changes that column and nothing else. Every
//! line must be a JSON object with a `file_name` that stays inside the
//! dataset; anything else is refused naming the line.

use std::fmt;
use std::path::{Component, Path, PathBuf};

use serde::de::{Deserializer, MapAccess, Visitor};
use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One recording's columns, in their order in the file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Row(Vec<(String, Value)>);

impl Row {
    /// The value of `column`, if the row has it.
    #[must_use]
    pub fn get(&self, column: &str) -> Option<&Value> {
        self.0.iter().find(|(k, _)| k == column).map(|(_, v)| v)
    }

    /// The text of `column`, if it is a string.
    #[must_use]
    pub fn text(&self, column: &str) -> Option<&str> {
        self.get(column).and_then(Value::as_str)
    }

    /// Set `column` to `value`: in place when the row has it, else last.
    pub fn set(&mut self, column: &str, value: Value) {
        match self.0.iter_mut().find(|(k, _)| k == column) {
            Some((_, v)) => *v = value,
            None => self.0.push((column.to_string(), value)),
        }
    }
}

impl Serialize for Row {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for Row {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Ordered;
        impl<'de> Visitor<'de> for Ordered {
            type Value = Row;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Row, A::Error> {
                let mut columns = Vec::new();
                while let Some((k, v)) = access.next_entry::<String, Value>()? {
                    if columns.iter().any(|(seen, _): &(String, Value)| *seen == k) {
                        return Err(serde::de::Error::custom(format!("column '{k}' twice")));
                    }
                    columns.push((k, v));
                }
                Ok(Row(columns))
            }
        }
        deserializer.deserialize_map(Ordered)
    }
}

/// A `metadata.jsonl` and its rows.
#[derive(Debug)]
pub struct AudioFolder {
    metadata: PathBuf,
    rows: Vec<Row>,
}

impl AudioFolder {
    /// The audio folder whose metadata is `metadata`.
    ///
    /// # Errors
    /// The file cannot be read, a line is not a JSON object, or a row's
    /// `file_name` is missing, not a string, or leads outside the dataset.
    pub fn open(metadata: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(metadata)
            .map_err(|e| format!("{}: {e}", metadata.display()))?;
        let mut rows = Vec::new();
        for (i, line) in text
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty())
        {
            let at = || format!("{} line {}", metadata.display(), i + 1);
            let row: Row = serde_json::from_str(line).map_err(|e| format!("{}: {e}", at()))?;
            let name = row
                .text("file_name")
                .ok_or_else(|| format!("{}: no string 'file_name'", at()))?;
            inside(name).map_err(|e| format!("{}: {e}", at()))?;
            rows.push(row);
        }
        Ok(AudioFolder {
            metadata: metadata.to_path_buf(),
            rows,
        })
    }

    /// The rows, in file order.
    #[must_use]
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// The rows, to change before [`Self::save`].
    pub fn rows_mut(&mut self) -> &mut [Row] {
        &mut self.rows
    }

    /// The recording of `row`: its `file_name` beside the metadata file.
    #[must_use]
    pub fn audio_path(&self, row: &Row) -> PathBuf {
        let dir = self.metadata.parent().unwrap_or_else(|| Path::new("."));
        dir.join(row.text("file_name").unwrap_or_default())
    }

    /// Write the rows back, replacing the file only once the whole of it is
    /// written.
    ///
    /// # Errors
    /// The file cannot be written.
    pub fn save(&self) -> Result<(), String> {
        let mut out = String::new();
        for row in &self.rows {
            out.push_str(&serde_json::to_string(row).map_err(|e| e.to_string())?);
            out.push('\n');
        }
        let tmp = self.metadata.with_extension("jsonl.tmp");
        std::fs::write(&tmp, out).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &self.metadata)
            .map_err(|e| format!("{}: {e}", self.metadata.display()))
    }
}

/// `name` as a path inside the dataset: relative, with no `..` or root.
fn inside(name: &str) -> Result<(), String> {
    let path = Path::new(name);
    if name.is_empty()
        || !path
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
    {
        return Err(format!(
            "file_name '{name}' is not a path inside the dataset"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folder(lines: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metadata.jsonl");
        std::fs::write(&path, lines).unwrap();
        (dir, path)
    }

    #[test]
    fn a_column_is_added_and_changed_without_disturbing_the_others() {
        let (_dir, path) = folder("{\"id\": \"a\", \"file_name\": \"wavs/a.wav\", \"text\": \"Hej\", \"zeta\": 1}\n\n{\"file_name\": \"b.wav\", \"text\": \"Hi\"}\n");
        let mut f = AudioFolder::open(&path).unwrap();
        assert_eq!(
            f.audio_path(&f.rows()[0]),
            path.parent().unwrap().join("wavs/a.wav")
        );
        f.rows_mut()[0].set("verdict", Value::from("pass"));
        f.rows_mut()[0].set("text", Value::from("Hej då"));
        f.save().unwrap();
        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            written.lines().next().unwrap(),
            r#"{"id":"a","file_name":"wavs/a.wav","text":"Hej då","zeta":1,"verdict":"pass"}"#
        );
        assert_eq!(written.lines().count(), 2);
    }

    #[test]
    fn a_row_without_a_file_or_with_one_outside_the_dataset_is_refused() {
        for bad in [
            "{\"text\": \"x\"}",
            "{\"file_name\": \"../secret.wav\"}",
            "{\"file_name\": \"/etc/passwd\"}",
            "[1, 2]",
            "{\"file_name\": \"a\", \"file_name\": \"b\"}",
        ] {
            let (_dir, path) = folder(bad);
            let err = AudioFolder::open(&path).unwrap_err();
            assert!(err.contains("line 1"), "{bad}: {err}");
        }
    }
}
