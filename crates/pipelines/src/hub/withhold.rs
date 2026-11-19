// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What is never published, as a typed check run before anything is staged
//! or sent: the examiner's private material, item-level examination files,
//! sealed exports, examination experiences, experiences of a sealed family,
//! and any verdict that is not a signed aggregate one.
//!
//! Every refusal is a [`Withheld`] naming what and why. A file is judged by
//! its path (and the path its links resolve to) before it is read, so the
//! examiner's private directory is never opened to find out.

use std::path::{Component, Path, PathBuf};

use serde::Serialize;
use serde_json::Value;
use splinter_core::experience::{Experience, ExperienceId};
use splinter_eval::promotion::verdict::{AGENT_SCHEMA, EXAMINATION_SCHEMA};
use splinter_eval::promotion::{verify_signature, PublicKey};

use crate::bench::split::{self, TrialRecord};
use crate::sealed::SealedFamily;

/// The examiner's private directory: sealed items, keys, exam runs.
pub const PRIVATE_DIR: &str = "bench-private";

/// The file the examiner leaves beside an export of sealed material.
pub const SEALED_EXPORT: &str = "sealed-export.json";

/// Fields that hold per-item records; a verdict carrying one under any of
/// these names is item-level, whatever else it holds.
pub const ITEM_LEVEL_FIELDS: [&str; 5] = ["items", "trials", "answers", "per_item", "responses"];

/// Why something is never published.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, thiserror::Error)]
#[serde(tag = "withheld", rename_all = "snake_case")]
pub enum Withheld {
    /// The file lies under the examiner's private directory.
    #[error("{} is under the examiner's private material ({PRIVATE_DIR}): never published", path.display())]
    PrivatePath {
        /// The file.
        path: PathBuf,
    },
    /// The file is an item-level file of an examination.
    #[error("{} is an examination's item-level file: never published", path.display())]
    ItemsFile {
        /// The file.
        path: PathBuf,
    },
    /// The file lies beside a sealed export.
    #[error("{} lies beside a sealed export ({SEALED_EXPORT}): sealed material is never published", path.display())]
    SealedExport {
        /// The file.
        path: PathBuf,
    },
    /// The experience is examination material.
    #[error("experience {experience} is examination material ({reason}): never published")]
    ExamMaterial {
        /// The experience.
        experience: ExperienceId,
        /// What makes it so.
        reason: String,
    },
    /// The experience is of a sealed final-exam family.
    #[error("experience {experience} is of the sealed family {family}: never published")]
    SealedFamily {
        /// The experience.
        experience: ExperienceId,
        /// The family.
        family: String,
    },
    /// The verdict carries per-item records.
    #[error("verdict {} carries the item-level field {field:?}: only aggregate verdicts are published", path.display())]
    ItemLevelVerdict {
        /// The verdict's file.
        path: PathBuf,
        /// The field.
        field: String,
    },
    /// The verdict is not the examiner's signed verdict of a schema the gate
    /// reads.
    #[error("{} is not a signed aggregate verdict of the examiner: {reason}", path.display())]
    UnverifiedVerdict {
        /// The verdict's file.
        path: PathBuf,
        /// Why.
        reason: String,
    },
}

fn names_private(path: &Path) -> bool {
    path.components()
        .any(|c| matches!(c, Component::Normal(name) if name == PRIVATE_DIR))
}

/// Whether the file at `path` may be published, judged by its path, the path
/// its links resolve to, its name and its directory, before it is read.
///
/// # Errors
/// The [`Withheld`] reason it may not.
pub fn screen_file(path: &Path) -> Result<(), Withheld> {
    let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if names_private(path) || names_private(&resolved) {
        return Err(Withheld::PrivatePath {
            path: path.to_path_buf(),
        });
    }
    let name = resolved
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if name.contains("items") {
        return Err(Withheld::ItemsFile {
            path: path.to_path_buf(),
        });
    }
    if resolved
        .parent()
        .is_some_and(|dir| dir.join(SEALED_EXPORT).exists())
    {
        return Err(Withheld::SealedExport {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

/// Whether `experience` (stored as `id`) may be published under the
/// examiner's trial `register` and the `sealed` final-exam families.
///
/// # Errors
/// The [`Withheld`] reason it may not.
pub fn screen_experience(
    register: &[TrialRecord],
    sealed: &[SealedFamily],
    id: &ExperienceId,
    experience: &Experience,
) -> Result<(), Withheld> {
    if let Some(reason) =
        split::exam_only(register, experience).or_else(|| split::exam_only_id(register, id))
    {
        return Err(Withheld::ExamMaterial {
            experience: id.clone(),
            reason,
        });
    }
    let spec = &experience.environment.spec;
    let tree = spec.get("tree_id").and_then(Value::as_str);
    let snapshot = spec.get("snapshot").and_then(Value::as_str);
    for family in sealed {
        let by_tree = tree == Some(family.tree.as_str());
        let by_snapshot = snapshot.is_some() && snapshot == family.environment.as_deref();
        let by_id = experience.concepts.iter().any(|c| c == &family.id);
        if by_tree || by_snapshot || by_id {
            return Err(Withheld::SealedFamily {
                experience: id.clone(),
                family: family.id.clone(),
            });
        }
    }
    Ok(())
}

/// The first item-level field of `value`, at any depth, holding a list or
/// an object.
fn item_level_field(value: &Value) -> Option<String> {
    match value {
        Value::Object(map) => map.iter().find_map(|(key, inner)| {
            let structured = inner.is_array() || inner.is_object();
            if structured && ITEM_LEVEL_FIELDS.contains(&key.as_str()) {
                Some(key.clone())
            } else {
                item_level_field(inner)
            }
        }),
        Value::Array(items) => items.iter().find_map(item_level_field),
        _ => None,
    }
}

/// Whether the verdict `document` read from `path` may be published: the
/// examiner's signature under `key`, a schema the gate reads, and no
/// item-level field. Returns its schema.
///
/// # Errors
/// The [`Withheld`] reason it may not.
pub fn screen_verdict(path: &Path, document: &Value, key: &PublicKey) -> Result<String, Withheld> {
    let unverified = |reason: String| Withheld::UnverifiedVerdict {
        path: path.to_path_buf(),
        reason,
    };
    verify_signature(document, key).map_err(|e| unverified(e.to_string()))?;
    let schema = document
        .get("schema")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if schema != EXAMINATION_SCHEMA && schema != AGENT_SCHEMA {
        return Err(unverified(format!(
            "its schema is {schema:?}, not {EXAMINATION_SCHEMA} or {AGENT_SCHEMA}"
        )));
    }
    if let Some(field) = item_level_field(document) {
        return Err(Withheld::ItemLevelVerdict {
            path: path.to_path_buf(),
            field,
        });
    }
    Ok(schema.to_owned())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use serde_json::json;

    #[test]
    fn private_paths_and_item_files_are_refused_by_name_alone() {
        let private = Path::new("/nowhere/bench-private/verdict.json");
        assert!(matches!(
            screen_file(private),
            Err(Withheld::PrivatePath { .. })
        ));
        assert!(matches!(
            screen_file(Path::new("/nowhere/exam.items.jsonl")),
            Err(Withheld::ItemsFile { .. })
        ));
        assert!(screen_file(Path::new("/nowhere/verdict.json")).is_ok());
    }

    #[test]
    fn an_item_level_field_is_found_at_any_depth() {
        assert_eq!(
            item_level_field(&json!({"overall": {"items": [1]}})),
            Some("items".into())
        );
        assert_eq!(item_level_field(&json!({"overall": {"items": 3}})), None);
    }
}
