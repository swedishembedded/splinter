// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that acquire knowledge
// their model does not have yet, for its clients. If your team needs
// expertise in knowledge distillation or continual learning, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What a teacher is shown of a task: the material it is grounded in.
//!
//! A task's grounding material is what only the teacher sees and the
//! student must learn to do without, in this order:
//!
//! 1. For each evidence span, the passage it grounds the task in: the whole
//!    section of its source part the span falls in, by the rule that places
//!    a span in a concept ([`crate::concepts`]: the last section starting
//!    at or before its first byte, the first when it starts before any), so
//!    a quote is shown in its context; a span that names no part, or a part
//!    that is not text or holds no section, is shown as its own bytes, when
//!    they are text.
//! 2. Every passage the task carries as a privileged item.
//! 3. Every hint, as `Hint: <hint>`.
//!
//! Each piece is shown once. The reference answer is never grounding
//! material: it is what the answer is graded against, so a teacher that
//! saw it would be graded on copying. Neither are the checks and tests
//! that grade an answer. (Whether a reference is supported by its evidence
//! is a different question, [`crate::tasks::grounding`]'s.)

use splinter_core::experience::{PrivilegedKind, Span, Task};
use splinter_record::error::StoreError;
use splinter_record::sources::SourceStore;

use crate::sections::sections;

/// The label a hint is shown under.
pub const HINT_LABEL: &str = "Hint:";

/// `task`'s grounding material, each piece once, in the module's order;
/// empty for a task grounded in nothing a teacher could be shown.
pub fn teacher_material(store: &SourceStore, task: &Task) -> Result<Vec<String>, StoreError> {
    let mut material: Vec<String> = Vec::new();
    let mut add = |piece: String| {
        if !piece.trim().is_empty() && !material.contains(&piece) {
            material.push(piece);
        }
    };
    for span in &task.evidence {
        if let Some(passage) = passage(store, span)? {
            add(passage);
        }
    }
    for item in &task.privileged {
        match item.kind {
            PrivilegedKind::Passage => add(item.content.clone()),
            PrivilegedKind::Hint => add(format!("{HINT_LABEL} {}", item.content)),
            _ => {}
        }
    }
    Ok(material)
}

/// The passage `span` grounds a task in; `None` when it is not text.
fn passage(store: &SourceStore, span: &Span) -> Result<Option<String>, StoreError> {
    if let Some(part) = &span.part {
        let source = store.get_source(&part.source)?;
        let found = source
            .part(&part.name)
            .ok_or_else(|| StoreError::UnknownPart {
                source_id: part.source.clone(),
                part: part.name.clone(),
            })?;
        if let Ok(text) = String::from_utf8(store.read_blob(&found.content)?) {
            let start = usize::try_from(span.start).unwrap_or(usize::MAX);
            let all = sections(&text, &found.media_type);
            let section = all
                .iter()
                .rfind(|s| s.range.start <= start)
                .or(all.first())
                .and_then(|s| s.text(&text).map(str::to_string));
            if section.is_some() {
                return Ok(section);
            }
        }
    }
    Ok(String::from_utf8(store.read_span(span)?).ok())
}
