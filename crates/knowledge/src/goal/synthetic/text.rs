// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! How a synthetic corpus words its documents: titles, sections and the
//! sentence that states one fact.

use std::collections::BTreeMap;

use super::{FactKey, Form, Topic};
use crate::goal::corpus::{Document, Provenance};

/// A document's title is its name without the chunk number its id carries
/// (`alquor-manual-3` is "alquor manual"; the 2024 edition keeps its year): the
/// compiler lets documents numbered differently in their titles never compete,
/// and the chunks of one manual are not different documents.
fn title_of(id: &str) -> String {
    let mut parts: Vec<&str> = id.split('-').collect();
    if parts.len() > 1
        && parts
            .last()
            .is_some_and(|p| p.chars().all(|c| c.is_ascii_digit()) && *p != "2024")
    {
        parts.pop();
    }
    parts.join(" ")
}

pub(super) fn document(id: String, provenance: Provenance, text: String) -> Document {
    Document {
        title: title_of(&id),
        id,
        text,
        provenance,
    }
}

pub(super) fn key(form: Form, attribute: &str, entity: &str) -> FactKey {
    FactKey {
        form,
        attribute: attribute.to_string(),
        entity: entity.to_string(),
    }
}

pub(super) fn sentence(key: &FactKey, value: &str) -> String {
    match key.form {
        Form::Property | Form::Link => {
            format!("The {} of {} is {value}.", key.attribute, key.entity)
        }
        Form::Behaviour => format!("When {}, a {} client {value}.", key.attribute, key.entity),
    }
}

/// One unit as a section, with the values `values` gives its keys.
pub(super) fn section(
    topic: &Topic,
    unit: &[FactKey],
    values: &BTreeMap<FactKey, String>,
    prose: &str,
    form: Form,
) -> String {
    let heading = match form {
        Form::Property | Form::Link => unit[0].entity.clone(),
        Form::Behaviour => format!("Observed behaviour of {} clients", topic.name),
    };
    let mut text = format!("## {heading}\n\n");
    text.push_str(&format!(
        "This part of the {} material is about {heading}. {prose}\n",
        topic.name
    ));
    for k in unit {
        text.push_str(&sentence(k, &values[k]));
        text.push('\n');
    }
    text
}
