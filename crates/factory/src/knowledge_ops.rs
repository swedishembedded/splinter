// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn documents into grounded,
// checkable knowledge for training and evaluation, for its clients. If your
// team needs expertise in knowledge extraction you can trust, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Knowledge operations: documents into sections, and a model's proposed
//! knowledge into what the sections support.
//!
//! * `source.ingest` splits each document of its input into sections (at
//!   headings for Markdown, at blank lines otherwise;
//!   [`splinter_knowledge::sections`]): one item per section, holding its
//!   `text.md` and `source.json` (the document, its title, the section's
//!   number and byte range in it, the document's digest), so whatever is
//!   drawn from a section names where it came from.
//! * `knowledge.admit` holds what an `agent` stage over those sections
//!   proposed (each item's `output.json`, `{ "objects": [...] }` of
//!   [`ProposedObject`]s) to the sections themselves
//!   ([`splinter_knowledge::objects::admit`]): an object stands only when its
//!   quote is in its section verbatim and its data has its kind's keys;
//!   conflicts are flagged, amendments resolved, every refusal kept with its
//!   reason. The product is the knowledge pack (`pack.json`). Admission
//!   relates what every section says: one `source.ingest` over all the
//!   documents and one admission over its sections is how knowledge from
//!   several documents is merged, their conflicts flagged and amendments
//!   resolved in one pack.
//! * `knowledge.query` finds, for each question of a file in the task
//!   repository, the sections that bear on it
//!   ([`splinter_knowledge::retrieve::Bm25`]: lexical, deterministic, rare
//!   words weighing more), best first. Each question is an item holding its
//!   `question.md` and `passages.json` (each passage's section, document,
//!   title, score and text), so a stage over the items reads what the
//!   documents say. The index is built from the sections each time: it is
//!   cheap, and a stored one would be a second copy of them to keep in step.

use std::path::Path;

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_core::digest::Digest;
use splinter_core::factory::execution::Isolation;
use splinter_core::factory::items::{item_id, Item, ItemIndex, ITEMS_FILE};
use splinter_core::factory::operation::{
    ArtifactKind, OperationDescriptor, Port, Reproducibility, OPERATION_SCHEMA,
};
use splinter_knowledge::objects::{admit, Document, ProposedObject};
use splinter_knowledge::retrieve::{Bm25, Passage};
use splinter_knowledge::sections::{sections, title, MARKDOWN};

use crate::execute::{Call, Execute, Outcome};
use crate::registry::OperationSpec;

/// `source.ingest`.
pub const INGEST: &str = "source.ingest";
/// `knowledge.admit`.
pub const ADMIT: &str = "knowledge.admit";
/// `knowledge.query`.
pub const QUERY: &str = "knowledge.query";

/// A section's text in its item.
const TEXT: &str = "text.md";
/// A section's provenance in its item.
const SOURCE: &str = "source.json";

fn port(name: &str, kind: ArtifactKind) -> Port {
    Port {
        name: name.into(),
        kind,
        many: false,
        optional: false,
    }
}

fn no_params(value: &Value) -> Result<(), String> {
    match value.as_object() {
        Some(map) if map.is_empty() => Ok(()),
        _ => Err("takes no parameters".into()),
    }
}

fn descriptor(
    name: &str,
    summary: &str,
    inputs: Vec<Port>,
    output: Port,
    parameters: Value,
) -> OperationDescriptor {
    OperationDescriptor {
        schema: OPERATION_SCHEMA.into(),
        name: name.into(),
        revision: 1,
        summary: summary.into(),
        inputs,
        outputs: vec![output],
        parameters,
        reproducibility: Reproducibility::Deterministic,
        isolation: Isolation::Any,
    }
}

fn empty_schema() -> Value {
    json!({ "type": "object", "additionalProperties": false })
}

fn write(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
}

fn index(out: &Path, items: Vec<Item>) -> Result<(), String> {
    let index = ItemIndex::new(items, Vec::new()).map_err(|e| e.to_string())?;
    write(&out.join(ITEMS_FILE), &index)
}

/// Where a section came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SectionSource {
    /// The document's path in the input.
    pub document: String,
    /// The document's title (its first heading), when it has one.
    pub title: Option<String>,
    /// The section's number in the document, from 1.
    pub section: usize,
    /// The section's byte range in the document.
    pub range: [usize; 2],
    /// The document's digest.
    pub digest: Digest,
}

/// The `source.ingest` operation.
pub struct Ingest;

impl OperationSpec for Ingest {
    fn descriptor(&self) -> OperationDescriptor {
        descriptor(
            INGEST,
            "documents split into sections, each an item naming where it came from",
            vec![port("documents", ArtifactKind::Files)],
            port("sections", ArtifactKind::Files),
            empty_schema(),
        )
    }
    fn check(&self, params: &Value) -> Result<(), String> {
        no_params(params)
    }
    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

/// Every document below `dir`, by its path relative to `root`, in order.
fn documents(
    root: &Path,
    dir: &Path,
    out: &mut Vec<(String, std::path::PathBuf)>,
) -> Result<(), String> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .collect();
    entries.sort();
    for path in entries {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if name.starts_with('.')
            || [ITEMS_FILE, "manifest.json", "stage.json"].contains(&name) && dir == root
        {
            continue;
        }
        if path.is_dir() {
            documents(root, &path, out)?;
        } else {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push((relative, path));
        }
    }
    Ok(())
}

/// An item id for section `n` of the document at `relative`.
fn section_id(relative: &str, n: usize) -> String {
    let stem: String = relative
        .trim_start_matches("files/")
        .chars()
        .map(|c| if c == '/' || c.is_control() { '-' } else { c })
        .collect();
    format!("{}--{n:03}", stem.trim_start_matches('.'))
}

impl Execute for Ingest {
    fn implementation(&self) -> String {
        format!("{INGEST} 1")
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let mut found = Vec::new();
        for root in call.inputs.get("documents").into_iter().flatten() {
            documents(root, root, &mut found)?;
        }
        let mut items = Vec::new();
        for (relative, path) in found {
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let Ok(text) = String::from_utf8(bytes.clone()) else {
                continue; // not a text document
            };
            let media = if relative.ends_with(".md") || relative.ends_with(".markdown") {
                MARKDOWN
            } else {
                "text/plain"
            };
            let heading = title(&text, media);
            for (n, section) in sections(&text, media).iter().enumerate() {
                let Some(body) = section.text(&text).filter(|b| !b.trim().is_empty()) else {
                    continue;
                };
                let id = section_id(&relative, n + 1);
                if !item_id(&id) {
                    return Err(format!("{relative}: its sections cannot be named ({id:?})"));
                }
                let dir = call.out.join(&id);
                std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                std::fs::write(dir.join(TEXT), body).map_err(|e| e.to_string())?;
                write(
                    &dir.join(SOURCE),
                    &SectionSource {
                        document: relative.trim_start_matches("files/").to_owned(),
                        title: heading.clone(),
                        section: n + 1,
                        range: [section.range.start, section.range.end],
                        digest: Digest::of(&bytes),
                    },
                )?;
                items.push(Item {
                    path: id.clone(),
                    id,
                });
            }
        }
        if items.is_empty() {
            return Ok(Outcome::Rejected("no text document to ingest".into()));
        }
        index(&call.out, items)?;
        Ok(Outcome::Produced)
    }
}

/// `knowledge.admit`'s parameters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AdmitParams {
    /// The pack's name.
    pub pack: String,
}

/// The `knowledge.admit` operation.
pub struct Admit;

impl OperationSpec for Admit {
    fn descriptor(&self) -> OperationDescriptor {
        descriptor(
            ADMIT,
            "proposed knowledge held to the sections it quotes: only what they support stands",
            vec![
                port("sections", ArtifactKind::Files),
                port("proposals", ArtifactKind::Files),
            ],
            port("knowledge", ArtifactKind::Knowledge),
            serde_json::to_value(schema_for!(AdmitParams)).unwrap_or(Value::Null),
        )
    }
    fn check(&self, params: &Value) -> Result<(), String> {
        serde_json::from_value::<AdmitParams>(params.clone())
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

fn read_index(view: &Path) -> Result<ItemIndex, String> {
    let path = view.join(ITEMS_FILE);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

impl Execute for Admit {
    fn implementation(&self) -> String {
        format!("{ADMIT} 1")
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let params: AdmitParams =
            serde_json::from_value(call.params.clone()).map_err(|e| e.to_string())?;
        let one = |port: &str| {
            call.inputs
                .get(port)
                .and_then(|v| v.first())
                .ok_or_else(|| format!("{port} is not bound"))
        };
        let (sections_view, proposals_view) = (one("sections")?, one("proposals")?);
        let proposed = read_index(proposals_view)?;
        let mut documents = Vec::new();
        let mut proposals = Vec::new();
        for item in read_index(sections_view)?.items {
            let dir = sections_view.join(&item.path);
            let text = std::fs::read_to_string(dir.join(TEXT)).map_err(|e| e.to_string())?;
            let source: SectionSource = serde_json::from_str(
                &std::fs::read_to_string(dir.join(SOURCE)).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            documents.push(Document {
                id: item.id.clone(),
                title: source.title.unwrap_or_else(|| source.document.clone()),
                text,
                provenance: Default::default(),
            });
            let Some(answered) = proposed.items.iter().find(|p| p.id == item.id) else {
                continue; // the extractor rejected the section, or never saw it
            };
            let output = proposals_view.join(&answered.path).join("output.json");
            let value: Value = serde_json::from_str(
                &std::fs::read_to_string(&output)
                    .map_err(|e| format!("{}: {e}", output.display()))?,
            )
            .map_err(|e| format!("{}: {e}", output.display()))?;
            for (n, object) in value
                .get("objects")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .enumerate()
            {
                let mut proposal: ProposedObject = serde_json::from_value(object.clone())
                    .map_err(|e| format!("{}: object {n}: {e}", output.display()))?;
                // An object rests on the section it was drawn from, under an
                // id unique in the pack.
                proposal.document = item.id.clone();
                proposal.id = format!("{}#{}", item.id, proposal.id);
                proposals.push(proposal);
            }
        }
        let pack = admit(&params.pack, &documents, &proposals);
        write(&call.out.join("pack.json"), &pack)?;
        Ok(Outcome::Produced)
    }
}

fn three() -> usize {
    3
}

/// `knowledge.query`'s parameters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QueryParams {
    /// The questions: a YAML list of strings, a file of the task repository.
    pub questions: String,
    /// The most passages each question is given.
    #[serde(default = "three")]
    pub passages: usize,
}

fn query_params(value: &Value) -> Result<QueryParams, String> {
    serde_json::from_value(value.clone()).map_err(|e| e.to_string())
}

fn questions(text: &str) -> Result<Vec<String>, String> {
    let questions: Vec<String> = serde_yaml::from_str(text).map_err(|e| e.to_string())?;
    if questions.iter().any(|q| q.trim().is_empty()) || questions.is_empty() {
        return Err("lists no question, or an empty one".into());
    }
    Ok(questions)
}

/// One passage given to a question.
#[derive(Serialize)]
struct Found<'a> {
    /// The section's item in the sections product.
    section: &'a str,
    document: &'a str,
    title: Option<&'a str>,
    /// Comparable only among one question's passages.
    score: f64,
    text: &'a str,
}

/// The `knowledge.query` operation.
pub struct Query;

impl OperationSpec for Query {
    fn descriptor(&self) -> OperationDescriptor {
        descriptor(
            QUERY,
            "each question given the sections that bear on it, best first",
            vec![port("sections", ArtifactKind::Files)],
            port("passages", ArtifactKind::Files),
            serde_json::to_value(schema_for!(QueryParams)).unwrap_or(Value::Null),
        )
    }
    fn check(&self, params: &Value) -> Result<(), String> {
        let p = query_params(params)?;
        if p.passages == 0 {
            return Err("passages is at least 1".into());
        }
        Ok(())
    }
    fn files(&self, params: &Value) -> Vec<String> {
        query_params(params)
            .map(|p| vec![p.questions])
            .unwrap_or_default()
    }
    fn check_files(
        &self,
        params: &Value,
        files: &std::collections::BTreeMap<String, String>,
    ) -> Vec<String> {
        let Ok(p) = query_params(params) else {
            return Vec::new();
        };
        match files.get(&p.questions).map(|text| questions(text)) {
            Some(Err(why)) => vec![format!("{}: {why}", p.questions)],
            _ => Vec::new(),
        }
    }
    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

impl Execute for Query {
    fn implementation(&self) -> String {
        format!("{QUERY} 1")
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let p = query_params(call.params)?;
        let asked = questions(
            call.files
                .get(&p.questions)
                .ok_or_else(|| format!("{} is not pinned", p.questions))?,
        )
        .map_err(|why| format!("{}: {why}", p.questions))?;
        let view = call
            .inputs
            .get("sections")
            .and_then(|v| v.first())
            .ok_or("sections is not bound")?;
        let mut held = Vec::new();
        for (n, item) in read_index(view)?.items.into_iter().enumerate() {
            let dir = view.join(&item.path);
            let text = std::fs::read_to_string(dir.join(TEXT)).map_err(|e| e.to_string())?;
            let source: SectionSource = serde_json::from_str(
                &std::fs::read_to_string(dir.join(SOURCE)).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            held.push((item.id, source, Passage::of_text(&item.path, n, &text)));
        }
        let passages: Vec<Passage> = held.iter().map(|(_, _, p)| p.clone()).collect();
        let bm25 = Bm25::new(&passages);
        let mut items = Vec::new();
        for (n, question) in asked.iter().enumerate() {
            let id = format!("question-{:03}", n + 1);
            let found: Vec<Found<'_>> = bm25
                .search(question, p.passages)
                .into_iter()
                .map(|hit| {
                    let (section, source, passage) = &held[hit.passage];
                    Found {
                        section,
                        document: &source.document,
                        title: source.title.as_deref(),
                        score: hit.score,
                        text: &passage.text,
                    }
                })
                .collect();
            let dir = call.out.join(&id);
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            std::fs::write(dir.join("question.md"), format!("{}\n", question.trim()))
                .map_err(|e| e.to_string())?;
            write(&dir.join("passages.json"), &found)?;
            items.push(Item {
                path: id.clone(),
                id,
            });
        }
        index(&call.out, items)?;
        Ok(Outcome::Produced)
    }
}
