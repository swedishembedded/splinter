// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements delegated-task coding agents that grow
// their own training data. If your team needs expertise in data extraction
// from technical documents, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Fact exploration: a markdown fact sheet becomes chat-training records.
//!
//! `explore` splits a document at its headings, asks the configured model
//! to enumerate EVERY factual claim of each section as a question/answer
//! pair, and appends one training record per fact to a JSONL file in the
//! SAME schema `learn` writes to the experience pool - so a trained-on
//! questions file is trainer input without translation.
//!
//! Two rules keep the dataset honest:
//!
//! - Strict parse: a reply that is not exactly one
//!   `{"facts": [{"question", "answer"}, ...]}` object is counted as a
//!   failure for its section and skipped, not salvaged - a half-parsed
//!   reply would silently weight whatever the model felt like emitting.
//! - Dedup by normalized question text: the same fact appearing in two
//!   sections (or re-run over a grown document) trains once.

use anyhow::Context;
use splinter_policy::{complete_text, ModelSelection};
use splinter_store::runs::{Limits, RunManifest};
use splinter_store::trace::Trace;
use splinter_store::{write_atomic, StateRoot};

use crate::extract::{facts_prompt, parse_facts_reply};
use crate::gates::{answer_numbers_traceable, normalize, question_is_anchored};
use crate::negatives::{negative_question, NOT_COVERED};
use crate::sections::{document_title, split_sections, title_identifiers};

/// Everything one exploration needs, as configured by the caller.
#[derive(Clone, Debug)]
pub struct ExploreOptions {
    /// The markdown fact sheet to read.
    pub file: std::path::PathBuf,
    /// The JSONL training file to write atomically at the end.
    pub out: std::path::PathBuf,
    /// Section size cap in lines; a section past it starts a new chunk at
    /// the next heading or paragraph boundary. `None` means uncapped.
    pub chunk_lines: Option<usize>,
    /// The model that extracts the facts.
    pub model: ModelSelection,
    /// Device identifiers OUTSIDE the document's scope. Every other
    /// accepted fact also trains a negative variant with one of these
    /// substituted, answered by [`NOT_COVERED`], so the adapter learns
    /// where its knowledge ends instead of answering other chips with
    /// this chip's numbers.
    pub scope_negatives: Vec<String>,
}

/// What one exploration produced, for the CLI's summary line.
#[derive(Debug, PartialEq, Eq)]
pub struct ExploreSummary {
    /// Id of the exploration's run (`explore-...`), naming its directory
    /// under `runs/` and stamped on every record it wrote.
    pub run_id: String,
    /// Sections the document was split into, after the chunk cap.
    pub sections: usize,
    /// Training records written to the output file: accepted facts plus
    /// the scope negatives derived from them.
    pub facts: usize,
    /// Sections that yielded nothing because the completion failed or its
    /// reply did not parse strictly.
    pub parse_failures: usize,
    /// Questions the anchor gate refused: the generator produced them
    /// without the device identifier the title carries.
    pub unanchored: usize,
    /// Facts the traceability gate refused: the answer states a number
    /// the section does not carry.
    pub untraceable: usize,
}

/// Explores `options.file` section by section with `options.model`, writes
/// the accepted facts to `options.out` as training records, and records the
/// exploration as its own run (manifest, trace, outcome) under `root`.
/// A section whose completion fails or whose reply does not parse is
/// counted in the summary, not an error; errors are failures of the run
/// itself - an unreadable or sectionless document, a model that does not
/// load, a state write that fails.
pub fn run(root: &StateRoot, options: ExploreOptions) -> anyhow::Result<ExploreSummary> {
    let text = std::fs::read_to_string(&options.file)
        .with_context(|| format!("reading {}", options.file.display()))?;
    let sections = split_sections(&text, options.chunk_lines);
    anyhow::ensure!(
        !sections.is_empty(),
        "{}: no sections found",
        options.file.display()
    );
    let identifiers = document_title(&text)
        .map(title_identifiers)
        .unwrap_or_default();

    let run_id = splinter_store::new_id_with_prefix("explore");
    let dir = root.run_dir(&run_id);
    let trace = Trace::open(&dir, &run_id, 1)?;

    let mut manifest = RunManifest {
        schema: 1,
        run_id: run_id.clone(),
        workspace: options.file.display().to_string(),
        task: format!("explore facts from {}", options.file.display()),
        status: "pending".into(),
        attempts: 1,
        started_ts: splinter_store::clock::utc_now(),
        updated_ts: splinter_store::clock::utc_now(),
        model: options.model.identity(),
        base_url: match &options.model {
            ModelSelection::Remote(remote) => remote.base_url.clone(),
            ModelSelection::Local(_) => None,
        },
        local_adapter: options.model.local().and_then(|w| w.adapter.clone()),
        limits: Limits::default(),
    };
    write_atomic(
        &dir.join("run.json"),
        &serde_json::to_string_pretty(&manifest)?,
    )?;

    // One provider for the whole run: the load is the expensive step, and
    // every section wants the same model anyway.
    let provider = options.model.provider()?;
    let rt = tokio::runtime::Runtime::new()?;

    let mut records: Vec<serde_json::Value> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut parse_failures = 0usize;
    let mut unanchored = 0usize;
    let mut untraceable = 0usize;
    let mut negatives_emitted = 0usize;
    for (n, section) in sections.iter().enumerate() {
        let prompt = facts_prompt(section);
        let result: anyhow::Result<String> =
            rt.block_on(async { complete_text(provider.as_ref(), &prompt).await });
        let mut payload = match result {
            Ok(reply) => match parse_facts_reply(&reply) {
                Ok(facts) => {
                    let mut added = 0usize;
                    let mut rejected = 0usize;
                    let mut untraceable_section = 0usize;
                    for (question, answer) in facts.pairs {
                        let key = normalize(&question);
                        if !seen.insert(key) {
                            continue; // already present: skipped, not rewritten
                        }
                        // The gate refuses what the instruction asked for
                        // and the generator half-delivered: a question the
                        // title cannot anchor would train this chip's
                        // answers onto other chips' questions.
                        if !question_is_anchored(&question, &identifiers) {
                            rejected += 1;
                            continue;
                        }
                        // The traceability gate refuses invented numbers:
                        // a trained wrong number is worse than no fact,
                        // and recall against the same wrong reference can
                        // never catch it.
                        if !answer_numbers_traceable(&answer, section) {
                            untraceable_section += 1;
                            continue;
                        }
                        records.push(splinter_lab::answers::training_record(
                            &run_id, &question, &answer,
                        ));
                        added += 1;
                        // One negative per second accepted fact: enough to
                        // teach the boundary without letting the shared
                        // abstention target outweigh the facts themselves.
                        if added.is_multiple_of(2) && !options.scope_negatives.is_empty() {
                            let id = identifiers.iter().find(|id| question.contains(id.as_str()));
                            if let (Some(id), Some(negative)) = (
                                id,
                                options
                                    .scope_negatives
                                    .get(negatives_emitted % options.scope_negatives.len()),
                            ) {
                                negatives_emitted += 1;
                                if let Some(negative) =
                                    negative_question(&question, id, negative, &identifiers)
                                {
                                    records.push(splinter_lab::answers::training_record(
                                        &run_id,
                                        &negative,
                                        NOT_COVERED,
                                    ));
                                }
                            }
                        }
                    }
                    unanchored += rejected;
                    untraceable += untraceable_section;
                    serde_json::json!({
                        "section": n, "facts": added,
                        "unanchored": rejected, "untraceable": untraceable_section
                    })
                }
                Err(e) => {
                    parse_failures += 1;
                    serde_json::json!({ "section": n, "facts": 0, "parse_failure": format!("{e:#}") })
                }
            },
            Err(e) => {
                parse_failures += 1;
                serde_json::json!({ "section": n, "facts": 0, "completion_error": format!("{e:#}") })
            }
        };
        trace.event("section_explored", &mut payload)?;
    }

    // The dataset is written atomically at the end: a crash mid-exploration
    // leaves no half-file the trainer could read as complete.
    let mut jsonl = String::new();
    for record in &records {
        jsonl.push_str(&serde_json::to_string(record)?);
        jsonl.push('\n');
    }
    write_atomic(&options.out, &jsonl)?;

    let summary = ExploreSummary {
        run_id: run_id.clone(),
        sections: sections.len(),
        facts: records.len(),
        parse_failures,
        unanchored,
        untraceable,
    };
    let mut outcome = serde_json::json!({
        "schema": 1,
        "run_id": run_id,
        "status": if parse_failures > 0 { "completed_with_parse_failures" } else { "completed" },
        "file": options.file.display().to_string(),
        "out": options.out.display().to_string(),
        "sections": summary.sections,
        "facts": summary.facts,
        "parse_failures": summary.parse_failures,
        "unanchored": summary.unanchored,
        "untraceable": summary.untraceable,
    });
    trace.event("explore_outcome", &mut outcome)?;
    write_atomic(
        &dir.join("outcome.json"),
        &serde_json::to_string_pretty(&outcome)?,
    )?;
    manifest.status = "completed".into();
    manifest.updated_ts = splinter_store::clock::utc_now();
    write_atomic(
        &dir.join("run.json"),
        &serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(summary)
}
