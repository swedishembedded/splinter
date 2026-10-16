// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in pre-registering a
// replication study so that no choice can follow its results, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The corpus stage of one author: write the essays, read the decisions out
//! of them, make the training records, and freeze the examination, all before
//! anything is trained.
//!
//! The stage is resumable and idempotent. The writer, the gate reader and the
//! miner keep their own logs, so a repeat asks nothing it has been answered;
//! the records and the examination are functions of what was read, so a repeat
//! makes the same ones. An author whose manifest exists is checked, not redone:
//! if the examination drawn from the corpus now is not the one frozen, the
//! corpus was changed after the freeze and the stage refuses.

use anyhow::Context as _;

use super::layout::{AuthorManifest, CorpusRecord, DatasetRecord, FrozenExam, Study, EXAM_ITEMS};
use crate::binding::{file_digest, items_digest};
use crate::corpus_cmd::{print_summary, Plan};
use crate::dataset::Briefings;
use crate::lab::{build_dataset, exam_items_of, item_view, mine_all, read_mined, MinedResult};
use crate::manifest::{begin_or_continue_mining, code_revision};
use crate::models::is_remote;
use crate::survivors::of_corpus;
use crate::transit::Transit;
use crate::writer::Models;

/// The models that do the corpus stage's work, as the protocol's three roles.
pub struct Staff<'a> {
    /// Writes the essays.
    pub writer: &'a dyn Models,
    /// Reads each draft for the gate.
    pub reader: &'a dyn Models,
    /// Reads the decisions out of the kept essays.
    pub miner: &'a dyn Models,
}

/// Runs the corpus stage for the author with seed `author_seed`.
///
/// # Errors
/// The author is not in the protocol; the corpus was begun under another
/// recipe or protocol; a model cannot be reached or the run's budget ran out
/// before every article was dealt with (run it again to resume); the
/// examination cannot be drawn at the protocol's size; or it differs from the
/// one already frozen for the author.
pub fn run(
    study: &Study,
    author_seed: u64,
    staff: &Staff<'_>,
    pace: &Transit,
) -> anyhow::Result<AuthorManifest> {
    let protocol = study.protocol();
    let kept = study.author_manifest(author_seed)?;
    let cache = study.cache(author_seed)?;
    let plan = Plan {
        writer: protocol.models.writer.clone(),
        reader: Some(protocol.models.reader.clone()),
        allow_same_family: false,
        articles: protocol.corpus.articles,
        episodes_per_article: protocol.corpus.episodes_per_article,
        min_words: protocol.corpus.min_words,
        attempts: protocol.corpus.attempts,
        max_calls: protocol.corpus.max_calls,
        max_output_tokens: protocol.corpus.max_output_tokens,
        workers: protocol.corpus.workers,
        transport_retries: pace.retries,
        transport_backoff: pace.backoff,
    };
    let prepared = plan.prepare(cache.dir(), Some(author_seed), None, None)?;
    let options = plan.begin(cache.dir(), &prepared)?;
    let author = prepared.author;
    let summary = crate::writer::write_corpus_read(
        &author.persona,
        author.seed,
        staff.writer,
        Some(staff.reader),
        &cache,
        &options,
    )?;
    print_summary(&author.persona.name, &summary);
    let kept_articles = cache.articles()?.len();
    if summary.stopped.is_some() || summary.unavailable > 0 || kept_articles == 0 {
        let mut causes = Vec::new();
        if let Some(why) = &summary.stopped {
            causes.push(format!("stopped: {why}"));
        }
        if summary.abandoned > 0 {
            causes.push(format!(
                "{} article(s) given up after {} attempts each{}",
                summary.abandoned,
                protocol.corpus.attempts,
                summary
                    .last_turned_away
                    .as_ref()
                    .map_or_else(String::new, |why| format!(
                        " (the last was turned away: {why})"
                    )),
            ));
        }
        let headline = if kept_articles == 0 {
            "no article was written".to_string()
        } else {
            format!(
                "{kept_articles} of {} articles written",
                protocol.corpus.articles
            )
        };
        anyhow::bail!(
            "author {author_seed}: {headline}, so the corpus is not finished ({}); run `study corpus` again to resume",
            causes.join("; "),
        );
    }

    begin_or_continue_mining(
        cache.dir(),
        &protocol.models.miner,
        None,
        !is_remote(&protocol.models.miner)?,
        false,
    )?;
    let read = mine_all(&cache, staff.miner, protocol.corpus.workers, pace)?;
    let mined = read_mined(&cache)?;
    let decisions_kept = mined
        .iter()
        .filter(|m| matches!(m.result, MinedResult::Kept(_)))
        .count();
    anyhow::ensure!(
        decisions_kept > 0,
        "author {author_seed}: the miner read no decision out of {kept_articles} article(s): {} case(s) were turned away{}; run `study corpus` again after looking at decisions.jsonl",
        mined.len(),
        mined
            .iter()
            .find_map(|m| match &m.result {
                MinedResult::Refused(why) => Some(format!(" (the first: {why})")),
                MinedResult::Kept(_) => None,
            })
            .unwrap_or_default(),
    );
    eprintln!(
        "author {author_seed}: {read} articles read; {decisions_kept} of {} cases kept",
        mined.len()
    );

    let mut datasets = Vec::new();
    for briefings in
        std::iter::once(Briefings::Both).chain(protocol.ablations.briefings.iter().copied())
    {
        let made = build_dataset(&cache, author.seed, briefings)?;
        datasets.push(DatasetRecord {
            briefings,
            records: made.records,
            without_record: made.without_record,
            digest: file_digest(&made.path)?,
        });
    }

    let items = exam_items_of(
        &cache,
        &author.persona,
        author.seed,
        protocol.corpus.articles,
        protocol.exam.per_cell,
        protocol.exam.version,
    )?;
    anyhow::ensure!(
        items.len() == protocol.exam.per_cell * super::power::CELLS,
        "author {author_seed}: the examination at {} items per kind has {} items: the situations the corpus leaves out cannot fill every kind",
        protocol.exam.per_cell,
        items.len()
    );
    let digest = items_digest(&items).to_string();
    if let Some(kept) = &kept {
        anyhow::ensure!(
            kept.exam.digest == digest,
            "author {author_seed}: the examination drawn from the corpus now is {digest} and the one frozen is {}: \
             the corpus was changed after the examination was frozen, and no adapter may be examined on one \
             that depends on what it was trained on",
            kept.exam.digest,
        );
    }
    let shown: String = items
        .iter()
        .map(|i| format!("{}\n", item_view(i)))
        .collect();
    let file = cache.dir().join(EXAM_ITEMS);
    crate::binding::write_whole(&file, shown.as_bytes())?;

    let manifest = AuthorManifest {
        protocol_digest: study.digest().to_string(),
        author_seed,
        author_name: author.persona.name.clone(),
        models: protocol.models.clone(),
        code_revision: code_revision().to_string(),
        corpus: CorpusRecord {
            articles_planned: protocol.corpus.articles,
            articles_kept: cache.articles()?.len(),
            cases_read: mined.len(),
            decisions_kept,
            usage: cache.usage()?,
        },
        datasets,
        survivors: of_corpus(
            &cache,
            &author,
            protocol.corpus.articles,
            protocol.corpus.episodes_per_article,
        )
        .context("accounting for what the gates took out")?,
        exam: FrozenExam {
            version: protocol.exam.version,
            articles: protocol.corpus.articles,
            per_cell: protocol.exam.per_cell,
            items: items.len(),
            digest,
            file_digest: file_digest(&file)?,
        },
    };
    study.write_author_manifest(&manifest)?;
    Ok(manifest)
}
