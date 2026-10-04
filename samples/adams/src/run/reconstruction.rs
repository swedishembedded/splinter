// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements historical-behavior reconstruction tests for
// language models for its clients. If your team needs expertise in measuring
// whether a model reproduces what a person actually did, from what they were
// shown and not from what they wrote, you can procure our services by sending
// an email to info@swedishembedded.com.

//! The reconstruction benchmark's commands: brief the letters, have a model
//! write the replies, judge them and report.

use super::*;

/// His own letters on one side of the split: the exam side is what the
/// reconstruction benchmark is made from, the training side what he is shown
/// doing.
pub(super) fn letters_of<'a>(
    resources: &Path,
    documents: &'a [Document],
    side: Split,
) -> anyhow::Result<Vec<&'a Document>> {
    let ids: std::collections::HashSet<String> = read_assignments(resources)?
        .into_iter()
        .filter(|a| a.split == side)
        .map(|a| a.doc_id)
        .collect();
    let mut letters: Vec<&Document> = documents
        .iter()
        .filter(|d| ids.contains(&d.id) && d.authorship.is_his_own_letter())
        .collect();
    letters.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(letters)
}

fn reconstruction_dir(resources: &Path) -> anyhow::Result<PathBuf> {
    let dir = resources.join("reconstruction");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Brief every held-out letter of his: the situation it answered, in a
/// helper's words, and what the real letter does, each point grounded in it.
pub fn briefings_command(
    resources: &Path,
    documents: &[Document],
    mine: &Mine<'_>,
    side: Split,
) -> anyhow::Result<()> {
    let letters = letters_of(resources, documents, side)?;
    let file = if side == Split::Exam {
        "briefings.jsonl"
    } else {
        "briefings-train.jsonl"
    };
    let path = reconstruction_dir(resources)?.join(file);
    let helper = crate::helper::Helper::served(&mine.served)?;
    let asked = crate::reconstruct::brief_all(&helper, &letters, &path, mine.limit)?;
    let made = crate::reconstruct::read_briefings(&path)?.len();
    println!(
        "letters of his on the {side:?} side: {}  asked this run: {asked}  briefings made: {made}",
        letters.len()
    );
    Ok(())
}

/// What one reconstruction arm needs.
pub struct Replies {
    pub briefings: PathBuf,
    pub out: PathBuf,
    pub arm: String,
    pub base: PathBuf,
    pub adapter: Option<PathBuf>,
    pub max_tokens: u32,
    pub limit: Option<usize>,
    pub framing: crate::persona::Framing,
    pub decoding: Decoding,
}

/// Have one model write the reply to every briefing, in full.
pub fn replies_command(r: &Replies) -> anyhow::Result<()> {
    ensure_unchanged(&r.briefings)?;
    let briefings = crate::reconstruct::read_briefings(&r.briefings)?;
    let answerer = splinter_sdk::model::answer::Answerer::load_with(
        &r.base,
        r.adapter.as_deref(),
        Some(8192),
        "adams",
        r.decoding,
    )?;
    let runtime = tokio::runtime::Runtime::new()?;
    let mut ask = |b: &crate::reconstruct::Briefing| {
        runtime.block_on(answerer.ask(
            r.framing.system(crate::reconstruct::SYSTEM),
            &crate::reconstruct::prompt(b),
            r.max_tokens,
        ))
    };
    let asked = crate::reconstruct::answer_all(&briefings, &r.out, &r.arm, r.limit, &mut ask)?;
    println!(
        "asked {asked} briefings as {}; replies in {}",
        r.arm,
        r.out.display()
    );
    Ok(())
}

/// Calibrate the judge once, on controls, and keep the result so every arm is
/// judged by the same standard; then score every reply of `replies`.
pub fn judge_command(
    resources: &Path,
    documents: &[Document],
    mine: &Mine<'_>,
    replies: &Path,
    out: &Path,
) -> anyhow::Result<()> {
    let dir = reconstruction_dir(resources)?;
    let briefings = crate::reconstruct::read_briefings(&dir.join("briefings.jsonl"))?;
    let helper = crate::helper::Helper::served(&mine.served)?;

    let calibration_path = dir.join("calibration.json");
    let calibration: crate::judge::Calibration = match std::fs::read_to_string(&calibration_path)
        .ok()
        .and_then(|text| serde_json::from_str::<crate::judge::Calibration>(&text).ok())
        .filter(|c| c.generic_reply.is_some())
    {
        Some(kept) => kept,
        None => {
            const CONTROLS: usize = 20;
            let body = |id: &str| {
                documents
                    .iter()
                    .find(|d| d.id == id)
                    .map(|d| d.body.clone())
                    .unwrap_or_default()
            };
            let mut texts: Vec<(usize, String, String, String)> = Vec::new();
            for (i, b) in briefings.iter().take(CONTROLS).enumerate() {
                texts.push((
                    i,
                    body(&b.doc_id),
                    body(&briefings[(i + 1) % briefings.len()].doc_id),
                    crate::judge::generic_reply(&helper, b)?,
                ));
            }
            let controls: Vec<crate::judge::Control<'_>> = texts
                .iter()
                .map(|(i, real, other, generic)| crate::judge::Control {
                    briefing: &briefings[*i],
                    real,
                    other,
                    generic,
                })
                .collect();
            let found = crate::judge::calibrate(&helper, &controls)?;
            std::fs::write(&calibration_path, serde_json::to_string_pretty(&found)?)?;
            found
        }
    };
    println!(
        "judge controls on {} briefings: real letter {:.0}%, another letter {:.0}%, a generic reply {:.0}%: {}",
        calibration.briefings,
        100.0 * calibration.real_letter,
        100.0 * calibration.other_letter,
        100.0 * calibration.generic_reply.unwrap_or(f64::NAN),
        if calibration.passes() {
            "passes"
        } else {
            "does not pass; no claim will be made about coverage"
        }
    );

    let replies = crate::reconstruct::read_replies(replies)?;
    let corpus = crate::transfer::corpus_index(documents);
    let scored =
        crate::reconstruct::score_all(&helper, &briefings, &replies, &corpus, out, mine.limit)?;
    println!("scored {scored} replies; scores in {}", out.display());
    Ok(())
}

/// The comparison of two arms' scores, with the judge's controls beside it.
pub fn reconstruct_report_command(
    resources: &Path,
    before: &Path,
    after: &Path,
) -> anyhow::Result<()> {
    let calibration: crate::judge::Calibration = serde_json::from_str(&std::fs::read_to_string(
        reconstruction_dir(resources)?.join("calibration.json"),
    )?)?;
    let (before_scores, after_scores) = (
        crate::reconstruct::read_scores(before)?,
        crate::reconstruct::read_scores(after)?,
    );
    let arm = |scores: &[crate::reconstruct::Scored], fallback: &Path| {
        scores
            .first()
            .map_or_else(|| fallback.display().to_string(), |s| s.arm.clone())
    };
    let summary = crate::reconstruct::summarize(&before_scores, &after_scores);
    print!(
        "{}",
        crate::reconstruct::render(
            &summary,
            &calibration,
            &arm(&before_scores, before),
            &arm(&after_scores, after)
        )
    );
    Ok(())
}
