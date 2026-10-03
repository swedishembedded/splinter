// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free train and exam splits for
// language-model experiments for its clients. If your team needs expertise in
// proving that an exam shares nothing with the training set, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Assembling the training set and the frozen exam.
//!
//! The exam is written before any training and never changes. It has two
//! parts: `exam` items come from letter families and passages no training
//! record contains, which measures whether the model knows the sources;
//! `seen` items are a sample of what training contains, which measures
//! whether it kept what it was shown. A model that improves on `seen` and not
//! on `exam` memorised; one that improves on both learned something general.

use std::collections::HashSet;
use std::path::Path;

use crate::corpus::{families, load_letters};
use crate::tasks::{letter_tasks, passage_task, Kind, Task};
use crate::works::{passages, Passage, WORKS};

/// How much of each kind goes into the training set and the exam.
#[derive(Clone, Copy, Debug)]
pub struct Sizes {
    /// Letters (each gives a recipient and a year task) used for training.
    pub train_letters: usize,
    /// Letters held out for the exam.
    pub exam_letters: usize,
    /// Training passages per work.
    pub train_passages: usize,
    /// Exam passages per work.
    pub exam_passages: usize,
    /// Training letters and passages probed again as `seen`.
    pub seen: usize,
}

/// The training tasks and the exam items.
pub struct Built {
    /// What fine-tuning sees.
    pub train: Vec<Task>,
    /// The exam: `exam` items first, then `seen` ones, marked in `split`.
    pub exam: Vec<Task>,
}

/// `items` ordered by a seeded hash of `key(item)` and cut to `count`: the
/// same choice whatever order the corpus was read in.
fn pick<T>(mut items: Vec<T>, count: usize, seed: u64, key: impl Fn(&T) -> String) -> Vec<T> {
    items.sort_by_cached_key(|item| {
        blake3::hash(format!("{seed}:{}", key(item)).as_bytes())
            .to_hex()
            .to_string()
    });
    items.truncate(count);
    items
}

/// Builds the training set and the exam from `resources`.
///
/// # Errors
/// A source file that exists but cannot be read.
pub fn build(resources: &Path, sizes: Sizes, seed: u64) -> anyhow::Result<Built> {
    let letters = load_letters(resources)?;
    let family = families(&letters);
    let mut by_family: Vec<(String, Vec<Task>)> = Vec::new();
    for (n, letter) in letters.iter().enumerate() {
        if family[n] != n {
            continue;
        }
        let tasks = letter_tasks(letter, &letter.id, seed);
        if !tasks.is_empty() {
            by_family.push((letter.id.clone(), tasks));
        }
    }
    let (exam_f, train_f): (Vec<_>, Vec<_>) = by_family
        .into_iter()
        .partition(|(_, t)| t[0].split == "exam");
    let exam_f = pick(exam_f, sizes.exam_letters, seed, |(id, _)| id.clone());
    let train_f = pick(train_f, sizes.train_letters, seed, |(id, _)| id.clone());

    let mut train: Vec<Task> = train_f.iter().flat_map(|(_, t)| t.clone()).collect();
    let mut exam_items: Vec<Task> = exam_f.iter().flat_map(|(_, t)| t.clone()).collect();

    for (index, work) in WORKS.iter().enumerate() {
        let path = resources.join(format!("{}.txt", work.file));
        let text = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        let mut seen_text = HashSet::new();
        let tasks: Vec<Task> = passages(index, &text, work.gutenberg)
            .into_iter()
            .filter(|p: &Passage| seen_text.insert(p.text.clone()))
            .map(|p| passage_task(&p, seed))
            .collect();
        let (exam_p, train_p): (Vec<_>, Vec<_>) =
            tasks.into_iter().partition(|t| t.split == "exam");
        train.extend(pick(train_p, sizes.train_passages, seed, |t| t.id.clone()));
        exam_items.extend(pick(exam_p, sizes.exam_passages, seed, |t| t.id.clone()));
    }

    // The seen probe: a seeded sample of each kind of what training contains.
    let seen_letters = pick(
        train
            .iter()
            .filter(|t| t.kind != Kind::Work)
            .cloned()
            .collect(),
        sizes.seen,
        seed,
        |t| t.id.clone(),
    );
    let seen_passages = pick(
        train
            .iter()
            .filter(|t| t.kind == Kind::Work)
            .cloned()
            .collect(),
        sizes.seen / 2,
        seed,
        |t| t.id.clone(),
    );
    let mut exam = exam_items;
    exam.extend(seen_letters.into_iter().chain(seen_passages).map(|mut t| {
        t.split = "seen".to_string();
        t
    }));
    // The trainer holds out the newest tenth of the file, so a seen item's
    // record goes first: it is always trained on, never held out.
    let seen_prompts: HashSet<&str> = exam
        .iter()
        .filter(|t| t.split == "seen")
        .map(|t| t.prompt.as_str())
        .collect();
    let mut train = pick(train, usize::MAX, seed, |t| t.id.clone());
    train.sort_by_key(|t| !seen_prompts.contains(t.prompt.as_str()));
    Ok(Built { train, exam })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picking_depends_on_the_seed_and_not_on_the_order_of_the_input() {
        let items: Vec<u32> = (0..50).collect();
        let reversed: Vec<u32> = items.iter().rev().copied().collect();
        let a = pick(items.clone(), 10, 1, |n| n.to_string());
        let b = pick(reversed, 10, 1, |n| n.to_string());
        assert_eq!(a, b);
        assert_ne!(
            a,
            pick(items, 10, 2, |n| n.to_string()),
            "another seed, another choice"
        );
        assert_eq!(a.len(), 10);
    }

    /// No exam item shares a family, a passage or a question with any
    /// training record, whatever the seed: the frozen exam cannot leak.
    #[test]
    fn nothing_in_the_exam_is_in_the_training_set() {
        let resources = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../resources");
        if !resources.join("thomas-jefferson").is_dir() {
            return; // the corpus is fetched, not checked in
        }
        let sizes = Sizes {
            train_letters: 40,
            exam_letters: 15,
            train_passages: 6,
            exam_passages: 3,
            seen: 10,
        };
        for seed in 1..=8 {
            let built =
                build(&resources, sizes, seed).unwrap_or_else(|e| panic!("build failed: {e}"));
            let trained_families: HashSet<&str> =
                built.train.iter().map(|t| t.family.as_str()).collect();
            let trained_prompts: HashSet<&str> =
                built.train.iter().map(|t| t.prompt.as_str()).collect();
            for item in built.exam.iter().filter(|t| t.split == "exam") {
                assert!(
                    !trained_families.contains(item.family.as_str()),
                    "{} leaks its family",
                    item.id
                );
                assert!(
                    !trained_prompts.contains(item.prompt.as_str()),
                    "{} leaks its question",
                    item.id
                );
            }
            for item in built.exam.iter().filter(|t| t.split == "seen") {
                assert!(
                    trained_prompts.contains(item.prompt.as_str()),
                    "a seen item is something training contains"
                );
            }
            assert!(
                built.exam.iter().any(|t| t.split == "exam")
                    && built.exam.iter().any(|t| t.split == "seen")
            );
            // The trainer holds out the newest tenth of the file; a seen item
            // whose record was held out would not have been trained on.
            let tail_from = built.train.len() - built.train.len().div_ceil(10);
            for item in built.exam.iter().filter(|t| t.split == "seen") {
                let at = built.train.iter().position(|t| t.prompt == item.prompt);
                assert!(
                    at.is_some_and(|n| n < tail_from),
                    "{} sits in the held-out tail",
                    item.id
                );
            }
        }
    }
}
