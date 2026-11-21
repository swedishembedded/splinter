// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec of the material a rationalisation reads: hints come only from
//! train-side, unsealed tasks the compile record names, and a refusal names
//! the task and why; the student's trials read the same packed or not; a
//! selection is stratified by rung and fixed by its seed.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::Path;

use serde_json::json;

use super::*;

fn task(root: &Path, rel: &str, extra_toml: &str, solution: bool) {
    let dir = root.join(rel);
    fs::create_dir_all(dir.join("environment/workspace")).unwrap();
    fs::create_dir_all(dir.join("tests/expected")).unwrap();
    fs::write(
        dir.join("task.toml"),
        format!("[task]\nname = \"t/{rel}\"\n[metadata]\nbench_difficulty = 3\n{extra_toml}"),
    )
    .unwrap();
    fs::write(dir.join("instruction.md"), format!("Do {rel}.\n")).unwrap();
    fs::write(dir.join("environment/workspace/main.txt"), "a\n").unwrap();
    fs::write(dir.join("tests/expected/main.txt"), "b\n").unwrap();
    if solution {
        fs::create_dir_all(dir.join("solution")).unwrap();
        fs::write(dir.join("solution/solve.sh"), "echo b > main.txt\n").unwrap();
    }
}

fn dataset(root: &Path) -> CompiledDataset {
    task(root, "train/a", "", true);
    task(root, "eval/b", "", true);
    task(root, "train/c", "", true);
    task(root, "train/d", "bench_sealed = true\n", true);
    task(root, "train/e", "", false);
    let entry = |path: &str, set: &str| json!({"name": format!("t/{path}"), "path": path, "digest": "d", "set": set});
    fs::write(
        root.join(COMPILE_FILE),
        json!({"schema": COMPILE_SCHEMA, "bench_commit": "abc", "tasks": [
            entry("train/a", "train"), entry("eval/b", "eval"),
            entry("train/d", "train"), entry("train/e", "train"),
        ]})
        .to_string(),
    )
    .unwrap();
    CompiledDataset::read(root).unwrap()
}

#[test]
fn hints_come_only_from_train_side_unsealed_tasks_the_record_names() {
    let scratch = tempfile::tempdir().unwrap();
    let root = scratch.path();
    let data = dataset(root);
    let a = data.train_task(&root.join("train/a")).unwrap();
    assert_eq!(a.rel, "train/a");
    assert_eq!(a.difficulty, Some(3));
    assert_eq!(a.hint, "echo b > main.txt\n");
    assert_eq!(a.files.changed(), Some(vec!["main.txt"]));
    for (rel, why) in [
        ("eval/b", "on the eval side"),
        ("train/c", "no record"),
        ("train/d", "sealed"),
        ("train/e", "no reference solution"),
    ] {
        let refused = data.train_task(&root.join(rel)).unwrap_err();
        assert!(
            refused.contains(why) && refused.contains(rel),
            "{rel}: {refused}"
        );
    }
    // A folder outside the dataset has no side.
    let other = tempfile::tempdir().unwrap();
    task(other.path(), "train/a", "", true);
    let refused = data.train_task(&other.path().join("train/a")).unwrap_err();
    assert!(refused.contains("not a folder of the dataset"), "{refused}");
    // A dataset without its compile record is refused whole.
    let bare = tempfile::tempdir().unwrap();
    assert!(CompiledDataset::read(bare.path())
        .unwrap_err()
        .contains(COMPILE_FILE));
}

fn result(trial: &str, task: &str, reward: Option<f64>, exception: bool) -> serde_json::Value {
    json!({
        "trial_name": trial,
        "config": {"task": {"path": task}},
        "verifier_result": reward.map(|r| json!({"rewards": {"reward": r}})),
        "exception_info": if exception { json!({"exception_type": "X"}) } else { json!(null) },
    })
}

#[test]
fn a_jobs_trials_read_the_same_packed_or_unpacked() {
    let scratch = tempfile::tempdir().unwrap();
    let rows = [
        result("a__1", "/d/train/a", Some(1.0), false),
        result("a__2", "/d/train/a", Some(0.0), false),
        result("b__1", "/d/train/b", None, true),
    ];
    let packed = scratch.path().join("packed");
    fs::create_dir_all(&packed).unwrap();
    let lines: Vec<String> = rows
        .iter()
        .map(|r| json!({"result": r, "score": {}}).to_string())
        .collect();
    fs::write(packed.join("trials.jsonl"), lines.join("\n")).unwrap();
    let unpacked = scratch.path().join("unpacked");
    for row in &rows {
        let dir = unpacked.join(row["trial_name"].as_str().unwrap());
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("result.json"), row.to_string()).unwrap();
    }
    fs::write(unpacked.join("result.json"), "{}").unwrap();
    let read = |job: &Path| {
        student_trials(job)
            .unwrap()
            .into_iter()
            .map(|t| (t.passed(), t.failed(), t))
            .map(|(passed, failed, t)| (t.trial, t.task, passed, failed))
            .collect::<Vec<_>>()
    };
    assert!(is_packed(&packed) && !is_packed(&unpacked));
    let expected = vec![
        ("a__1".to_string(), PathBuf::from("/d/train/a"), true, false),
        ("a__2".to_string(), PathBuf::from("/d/train/a"), false, true),
        // An exception is neither a pass nor a fail.
        (
            "b__1".to_string(),
            PathBuf::from("/d/train/b"),
            false,
            false,
        ),
    ];
    assert_eq!(read(&packed), expected);
    assert_eq!(read(&unpacked), expected);
}

#[test]
fn a_selection_takes_each_rung_its_share_and_the_seed_fixes_it() {
    // Rungs 1, 2 and 3 with 6, 3 and 3 tasks; 6 drawn: 3, 1.5, 1.5 -> 3, 2, 1
    // or 3, 1, 2 by the seeded tie.
    let items: Vec<(i64, String)> = [(1, 6), (2, 3), (3, 3)]
        .iter()
        .flat_map(|(rung, n)| (0..*n).map(move |i| (*rung, format!("r{rung}-{i}"))))
        .collect();
    let seed = splinter_model::seed::draw().value();
    // Any seed: the same one must select the same items.
    let pick = |count| stratified(items.clone(), |i| Some(i.0), |i| i.1.as_str(), count, seed);
    let chosen = pick(6);
    let per = |rung| chosen.iter().filter(|i| i.0 == rung).count();
    assert_eq!(chosen.len(), 6);
    assert_eq!(per(1), 3);
    assert_eq!(per(2) + per(3), 3);
    assert!((1..=2).contains(&per(2)) && (1..=2).contains(&per(3)));
    assert_eq!(chosen, pick(6));
    // Asking for more than there are takes them all.
    assert_eq!(pick(99).len(), items.len());
    assert!(pick(0).is_empty());
}
