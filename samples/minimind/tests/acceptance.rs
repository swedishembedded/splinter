// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The run's contract, on the smoke profile against a model service double:
//! a fresh run goes end to end; a second run trains nothing and repeats no
//! episode; a run killed in a stage resumes it; a corrupt checkpoint is
//! caught; a changed seed is a new run; a changed setting or service
//! revision recomputes only what reads it, and a change to this repository's
//! own code alone recomputes nothing; the same episodes write the same
//! records; and nothing is promoted without a verified Promote.

#![allow(clippy::unwrap_used)]

mod common;

use std::path::Path;
use std::sync::atomic::Ordering;

use common::{FakeBrain, FakeExaminer};
use splinter_minimind::config::{Config, Profile};
use splinter_minimind::stages::{run_all, Revisions, Run};
use splinter_sdk::bench::examine::Examiner;
use splinter_sdk::store::stage_ledger::{Commit, StageLedger};

/// A directory for one spec, in memory when the machine has a memory-backed
/// file system, removed when the spec ends, a failing one included.
fn temp() -> tempfile::TempDir {
    let memory = Path::new("/dev/shm");
    let root = if memory.is_dir() {
        memory.to_path_buf()
    } else {
        std::env::temp_dir()
    };
    tempfile::Builder::new()
        .prefix("minimind-spec-")
        .tempdir_in(root)
        .unwrap()
}

/// The regular files below `dir`, symbolic links not followed.
fn files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(at) = stack.pop() {
        for entry in std::fs::read_dir(&at).unwrap() {
            let entry = entry.unwrap();
            let kind = entry.file_type().unwrap();
            if kind.is_dir() {
                stack.push(entry.path());
            } else if kind.is_file() {
                out.push(entry.path());
            }
        }
    }
    out
}

/// No episode, task folder or workspace left as loose files in the run
/// directory: only the stages' packs.
fn only_packs(dir: &Path) {
    for f in files(dir) {
        let rel = f.strip_prefix(dir).unwrap().to_string_lossy().into_owned();
        assert!(
            !(rel.starts_with("episodes/") || rel.starts_with("tasks/harbor"))
                || rel.ends_with(".jsonl"),
            "{rel} is a loose file"
        );
        assert!(!rel.contains("/workspace/"), "{rel} is a workspace file");
    }
}

fn revisions() -> Revisions {
    let mut r = Revisions::default();
    for name in ["splinter", "sven", "brain", "bench"] {
        r.0.insert(name.into(), "r1".into());
    }
    r
}

fn go(
    dir: &Path,
    config: Config,
    seed: u64,
    brain: &FakeBrain,
    examiner: Option<&dyn Examiner>,
) -> anyhow::Result<String> {
    go_at(dir, config, seed, brain, examiner, revisions())
}

fn go_at(
    dir: &Path,
    config: Config,
    seed: u64,
    brain: &FakeBrain,
    examiner: Option<&dyn Examiner>,
    revisions: Revisions,
) -> anyhow::Result<String> {
    go_taught(dir, config, seed, brain, examiner, revisions, None)
}

fn go_taught(
    dir: &Path,
    config: Config,
    seed: u64,
    brain: &FakeBrain,
    examiner: Option<&dyn Examiner>,
    revisions: Revisions,
    teacher: Option<splinter_minimind::teacher::Teacher>,
) -> anyhow::Result<String> {
    let mut out = Vec::new();
    let scratch = temp();
    let result = {
        let mut run = Run::open(dir, config, seed, revisions, brain, examiner, &mut out)?;
        run.use_scratch_root(scratch.path())?;
        run.teacher = teacher;
        run_all(&mut run)
    };
    assert_eq!(
        std::fs::read_dir(scratch.path()).unwrap().count(),
        0,
        "the run removes its scratch, whatever its outcome"
    );
    let shown = String::from_utf8(out).unwrap();
    result.map(|()| shown)
}

fn commits(dir: &Path) -> Vec<Commit> {
    StageLedger::open(dir).unwrap().commits().unwrap()
}

fn commit<'a>(commits: &'a [Commit], unit: &str) -> &'a Commit {
    commits
        .iter()
        .rev()
        .find(|c| c.unit == unit)
        .unwrap_or_else(|| panic!("no commit of {unit}"))
}

fn begins(dir: &Path) -> usize {
    std::fs::read_to_string(dir.join("ledger.jsonl"))
        .unwrap()
        .lines()
        .filter(|l| l.contains("\"record\":\"begin\""))
        .count()
}

/// The smoke profile with fewer instances and documents, so the specs run
/// in seconds.
fn smoke() -> Config {
    let mut config = Config::of(Profile::Smoke);
    config.tasks.demonstrations = 8;
    config.tasks.explore = 8;
    config.tasks.probes = 4;
    config.tasks.midtrain = 8;
    config.world.trajectories = 6;
    config.world.recall = 20;
    // One fine-tune segment, so the specs name one fine-tune unit.
    config.sft_segments = 1;
    config
}

// seed: the specs compare runs of one seed with each other; any seed works.
const SEED: u64 = 17;

#[test]
fn a_fresh_run_goes_end_to_end_and_a_second_run_trains_nothing() {
    let dir = temp();
    let brain = FakeBrain::new(dir.path());
    let jobs = temp();
    let examiner = FakeExaminer::new("Promote", true, jobs.path());
    let shown = go(dir.path(), smoke(), SEED, &brain, Some(&examiner)).unwrap();
    let all = commits(dir.path());
    assert_eq!(
        commit(&all, "demonstrate").summary["accepted"],
        8,
        "the expert solves every training instance through sven"
    );
    assert_eq!(
        commit(&all, "ouro/explore").summary["episodes"],
        8 * smoke().explore.samples,
        "every explored instance is attempted once per sample"
    );
    assert_eq!(commit(&all, "probe/ouro/init").summary["passed"], 0);
    assert_eq!(commit(&all, "probe/ouro/sft").summary["passed"], 4);
    assert!(commit(&all, "ouro/experience/verdict").summary["promotes"]
        .as_bool()
        .unwrap());
    assert!(shown.contains("probe ouro/sft: 4/4"), "{shown}");
    // Each checkpoint sits the exam once, the control arm among them; a
    // sitting is reused by both verdicts it takes part in.
    let sat = examiner.sittings.lock().unwrap().clone();
    assert_eq!(sat.len(), 4, "pretrained, sft, improved, control: {sat:?}");
    assert!(shown.contains("exam ouro/experience: Promote"), "{shown}");
    assert!(
        shown.contains("what each checkpoint can do"),
        "the summary table: {shown}"
    );
    assert!(dir.path().join("report.md").exists());
    // The worked trajectories the model is pretrained on show a shell
    // result exactly as sven's shell tool returned it in the demonstrations.
    let records = std::fs::read_to_string(dir.path().join("data/demonstrate.jsonl")).unwrap();
    let mut shell_results = 0;
    for line in records.lines() {
        let record: serde_json::Value = serde_json::from_str(line).unwrap();
        for m in record["messages"].as_array().unwrap() {
            let content =
                splinter_minimind::world::without_time_notice(m["content"].as_str().unwrap_or(""));
            if m["role"] == "tool" && content.contains("Exit: ") {
                let (stdout, status) = content.rsplit_once("Exit: ").unwrap();
                let exit: i32 = status.split(' ').next().unwrap().parse().unwrap();
                assert_eq!(
                    splinter_minimind::world::shell_result(stdout, exit),
                    content
                );
                shell_results += 1;
            }
        }
    }
    assert!(shell_results > 0, "the demonstrations ran shell commands");
    // The demonstrations are the requests the examination's headless agent
    // sends: the task file's text and every tool result end with the time
    // notice, and the pretraining documents list the tools in their order.
    let notice = splinter_minimind::world::time_notice(smoke().agent.deadline_secs);
    let first: serde_json::Value = serde_json::from_str(records.lines().next().unwrap()).unwrap();
    let messages = first["messages"].as_array().unwrap();
    let user = messages.iter().find(|m| m["role"] == "user").unwrap();
    assert!(
        user["content"]
            .as_str()
            .unwrap()
            .ends_with(&format!("\n\n\n{notice}")),
        "{user}"
    );
    for m in messages.iter().filter(|m| m["role"] == "tool") {
        assert!(m["content"].as_str().unwrap().ends_with(&notice), "{m}");
    }
    let offered: Vec<&str> = first["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap())
        .collect();
    let listed: Vec<String> = splinter_minimind::collect::tool_sigs()
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert_eq!(
        offered, listed,
        "the tools in the order the agent offers them"
    );
    // A committed stage leaves its thousands of small files as one pack.
    for tree in [
        "episodes/demonstrate",
        "episodes/ouro/explore",
        "episodes/probe/ouro/sft",
        "tasks/harbor",
    ] {
        assert!(!dir.path().join(tree).exists(), "{tree} is packed");
        assert!(
            dir.path().join(format!("{tree}.jsonl")).is_file(),
            "{tree}.jsonl"
        );
    }
    only_packs(dir.path());
    let trained = brain.trained();
    let started = begins(dir.path());

    let again = go(dir.path(), smoke(), SEED, &brain, Some(&examiner)).unwrap();
    assert_eq!(brain.trained(), trained, "a finished run trains nothing");
    assert_eq!(
        examiner.sittings.lock().unwrap().len(),
        4,
        "and sits no exam"
    );
    assert_eq!(
        begins(dir.path()),
        started,
        "and starts no unit, no episode among them"
    );
    assert!(
        again.contains("probe ouro/sft: 4/4"),
        "the probes are shown from the ledger: {again}"
    );
}

#[test]
fn a_teachers_verified_episodes_join_the_fine_tune_and_a_missing_teacher_is_named() {
    let dir = temp();
    let brain = FakeBrain::new(dir.path());
    let mut config = smoke();
    config.tasks.teach = 4;
    let refused = go(dir.path(), config.clone(), SEED, &brain, None).unwrap_err();
    assert!(format!("{refused:#}").contains("--teacher"), "{refused:#}");
    let models = dir.path().join("teacher-store");
    let source = dir.path().join("Teacher-1");
    std::fs::create_dir_all(&source).unwrap();
    std::fs::write(source.join("config.json"), "{}").unwrap();
    let teacher = splinter_minimind::teacher::Teacher::install(&source, &models).unwrap();
    assert_eq!(teacher.checkpoint.model, "local/teacher-teacher-1");
    assert!(
        models.join("local/teacher-teacher-1").exists(),
        "linked into the store"
    );
    go_taught(
        dir.path(),
        config,
        SEED,
        &brain,
        None,
        revisions(),
        Some(teacher),
    )
    .unwrap();
    let all = commits(dir.path());
    let taught = commit(&all, "teach");
    assert_eq!(
        taught.summary["episodes"],
        4 * 2,
        "every instance, every sample"
    );
    assert_eq!(taught.summary["accepted"], 4 * 2);
    let data = commit(&all, "sft-data");
    assert!(
        data.summary["teach"]["records"].as_u64().unwrap() > 0,
        "{}",
        data.summary
    );
    assert!(data.summary["demonstrate"]["records"].as_u64().unwrap() > 0);
    let sft = std::fs::read_to_string(dir.path().join("data/sft.jsonl")).unwrap();
    assert!(
        sft.contains("teach-00000"),
        "the fine-tune's data holds the teacher's episodes"
    );
}

#[test]
fn a_run_killed_in_fine_tuning_resumes_it_and_reuses_what_came_before() {
    let dir = temp();
    let brain = FakeBrain::new(dir.path());
    *brain.fail_once.lock().unwrap() = Some("ouro-sft".into());
    let err = go(dir.path(), smoke(), SEED, &brain, None).unwrap_err();
    assert!(format!("{err:#}").contains("killed during"), "{err:#}");
    let before = brain.trained();
    go(dir.path(), smoke(), SEED, &brain, None).unwrap();
    let all = commits(dir.path());
    assert_eq!(
        commit(&all, "ouro/sft").attempt,
        2,
        "the fine-tune resumed under a new attempt"
    );
    assert_eq!(
        commit(&all, "ouro/pretrain").attempt,
        1,
        "pretraining was not redone"
    );
    assert!(brain.trained() > before);
}

#[test]
fn an_episode_interrupted_by_a_failing_server_runs_again_under_a_new_attempt_and_counts_once() {
    let dir = temp();
    let brain = FakeBrain::new(dir.path());
    brain.down.store(true, Ordering::SeqCst);
    *brain.down_for.lock().unwrap() = Some("sft".into());
    assert!(
        go(dir.path(), smoke(), SEED, &brain, None).is_err(),
        "a model server that is down stops the run"
    );
    // Stopped mid-stage: what finished is in the packs, nothing is loose.
    only_packs(dir.path());
    let demonstrations =
        std::fs::read_to_string(dir.path().join("episodes/demonstrate.jsonl")).unwrap();
    assert_eq!(
        demonstrations.matches("/outcome.json\"").count(),
        8,
        "every finished demonstration is in its stage's pack"
    );
    brain.down.store(false, Ordering::SeqCst);
    go(dir.path(), smoke(), SEED, &brain, None).unwrap();
    let all = commits(dir.path());
    let probe = commit(&all, "probe/ouro/sft");
    assert_eq!(probe.summary["total"], 4, "every episode counted once");
    let retried: Vec<&Commit> = all
        .iter()
        .filter(|c| c.unit.starts_with("probe/ouro/sft/") && c.attempt > 1)
        .collect();
    // The episodes run at once, so every one the down server reached was
    // interrupted; each ran again under its own next attempt id.
    assert!(
        !retried.is_empty(),
        "an interrupted episode ran again under a new attempt id"
    );
    assert!(retried.iter().all(|c| c.attempt == 2));
    let units: std::collections::BTreeSet<&str> = retried.iter().map(|c| c.unit.as_str()).collect();
    assert_eq!(units.len(), retried.len(), "no episode committed twice");
}

#[test]
fn a_corrupt_checkpoint_is_caught_and_only_it_is_recomputed() {
    let dir = temp();
    let brain = FakeBrain::new(dir.path());
    go(dir.path(), smoke(), SEED, &brain, None).unwrap();
    let trained = brain.trained();
    std::fs::write(
        dir.path().join("checkpoints/ouro/sft/weights.bin"),
        "flipped",
    )
    .unwrap();
    let shown = go(dir.path(), smoke(), SEED, &brain, None).unwrap();
    assert!(shown.contains("corrupt"), "{shown}");
    assert_eq!(
        brain.trained(),
        trained + 1,
        "the corrupt checkpoint, and nothing else, is trained again"
    );
}

#[test]
fn a_changed_seed_is_a_new_run_and_a_changed_setting_recomputes_only_what_reads_it() {
    let dir = temp();
    let brain = FakeBrain::new(dir.path());
    go(dir.path(), smoke(), SEED, &brain, None).unwrap();
    let trained = brain.trained();

    let mut longer = smoke();
    longer.sft.steps += 5;
    go(dir.path(), longer, SEED, &brain, None).unwrap();
    let all = commits(dir.path());
    assert_eq!(
        brain.trained(),
        trained + 2,
        "the fine-tune and the training after it, nothing before"
    );
    assert_eq!(commit(&all, "ouro/pretrain").attempt, 1);
    let demonstrations = all.iter().filter(|c| c.unit == "demonstrate").count();
    assert_eq!(demonstrations, 1, "the demonstrations are reused");

    let other = temp();
    let fresh = FakeBrain::new(other.path());
    go(other.path(), smoke(), SEED + 1, &fresh, None).unwrap();
    let a = commit(&commits(dir.path()), "world").key.clone();
    let b = commit(&commits(other.path()), "world").key.clone();
    assert_ne!(a, b, "a changed seed is a new identity");
}

#[test]
fn no_promotion_without_a_verified_promote() {
    for (verdict, verified) in [("Reject", true), ("Inconclusive", true), ("Promote", false)] {
        let dir = temp();
        let brain = FakeBrain::new(dir.path());
        let jobs = temp();
        let examiner = FakeExaminer::new(verdict, verified, jobs.path());
        go(dir.path(), smoke(), SEED, &brain, Some(&examiner)).unwrap();
        let all = commits(dir.path());
        assert!(
            !commit(&all, "ouro/experience/verdict").summary["promotes"]
                .as_bool()
                .unwrap(),
            "{verdict} {verified}"
        );
        let report = std::fs::read_to_string(dir.path().join("report.md")).unwrap();
        assert!(report.contains("not promoted"), "{report}");
    }
}

#[test]
fn a_fetched_corpus_is_checked_against_its_pin_and_joins_the_world() {
    let dir = temp();
    let source = dir.path().join("corpus.txt");
    std::fs::write(&source, "The river runs north.\n\nThe road runs south.\n").unwrap();
    let sha = splinter_sdk::vocabulary::digest::Digest::sha256_of(&std::fs::read(&source).unwrap())
        .hex()
        .to_string();
    let corpus = |sha256: &str| splinter_minimind::config::FetchedCorpus {
        name: "local".into(),
        url: format!("file://{}", source.display()),
        sha256: sha256.into(),
        license: "CC0-1.0".into(),
        slice_bytes: 1000,
        separator: "\n\n".into(),
    };
    let run_dir = dir.path().join("run");
    let brain = FakeBrain::new(&run_dir);
    let mut config = smoke();
    config.fetch = vec![corpus(&"0".repeat(64))];
    let refused = go(&run_dir, config.clone(), SEED, &brain, None).unwrap_err();
    assert!(
        format!("{refused:#}").contains("not the pinned"),
        "{refused:#}"
    );
    config.fetch = vec![corpus(&sha)];
    go(&run_dir, config, SEED, &brain, None).unwrap();
    let all = commits(&run_dir);
    assert_eq!(commit(&all, "world").summary["kinds"]["fetched"], 2);
    assert_eq!(commit(&all, "fetch/local").summary["license"], "CC0-1.0");
    // A corpus that marks its documents splits on the marker, which is not
    // trained on.
    std::fs::write(
        &source,
        "One story.\n~~ story ends ~~\nAnother story.\n~~ story ends ~~\nA third.\n",
    )
    .unwrap();
    let sha = splinter_sdk::vocabulary::digest::Digest::sha256_of(&std::fs::read(&source).unwrap())
        .hex()
        .to_string();
    let run_dir = dir.path().join("run-marked");
    let brain = FakeBrain::new(&run_dir);
    let mut config = smoke();
    config.fetch = vec![splinter_minimind::config::FetchedCorpus {
        separator: "~~ story ends ~~".into(),
        ..corpus(&sha)
    }];
    go(&run_dir, config, SEED, &brain, None).unwrap();
    let all = commits(&run_dir);
    assert_eq!(commit(&all, "world").summary["kinds"]["fetched"], 3);
    let world = std::fs::read_to_string(run_dir.join("world/corpus.jsonl")).unwrap();
    assert!(world.contains("Another story.") && !world.contains("~~ story ends ~~"));
}

/// The units that began in `dir` after the first `skip` begin lines.
fn begun_after(dir: &Path, skip: usize) -> Vec<String> {
    std::fs::read_to_string(dir.join("ledger.jsonl"))
        .unwrap()
        .lines()
        .filter(|l| l.contains("\"record\":\"begin\""))
        .skip(skip)
        .map(|l| {
            serde_json::from_str::<serde_json::Value>(l).unwrap()["unit"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect()
}

fn with(name: &str, rev: &str) -> Revisions {
    let mut r = revisions();
    r.0.insert(name.into(), rev.into());
    r
}

#[test]
fn a_change_recomputes_what_consumes_it_and_nothing_else() {
    let dir = temp();
    let brain = FakeBrain::new(dir.path());
    go(dir.path(), smoke(), SEED, &brain, None).unwrap();
    let trained = brain.trained();

    // This repository's own code (the report, the terminal output) is in
    // no stage key: only the manifest is written again.
    let started = begins(dir.path());
    go_at(
        dir.path(),
        smoke(),
        SEED,
        &brain,
        None,
        with("splinter", "r2"),
    )
    .unwrap();
    assert_eq!(begun_after(dir.path(), started), ["init"]);
    assert_eq!(brain.trained(), trained, "nothing is trained");

    // Another sven runs every episode again; the expert's episodes write
    // the same records, so no training run reads anything new.
    let started = begins(dir.path());
    go_at(dir.path(), smoke(), SEED, &brain, None, with("sven", "r2")).unwrap();
    let again = begun_after(dir.path(), started);
    assert!(
        again.iter().any(|u| u.starts_with("demonstrate/train-")),
        "{again:?}"
    );
    assert!(
        !again.iter().any(|u| u == "world" || u == "tasks"),
        "{again:?}"
    );
    assert_eq!(
        brain.trained(),
        trained,
        "the same records train nothing again: {again:?}"
    );

    // Another brain recomputes what brain computes or serves, and none of
    // the data before it.
    let started = begins(dir.path());
    go_at(dir.path(), smoke(), SEED, &brain, None, with("brain", "r2")).unwrap();
    let again = begun_after(dir.path(), started);
    for unit in [
        "ouro/checkpoint",
        "ouro/pretrain",
        "ouro/sft",
        "probe/ouro/sft",
    ] {
        assert!(
            again.iter().any(|u| u == unit),
            "{unit} is recomputed: {again:?}"
        );
    }
    assert!(
        !again
            .iter()
            .any(|u| u == "world" || u == "tasks" || u.starts_with("demonstrate")),
        "the data stages are reused: {again:?}"
    );
}

#[test]
fn the_same_episodes_write_the_same_bytes_whatever_was_interrupted() {
    let read = |dir: &Path, rel: &str| std::fs::read(dir.join(rel)).unwrap();
    let a = temp();
    let brain = FakeBrain::new(a.path());
    go(a.path(), smoke(), SEED, &brain, None).unwrap();

    // A second collection of the same seed, its probe episodes interrupted
    // by a server that went down and run again under new attempts.
    let b = temp();
    let other = FakeBrain::new(b.path());
    other.down.store(true, Ordering::SeqCst);
    *other.down_for.lock().unwrap() = Some("sft".into());
    assert!(go(b.path(), smoke(), SEED, &other, None).is_err());
    other.down.store(false, Ordering::SeqCst);
    go(b.path(), smoke(), SEED, &other, None).unwrap();

    for rel in [
        "world/corpus.jsonl",
        "data/demonstrate.jsonl",
        "data/ouro/explore.jsonl",
    ] {
        assert_eq!(read(a.path(), rel), read(b.path(), rel), "{rel} differs");
    }
    let (ca, cb) = (commits(a.path()), commits(b.path()));
    for unit in ["demonstrate", "probe/ouro/sft", "ouro/sft", "ouro/improve"] {
        assert_eq!(
            commit(&ca, unit).key,
            commit(&cb, unit).key,
            "{unit}'s key differs"
        );
    }
}

#[test]
fn a_pack_that_no_longer_matches_its_episodes_is_unpacked_and_they_run_again() {
    let dir = temp();
    let brain = FakeBrain::new(dir.path());
    go(dir.path(), smoke(), SEED, &brain, None).unwrap();
    // Another setting's episodes packed over these: one record changed.
    let pack = dir.path().join("episodes/demonstrate.jsonl");
    let text = std::fs::read_to_string(&pack).unwrap();
    let mut done = false;
    let changed: String = text
        .lines()
        .map(|line| {
            let mut member: serde_json::Value = serde_json::from_str(line).unwrap();
            if !done && member["path"].as_str().unwrap().ends_with("records.jsonl") {
                done = true;
                member["text"] = format!("{} ", member["text"].as_str().unwrap()).into();
            }
            format!("{member}\n")
        })
        .collect();
    assert!(done, "the pack holds records");
    std::fs::write(&pack, changed).unwrap();
    let started = begins(dir.path());
    go(dir.path(), smoke(), SEED, &brain, None).unwrap();
    let again = begun_after(dir.path(), started);
    assert_eq!(
        again
            .iter()
            .filter(|u| u.starts_with("demonstrate/"))
            .count(),
        1,
        "only the changed episode runs again: {again:?}"
    );
}
