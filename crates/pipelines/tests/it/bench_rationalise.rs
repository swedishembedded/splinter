// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: the hinted retry (rationalisation) over a compiled Harbor dataset.
//!
//! * The student's failed train-side tasks are worked again as Harbor trials
//!   of staged copies whose instruction carries the hint, through the
//!   examiner's gateway (its state file named to the agent adapter) and
//!   sandbox.
//! * By default an attempt is kept when it is verified and no answer content
//!   is present in the model's own text (the hint's wording, a line of the
//!   task's files nothing showed yet); a mention of "hint" or "expected" is
//!   no leak. The word list, the real-observations rule and the sufficiency
//!   run (its reasoning alone must pass again) are opt-in.
//! * Every record carries every verdict reached, kept or not; the kept
//!   trajectories (rationalised and the student's own passes) are one
//!   experience set an sft-step dataset is built from, and nothing of the
//!   hint reaches it.
//! * A rerun runs nothing again and changes nothing.
//! * A re-grade judges an earlier run's graded attempts again with the
//!   current checks, unpacking its packed job in a copy: by default offline,
//!   no gateway and no Harbor job; with the sufficiency run, that run only.
//! * A student trial of an eval-side task refuses the run by name.
//!
//! The examiner and Harbor are stand-ins: the gateway announces it serves
//! and waits to be stopped, `bench unpack` restores a job packed here;
//! Harbor (a Python script) lays out one trial per staged folder with a
//! scripted trajectory: task a is worked well and its reasoning suffices,
//! task b's reasoning names the hint and the expected result (words, no
//! content) and suffices, task c is worked well but its reasoning alone does
//! not pass, task e repeats the hint's wording (content).

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::json;
use splinter_agent::CancelToken;
use splinter_pipelines::bench::practice::BenchCheckout;
use splinter_pipelines::bench::repair::hinted::filters::{HintFraming, HINT_BEGIN, HINT_PREAMBLE};
use splinter_pipelines::bench::repair::hinted::material::{COMPILE_FILE, COMPILE_SCHEMA};
use splinter_pipelines::bench::repair::hinted::{
    rationalise, Attempts, Checks, Origin, Outcome, RationaliseReport, RationaliseRequest,
};
use splinter_pipelines::datasets::build;

use crate::common::bench::{served, sft_request};

const STARTER: &str = "top line\nbottom line\n";
const TARGET: &str = "bottom line\ntop line\nbottom line\n";

fn instruction(name: &str) -> String {
    format!(
        "# {name}\n\nCopy line two above line one.\n\n**Starter**\n\n```text\n{STARTER}```\n\n\
         **Target**\n\n```text\n{TARGET}```\n"
    )
}

/// A compiled task folder.
fn task(root: &Path, rel: &str, difficulty: i64) {
    let dir = root.join(rel);
    fs::create_dir_all(dir.join("environment/workspace")).unwrap();
    fs::create_dir_all(dir.join("tests/expected")).unwrap();
    fs::create_dir_all(dir.join("solution")).unwrap();
    fs::write(
        dir.join("task.toml"),
        format!("[task]\nname = \"t/{rel}\"\n[metadata]\nbench_difficulty = {difficulty}\n"),
    )
    .unwrap();
    fs::write(dir.join("instruction.md"), instruction(rel)).unwrap();
    fs::write(dir.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(dir.join("environment/workspace/main.txt"), STARTER).unwrap();
    fs::write(dir.join("tests/expected/main.txt"), TARGET).unwrap();
    fs::write(dir.join("tests/test.sh"), "exit 0\n").unwrap();
    fs::write(
        dir.join("solution/solve.sh"),
        "# Reference solution: one call of the editor, as the tool makes it.\nnormal! jyyggP\n",
    )
    .unwrap();
}

fn dataset(root: &Path) -> PathBuf {
    let data = root.join("data");
    task(&data, "train/001-a", 1);
    task(&data, "train/002-b", 2);
    task(&data, "train/003-c", 2);
    task(&data, "eval/004-d", 2);
    task(&data, "train/005-e", 1);
    let entry = |path: &str, set: &str| json!({"name": format!("t/{path}"), "path": path, "digest": format!("d-{path}"), "set": set});
    fs::write(
        data.join(COMPILE_FILE),
        json!({"schema": COMPILE_SCHEMA, "bench_commit": "abc", "tasks": [
            entry("train/001-a", "train"), entry("train/002-b", "train"),
            entry("train/003-c", "train"), entry("eval/004-d", "eval"),
            entry("train/005-e", "train"),
        ]})
        .to_string(),
    )
    .unwrap();
    data
}

/// The scripted trajectory and first request of a student's trial of
/// `instruction`.
fn trial_files(dir: &Path, instruction: &str) {
    fs::create_dir_all(dir.join("agent")).unwrap();
    let first = "I print main.txt to see it.";
    let trajectory = json!({
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "sven", "version": "1"},
        "steps": [
            {"step_id": 1, "source": "user", "message": instruction},
            {"step_id": 2, "source": "agent", "model_name": "Qwen/Qwen3-4B", "message": "",
             "reasoning_content": first,
             "tool_calls": [{"tool_call_id": "c1", "function_name": "vim-vim",
                             "arguments": {"commands": ["args main.txt", "%print"]}}],
             "observation": {"results": [{"source_call_id": "c1",
                             "content": "exit status: 0\noutput:\ntop line\nbottom line"}]}},
            {"step_id": 3, "source": "agent", "model_name": "Qwen/Qwen3-4B", "message": "",
             "reasoning_content": "Line two goes above line one: yank it and put it above.",
             "tool_calls": [{"tool_call_id": "c2", "function_name": "vim-vim",
                             "arguments": {"commands": ["args main.txt", "normal! jyyggP", "wqall"]}}],
             "observation": {"results": [{"source_call_id": "c2",
                             "content": "exit status: 0\nmessages:\n\"main.txt\" 3L, 33B written"}]}},
            {"step_id": 4, "source": "agent", "model_name": "Qwen/Qwen3-4B", "message": "Done."},
        ],
    });
    fs::write(dir.join("agent/trajectory.json"), trajectory.to_string()).unwrap();
    let request = json!({"kind": "request", "seq": 1, "model": "main", "request": {
        "messages": [{"role": "system", "content": "You are an agent with one vim tool."},
                     {"role": "user", "content": instruction}],
        "tools": [{"type": "function", "function": {"name": "vim-vim", "parameters": {}}}]}});
    fs::write(dir.join("agent/requests.jsonl"), format!("{request}\n")).unwrap();
}

fn student_trial(job: &Path, data: &Path, name: &str, rel: &str, reward: f64) {
    let dir = job.join(name);
    fs::create_dir_all(&dir).unwrap();
    let result = json!({"trial_name": name, "config": {"task": {"path": data.join(rel)}},
        "verifier_result": {"rewards": {"reward": reward}}, "exception_info": null});
    fs::write(dir.join("result.json"), result.to_string()).unwrap();
    trial_files(
        &dir,
        &fs::read_to_string(data.join(rel).join("instruction.md")).unwrap(),
    );
}

/// The stand-in examiner and Harbor; every call is logged to
/// `root/calls.log`.
fn checkout(root: &Path) -> BenchCheckout {
    let dir = root.join("bench");
    fs::create_dir_all(&dir).unwrap();
    let calls = root.join("calls.log");
    let bench = format!(
        "#!/bin/sh\necho \"bench $*\" >> '{calls}'\ncase \"$1\" in\ngateway)\n    \
         echo \"bench gateway: serving\" >&2\n    exec sleep 600 ;;\nunpack)\n    \
         mv \"$2\"/.packed/* \"$2\"/ && rmdir \"$2/.packed\" && rm \"$2/trials.jsonl\" ;;\nesac\n",
        calls = calls.display()
    );
    let harbor = format!(
        r#"#!/usr/bin/env -S python3 -I
import json, os, sys
args = sys.argv[1:]
with open({calls:?}, "a") as log:
    log.write("harbor " + " ".join(args) + "\n")
value = lambda flag: args[args.index(flag) + 1]
dataset, out, name = value("-p"), value("-o"), value("--job-name")
job = os.path.join(out, name)
os.makedirs(job, exist_ok=True)
for n, folder in enumerate(sorted(os.listdir(dataset))):
    task = os.path.join(dataset, folder)
    instruction = open(os.path.join(task, "instruction.md")).read()
    hinted = {hint:?} in instruction
    if hinted:
        reward = 1.0
        if folder.endswith("002-b"):
            first = "The hint shows the expected result; as the reference solution has it, I print main.txt."
        elif folder.endswith("005-e"):
            first = "It is one call of the editor, as the tool makes it; I print main.txt."
        else:
            first = "I print main.txt to see it."
    else:
        reward = 1.0 if folder.endswith(("001-a", "002-b")) else 0.0
        first = "I print main.txt to see it."
    trial = os.path.join(job, folder[:20] + "__x" + str(n))
    os.makedirs(os.path.join(trial, "agent"), exist_ok=True)
    json.dump({{"task": {{"path": task}}}}, open(os.path.join(trial, "config.json"), "w"))
    json.dump({{"trial_name": os.path.basename(trial), "exception_info": None,
               "verifier_result": {{"rewards": {{"reward": reward}}}}}},
              open(os.path.join(trial, "result.json"), "w"))
    steps = [
        {{"step_id": 1, "source": "user", "message": instruction}},
        {{"step_id": 2, "source": "agent", "model_name": "Qwen/Qwen3-4B", "message": "",
         "reasoning_content": first,
         "tool_calls": [{{"tool_call_id": "c1", "function_name": "vim-vim",
                         "arguments": {{"commands": ["args main.txt", "%print"]}}}}],
         "observation": {{"results": [{{"source_call_id": "c1",
                         "content": "exit status: 0\noutput:\ntop line\nbottom line"}}]}}}},
        {{"step_id": 3, "source": "agent", "model_name": "Qwen/Qwen3-4B", "message": "",
         "reasoning_content": "Line two goes above line one: yank it and put it above.",
         "tool_calls": [{{"tool_call_id": "c2", "function_name": "vim-vim",
                         "arguments": {{"commands": ["args main.txt", "normal! jyyggP", "wqall"]}}}}],
         "observation": {{"results": [{{"source_call_id": "c2",
                         "content": "exit status: 0\nmessages:\n\"main.txt\" 3L, 33B written"}}]}}}},
        {{"step_id": 4, "source": "agent", "model_name": "Qwen/Qwen3-4B", "message": "Done."}},
    ]
    json.dump({{"schema_version": "ATIF-v1.7", "agent": {{"name": "sven", "version": "1"}},
               "steps": steps}}, open(os.path.join(trial, "agent", "trajectory.json"), "w"))
    request = {{"kind": "request", "seq": 1, "model": "main", "request": {{
        "messages": [{{"role": "system", "content": "You are an agent with one vim tool."}},
                     {{"role": "user", "content": instruction}}],
        "tools": [{{"type": "function", "function": {{"name": "vim-vim", "parameters": {{}}}}}}]}}}}
    open(os.path.join(trial, "agent", "requests.jsonl"), "w").write(json.dumps(request) + "\n")
json.dump({{"finished_at": "2026-11-21T00:00:00"}}, open(os.path.join(job, "result.json"), "w"))
"#,
        calls = calls.display().to_string(),
        hint = HINT_BEGIN,
    );
    for (name, script) in [("bench", bench), ("harbor", harbor)] {
        let path = dir.join(name);
        fs::write(&path, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        }
    }
    fs::write(root.join("sven"), "").unwrap();
    BenchCheckout {
        harbor: dir.join("harbor"),
        dir,
        sven_build: Some(root.join("sven")),
    }
}

const PROFILE: &str =
    "[sven]\nbuild = \"${BENCH_SVEN_BUILD}\"\n[sven.sampling]\ntemperature = 1.0\n\
[run]\nenvironment = \"userns\"\nsession = \"x\"\n[roles]\nmain = \"brain\"\n[upstreams.brain]\n\
base_url = \"http://127.0.0.1:1/v1\"\nmodel = \"other\"\napi_key_env = \"BRAIN_KEY\"\n";

fn harbor_calls(root: &Path) -> Vec<String> {
    fs::read_to_string(root.join("calls.log"))
        .unwrap_or_default()
        .lines()
        .filter(|l| l.starts_with("harbor "))
        .map(str::to_string)
        .collect()
}

/// The words a run may opt into with `--leak-term`: the list gates a and b
/// named.
const WORDS: [&str; 4] = ["as given", "solve.sh", "reference", "expected"];

/// Everything a test's runs share.
struct Bed {
    _scratch: crate::common::Scratch,
    ctx: splinter_orchestrator::context::Context,
    rationaliser: splinter_core::model_ref::ModelRef,
    root: PathBuf,
    data: PathBuf,
    students: Vec<PathBuf>,
    checkout: BenchCheckout,
    profile: PathBuf,
    /// Any seed: four failed tasks and a count of four pick them all.
    seed: u64,
}

/// Four failed train tasks (a, b, c, e) and one pass of c.
fn bed(name: &str) -> Bed {
    let (scratch, ctx, rationaliser) = served(name);
    let root = scratch.0.clone();
    let data = dataset(&root);
    let student = root.join("student");
    student_trial(&student, &data, "a__1", "train/001-a", 0.0);
    student_trial(&student, &data, "b__1", "train/002-b", 0.0);
    student_trial(&student, &data, "c__1", "train/003-c", 0.0);
    student_trial(&student, &data, "c__2", "train/003-c", 1.0);
    student_trial(&student, &data, "e__1", "train/005-e", 0.0);
    let checkout = checkout(&root);
    let profile = root.join("base.toml");
    fs::write(&profile, PROFILE).unwrap();
    Bed {
        _scratch: scratch,
        ctx,
        rationaliser,
        root,
        data,
        students: vec![student],
        checkout,
        profile,
        seed: splinter_model::seed::draw().value(),
    }
}

impl Bed {
    fn run(
        &self,
        attempts: Attempts<'_>,
        terms: &[String],
        checks: Checks,
        served: bool,
    ) -> RationaliseReport {
        let images = self.root.join("images");
        rationalise(
            &self.ctx,
            &RationaliseRequest {
                dataset: &self.data,
                students: &self.students,
                checkout: &self.checkout,
                rationaliser: served.then_some(&self.rationaliser),
                profile: served.then_some(self.profile.as_path()),
                images: &images,
                attempts,
                jobs: 2,
                leak_terms: terms,
                framing: HintFraming::default(),
                checks,
                cancel: CancelToken::new(),
            },
        )
        .unwrap()
    }

    fn draw(&self, terms: &[String], checks: Checks) -> RationaliseReport {
        let seed = self.seed;
        self.run(
            Attempts::Draw {
                count: 4,
                samples: 1,
                seed,
            },
            terms,
            checks,
            true,
        )
    }
}

fn words() -> Vec<String> {
    WORDS.iter().map(|t| (*t).to_string()).collect()
}

fn record(
    report: &RationaliseReport,
    folder: &str,
    origin: Origin,
) -> splinter_pipelines::bench::repair::hinted::RationaliseRecord {
    report
        .records
        .iter()
        .find(|r| r.folder == folder && r.origin == origin)
        .unwrap()
        .clone()
}

fn gateway_calls(root: &Path) -> usize {
    fs::read_to_string(root.join("calls.log"))
        .unwrap_or_default()
        .lines()
        .filter(|l| l.starts_with("bench gateway"))
        .count()
}

#[test]
fn with_every_check_opted_in_failed_train_tasks_are_filtered_and_kept_without_the_hint() {
    let bed = bed("rationalise");
    let root = &bed.root;
    let terms = words();
    let report = bed.draw(&terms, Checks::ALL);
    assert_eq!(report.picked.len(), 4);
    assert_eq!(report.students.failed_tasks, 4);
    let f = &report.funnel;
    assert_eq!(
        (
            f.graded,
            f.verified,
            f.real_observations,
            f.no_leak,
            f.sufficient
        ),
        (4, 4, 4, 2, 1),
        "{:#?}",
        report.records
    );
    assert_eq!(
        (report.own, report.rationalised, report.undecided),
        (1, 1, 0)
    );
    // The student's own pass mentions no leak term: the base rate is zero.
    assert_eq!(report.own_term_mentions.len(), terms.len());
    assert!(report.own_term_mentions.values().all(|n| *n == 0));
    assert_eq!(report.own_real_observations, 1);
    let a = record(&report, "train/001-a", Origin::Rationalised);
    assert_eq!(a.outcome, Outcome::Kept);
    assert!(a.filters.sufficient.as_ref().unwrap().passed && a.experience.is_some());
    let b = record(&report, "train/002-b", Origin::Rationalised);
    assert_eq!(b.outcome, Outcome::Rejected);
    let leak = b.filters.no_leak.as_ref().unwrap();
    assert!(!leak.passed && leak.reason.as_ref().unwrap().contains("expected"));
    assert!(
        b.filters.sufficient.is_none(),
        "no sufficiency run after a leak"
    );
    let e = record(&report, "train/005-e", Origin::Rationalised);
    let leak = e.filters.no_leak.as_ref().unwrap();
    assert!(!leak.passed && leak.reason.as_ref().unwrap().contains("wording"));
    let c = record(&report, "train/003-c", Origin::Rationalised);
    assert_eq!(c.outcome, Outcome::Rejected);
    assert!(!c.filters.sufficient.as_ref().unwrap().passed);
    assert_eq!(
        record(&report, "train/003-c", Origin::Own).outcome,
        Outcome::Kept
    );

    // Two Harbor jobs, each a plain run naming the gateway's state file and
    // port; the gateway wrote its state where the agent adapter reads it.
    let calls = harbor_calls(root);
    assert_eq!(calls.len(), 2, "{calls:?}");
    let log = fs::read_to_string(root.join("calls.log")).unwrap();
    assert!(
        log.contains("gateway serve --profile") && log.contains("--state"),
        "{log}"
    );
    for call in &calls {
        assert!(call.contains("--ak gateway_state=") && call.contains("--ek allow_ports="));
        assert!(call.contains("--agent agents.sven:SvenAgent") && call.contains("-k 1"));
    }
    // The sufficiency run saw the reasoning and no hint.
    let sufficiency = fs::read_dir(report.dir.clone())
        .unwrap()
        .filter_map(Result::ok)
        .find(|e| e.file_name().to_string_lossy().starts_with("sufficiency-"))
        .unwrap()
        .path();
    for staged in fs::read_dir(sufficiency.join("tasks")).unwrap() {
        let staged = staged.unwrap().path();
        let text = fs::read_to_string(staged.join("instruction.md")).unwrap();
        assert!(text.contains("yank it and put it above") && !text.contains(HINT_BEGIN));
        assert!(!staged.join("solution").exists());
    }

    // The kept set builds into sft-step records that never show the hint.
    let set = report.set.clone().unwrap();
    let built = build(&bed.ctx, &sft_request(set.clone())).unwrap();
    assert!(built.records > 0, "{built:?}");
    let records = fs::read_to_string(&built.path).unwrap();
    assert!(records.contains("Copy line two above line one."));
    assert!(!records.contains(HINT_BEGIN) && !records.contains(HINT_PREAMBLE));
    assert!(!records.contains("Reference solution"));

    // A rerun runs nothing and keeps the same set.
    let again = bed.draw(&terms, Checks::ALL);
    assert_eq!(harbor_calls(root).len(), 2);
    assert_eq!(again.set, Some(set));
    assert_eq!(again.records, report.records);
}

/// The default is two checks: verified, and no answer content in the model's text. Naming the hint is no leak, no
/// real-observations verdict is reached, and no sufficiency trial runs: a passing attempt is kept at once.
#[test]
fn by_default_an_attempt_is_kept_when_verified_and_no_answer_content_is_present() {
    let bed = bed("rationalise-default");
    let report = bed.draw(&[], Checks::default());
    assert_eq!(harbor_calls(&bed.root).len(), 1, "only the hinted job runs");
    let kept: Vec<_> = ["train/001-a", "train/002-b", "train/003-c"]
        .into_iter()
        .map(|f| record(&report, f, Origin::Rationalised))
        .collect();
    for r in &kept {
        assert_eq!(r.outcome, Outcome::Kept, "{r:#?}");
        assert!(r.filters.real_observations.is_none() && r.filters.sufficient.is_none());
        assert!(r.sufficiency_trial.is_none() && r.experience.is_some());
    }
    let e = record(&report, "train/005-e", Origin::Rationalised);
    assert_eq!(e.outcome, Outcome::Rejected);
    assert!(e
        .filters
        .no_leak
        .as_ref()
        .unwrap()
        .reason
        .as_ref()
        .unwrap()
        .contains("wording"));
    let f = &report.funnel;
    assert_eq!(
        (f.graded, f.verified, f.no_leak, f.sufficient),
        (4, 4, 3, 3)
    );
    assert_eq!((report.own, report.rationalised), (1, 3));
    assert_eq!(report.checks, Checks::default());
}

/// Packs a Harbor job the way the stand-in `bench unpack` restores it: its
/// trial directories moved aside, their results in `trials.jsonl`.
fn pack(job: &Path) {
    let packed = job.join(".packed");
    fs::create_dir_all(&packed).unwrap();
    let mut rows = String::new();
    for entry in fs::read_dir(job).unwrap() {
        let path = entry.unwrap().path();
        if !path.join("result.json").is_file() {
            continue;
        }
        let result: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join("result.json")).unwrap()).unwrap();
        rows.push_str(&format!("{}\n", json!({"result": result})));
        fs::rename(&path, packed.join(path.file_name().unwrap())).unwrap();
    }
    fs::write(job.join("trials.jsonl"), rows).unwrap();
}

/// The Harbor job directories of a run's hinted phase.
fn hinted_jobs(run: &Path) -> Vec<PathBuf> {
    let mut jobs = Vec::new();
    for entry in fs::read_dir(run).unwrap().filter_map(Result::ok) {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("hinted-") {
            jobs.push(entry.path().join("jobs").join(name));
        }
    }
    jobs
}

#[test]
fn a_regrade_judges_an_earlier_runs_attempts_again_offline_and_keeps_its_evidence() {
    let bed = bed("rationalise-regrade");
    let root = &bed.root;
    // The earlier run: every check opted in, the word list named.
    let earlier = bed.draw(&words(), Checks::ALL);
    assert_eq!(earlier.rationalised, 1);
    let jobs = hinted_jobs(&earlier.dir);
    assert_eq!(jobs.len(), 1);
    pack(&jobs[0]);
    let before = (harbor_calls(root).len(), gateway_calls(root));

    // Re-graded with the default checks, no endpoint and no profile: offline.
    let runs = [earlier.dir.clone()];
    let report = bed.run(Attempts::Regrade(&runs), &[], Checks::default(), false);
    assert_eq!(
        (harbor_calls(root).len(), gateway_calls(root)),
        before,
        "no Harbor job, no gateway"
    );
    assert!(fs::read_to_string(root.join("calls.log"))
        .unwrap()
        .contains("bench unpack"));
    // The earlier run's evidence is as it was: still packed.
    assert!(jobs[0].join("trials.jsonl").is_file() && jobs[0].join(".packed").is_dir());
    assert_eq!(report.regraded, runs.to_vec());
    assert!(report.picked.is_empty());
    let attempts: Vec<_> = report
        .records
        .iter()
        .filter(|r| r.origin == Origin::Rationalised)
        .collect();
    assert_eq!(attempts.len(), 4);
    for r in &attempts {
        let source = r.regrade_of.as_ref().unwrap();
        assert_eq!(source.run, earlier.dir);
        let original = earlier
            .records
            .iter()
            .find(|o| o.key == source.key)
            .unwrap();
        assert_ne!(r.key, original.key);
        assert_eq!((&r.folder, r.sample), (&original.folder, original.sample));
        assert_eq!(r.rationaliser, original.rationaliser);
        assert!(r.hinted_trial.as_ref().unwrap().starts_with(&report.dir));
    }
    for folder in ["train/001-a", "train/002-b", "train/003-c"] {
        assert_eq!(
            record(&report, folder, Origin::Rationalised).outcome,
            Outcome::Kept,
            "{folder}"
        );
    }
    assert_eq!(
        record(&report, "train/005-e", Origin::Rationalised).outcome,
        Outcome::Rejected
    );
    assert_eq!((report.funnel.graded, report.rationalised), (4, 3));

    // A rerun of the re-grade changes nothing.
    let again = bed.run(Attempts::Regrade(&runs), &[], Checks::default(), false);
    assert_eq!(again.records, report.records);

    // With the sufficiency run opted in, a re-grade runs that run only: no
    // hinted job.
    let checks = Checks {
        sufficiency: true,
        ..Checks::default()
    };
    let with = bed.run(Attempts::Regrade(&runs), &[], checks, true);
    let calls = harbor_calls(root);
    assert_eq!(calls.len(), before.0 + 1, "{calls:?}");
    assert!(
        calls.last().unwrap().contains("sufficiency-")
            && !calls.last().unwrap().contains("hinted-")
    );
    assert_eq!(
        record(&with, "train/003-c", Origin::Rationalised).outcome,
        Outcome::Rejected
    );
    assert_eq!(with.rationalised, 2);
}

#[test]
fn a_student_trial_of_an_eval_task_refuses_the_run_by_name() {
    let (scratch, ctx, rationaliser) = served("rationalise-eval");
    let root = scratch.0.clone();
    let data = dataset(&root);
    let student = root.join("student");
    student_trial(&student, &data, "a__1", "train/001-a", 0.0);
    student_trial(&student, &data, "d__1", "eval/004-d", 0.0);
    let checkout = checkout(&root);
    let profile = root.join("base.toml");
    fs::write(&profile, PROFILE).unwrap();
    let error = rationalise(
        &ctx,
        &RationaliseRequest {
            dataset: &data,
            students: &[student],
            checkout: &checkout,
            rationaliser: Some(&rationaliser),
            profile: Some(&profile),
            images: &root,
            attempts: Attempts::Draw {
                count: 2,
                samples: 1,
                seed: 0, // seed: the run is refused before anything is drawn
            },
            jobs: 1,
            leak_terms: &[],
            framing: HintFraming::default(),
            checks: Checks::default(),
            cancel: CancelToken::new(),
        },
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("d__1") && error.contains("004-d"), "{error}");
    assert!(error.contains("eval side"), "{error}");
    assert!(harbor_calls(&root).is_empty());
}
