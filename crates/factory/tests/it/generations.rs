// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning campaigns whose every generation is
// held to one promotion gate, for its clients. If your team needs expertise
// in continual training of agent models, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Generations of a model as a chain of forked runs: a run's release names
//! the champion it leaves in use (the promoted candidate, else the one it
//! was measured against), with that champion's adapter in the release
//! product; the next generation's recipe names it `champion:RUN`, so the
//! champion attempts the tasks no earlier pass solved and is the baseline
//! its successors must beat, while every product the chain already holds is
//! reused.

use std::path::Path;

use serde_json::Value;
use splinter_factory::resolve::resolve;
use splinter_factory::run::fork;

use crate::factory_run::{decisive, resolved, run, signed_verdict};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::{connections, evaluation, Repo, DATASET, EVALUATION, RECIPE};

fn release(run: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(run.join("release/release.json")).unwrap()).unwrap()
}

/// Generation 1 with promotion, and the recipe of generation 2 after it.
fn generations(first: &Path) -> (String, String) {
    // The chain is what is tested here, not the gates: the candidates are
    // promoted on their target evaluation alone.
    let one = RECIPE.replace(
        "with: { promote: false }",
        "with: { promote: true, requires: [target] }",
    );
    let champion = format!("champion:{}", first.display());
    let two = one
        .replace(
            "  teacher: \"hosted:deepseek/deepseek-v4.1-flash\"\n",
            &format!("  teacher: \"hosted:deepseek/deepseek-v4.1-flash\"\n  champion: \"{champion}\"\n"),
        )
        .replace(
            DATASET,
            // seed: a recipe states every pass's seed; the fake backend draws nothing from it.
            &format!(
                "  - id: g2\n    uses: attempt.collect\n    \
                 inputs: {{ tasks: $stages.tasks.tasks, unsolved_by: [$stages.local.attempts, $stages.remote.attempts] }}\n    \
                 with: {{ agent: sven, model: champion, environment: environments/vim.yaml, attempts: 2, seed: 21 }}\n{DATASET}"
            ),
        )
        .replace(
            "inputs: { attempts: [$stages.local.attempts, $stages.remote.attempts] }",
            "inputs: { attempts: [$stages.local.attempts, $stages.remote.attempts, $stages.g2.attempts] }",
        )
        .replace("baseline: student", "baseline: champion");
    (one, two)
}

#[test]
fn a_generation_is_a_fork_whose_champion_plays_the_frontier_and_is_the_baseline_to_beat() {
    let repo = Repo::new();
    decisive(&repo);
    let backend = FakeBackend {
        candidate: Some(Outcome::Pass),
        ..FakeBackend::with(Outcome::Fail)
    };
    backend.script("train/001-first", &[Outcome::Pass, Outcome::Pass]);
    let trainer = FakeTrainer::default();
    let first = tempfile::tempdir().unwrap();
    let (one, two) = generations(first.path());
    run(&resolved(&repo, &one), first.path(), &backend, &trainer).unwrap();

    // Generation 1 promoted a candidate, and its release carries it.
    let g1 = release(first.path());
    let promoted = g1["promoted"].as_str().unwrap().to_owned();
    assert_eq!(g1["champion"]["candidate"], promoted.as_str());
    assert_eq!(g1["champion"]["base"], "brain:Qwen/Qwen3-4B");
    // The promoted candidate starts the lineage the divergence alarm reads:
    // its target pass rate at generation 0.
    let lineage = g1["lineage"].as_array().unwrap();
    assert_eq!(lineage.len(), 1, "{g1:#}");
    assert_eq!(lineage[0]["generation"], 0);
    assert_eq!(lineage[0]["candidate"], promoted.as_str());
    let adapter = first
        .path()
        .join("release")
        .join(g1["champion"]["adapter"].as_str().unwrap());
    assert!(
        adapter.is_file(),
        "the champion's adapter is in the release product"
    );

    // Generation 2: what generation 1 ran is reused; the champion plays the
    // frontier (the task no pass solved) and is the baseline to beat.
    let second = tempfile::tempdir().unwrap();
    fork(first.path(), second.path()).unwrap();
    let executed = backend.executed().len();
    run(&resolved(&repo, &two), second.path(), &backend, &trainer).unwrap();
    let new = &backend.executed()[executed..];
    let g2: Vec<_> = new.iter().filter(|s| s.slot.pass == "g2").collect();
    assert!(!g2.is_empty() && g2.iter().all(|s| s.task.id == "train/002-second"));
    assert!(
        g2.iter()
            .all(|s| s.model.as_ref().unwrap().model.contains("candidate")),
        "the champion, served, not the base"
    );
    assert!(!new
        .iter()
        .any(|s| s.slot.pass == "local" || s.slot.pass == "remote"));
    assert!(new
        .iter()
        .filter(|s| s.slot.pass.starts_with("baseline-"))
        .all(|s| s.model.as_ref().unwrap().model.contains("candidate")));

    // No candidate beat the champion, so generation 1's champion stays.
    let g2 = release(second.path());
    assert_eq!(g2["promoted"], Value::Null);
    assert_eq!(g2["champion"]["candidate"], promoted.as_str());
    // The champion stays, and so does its lineage: generation 2 read it.
    assert_eq!(g2["lineage"], g1["lineage"]);
    assert_eq!(g2["candidates"][0]["point"]["generation"], 1, "{g2:#}");
    let kept = second
        .path()
        .join("release")
        .join(g2["champion"]["adapter"].as_str().unwrap());
    assert_eq!(
        std::fs::read(kept).unwrap(),
        std::fs::read(adapter).unwrap()
    );
}

#[test]
fn a_candidate_may_continue_a_champions_adapter_when_it_is_served_from_the_same_base() {
    let repo = Repo::new();
    decisive(&repo);
    let backend = FakeBackend {
        candidate: Some(Outcome::Pass),
        ..FakeBackend::with(Outcome::Fail)
    };
    backend.script("train/001-first", &[Outcome::Pass, Outcome::Pass]);
    let trainer = FakeTrainer::default();
    let first = tempfile::tempdir().unwrap();
    let (one, two) = generations(first.path());
    run(&resolved(&repo, &one), first.path(), &backend, &trainer).unwrap();
    let g1 = release(first.path());
    let adapter = first
        .path()
        .join("release")
        .join(g1["champion"]["adapter"].as_str().unwrap());

    // seed: a recipe states every stage's seed; the fake trainer draws nothing from it.
    // A stage that polishes the champion: it trains on the base, starting from the champion's adapter.
    let polish = format!(
        "{}\n  - id: polish\n    uses: model.train\n    inputs: {{ dataset: $stages.sft.dataset }}\n    \
         with: {{ base: student, method: lora, start: champion, params: {{ rank: 4 }}, seed: 5 }}\n",
        two.trim_end()
    );
    assert!(
        polish.contains("start: champion"),
        "the recipe has the stage"
    );
    let second = tempfile::tempdir().unwrap();
    fork(first.path(), second.path()).unwrap();
    run(&resolved(&repo, &polish), second.path(), &backend, &trainer).unwrap();
    let starts = trainer.starts.lock().unwrap().clone();
    assert!(
        starts.iter().any(|s| s
            .as_deref()
            .map(std::fs::read)
            .is_some_and(|b| b.ok() == std::fs::read(&adapter).ok())),
        "the trainer is handed the champion's adapter: {starts:?}"
    );
    // Without start, nothing continues an adapter.
    assert!(starts.iter().filter(|s| s.is_none()).count() >= 1);
}

#[test]
fn a_champion_is_served_never_trained_from_and_must_name_a_finished_run() {
    let repo = Repo::new();
    let nowhere = tempfile::tempdir().unwrap();
    let champion = format!("champion:{}", nowhere.path().display());
    let recipe = RECIPE
        .replace(
            "  teacher: \"hosted:deepseek/deepseek-v4.1-flash\"\n",
            &format!(
                "  teacher: \"hosted:deepseek/deepseek-v4.1-flash\"\n  champion: \"{champion}\"\n"
            ),
        )
        .replace("base: student", "base: champion");
    let problems = resolve(&repo.root, &recipe, connections())
        .expect_err("refused")
        .0;
    let text = problems.join("\n");
    assert!(text.contains("trains from a base model"), "{text}");
    assert!(text.contains("release/release.json"), "{text}");
}

#[test]
fn a_protected_plan_examines_a_candidate_against_the_champion_as_its_predecessor() {
    let repo = Repo::new();
    decisive(&repo);
    let backend = FakeBackend {
        candidate: Some(Outcome::Pass),
        ..FakeBackend::with(Outcome::Fail)
    };
    backend.script("train/001-first", &[Outcome::Pass, Outcome::Pass]);
    let trainer = FakeTrainer::default();
    let first = tempfile::tempdir().unwrap();
    let (one, two) = generations(first.path());
    run(&resolved(&repo, &one), first.path(), &backend, &trainer).unwrap();
    let champion = release(first.path())["champion"]["adapter_digest"]
        .as_str()
        .unwrap()
        .to_owned();

    // Generation 2's candidates, trained, then examined by an authority.
    let second = tempfile::tempdir().unwrap();
    fork(first.path(), second.path()).unwrap();
    let training_only = two.split(EVALUATION).next().unwrap().to_owned();
    run(
        &resolved(&repo, &training_only),
        second.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    let candidate = splinter_factory::run::training::candidate_of(&second.path().join("lora-r8"))
        .unwrap()
        .artifacts
        .into_iter()
        .find(|a| a.path.ends_with(".safetensors"))
        .unwrap()
        .digest
        .to_string();
    // seed: the examiner's test key; the gate checks a signature under it.
    let signer = ed25519_dalek::SigningKey::from_bytes(&[5; 32]);
    let public: String = signer
        .verifying_key()
        .to_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    repo.write("eval/examiner.pub", &public);
    repo.write(
        "eval/verdict.json",
        &signed_verdict(&signer, &candidate, Some(&champion), 0.05, "Promote"),
    );
    // The examiner keeps what it was handed and answers with the verdict.
    repo.write(
        "eval/examine.sh",
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > eval/argv.txt\n\
         while [ $# -gt 0 ]; do case $1 in --out) out=$2;; --predecessor) cp \"$2\" eval/predecessor.json;; esac; shift; done\n\
         cp eval/verdict.json \"$out\"\n",
    );
    // seed: the plan states its seed; the examiner draws nothing from it.
    repo.write(
        "eval/sealed.yaml",
        "schema: splinter.evalplan/v1\nattempts: 1\nseed: 5\ndecision: { alpha: 0.05, comparisons: 1 }\n\
         authority: { command: [sh, eval/examine.sh], public_key: eval/examiner.pub }\n",
    );
    let sealed = training_only.clone()
        + &evaluation("lora-r8", "eval/sealed.yaml", "champion")
        + "  - id: release\n    uses: promotion\n    inputs: { evaluations: [$stages.eval-lora-r8.evaluation] }\n    \
           with: { promote: true, requires: [target] }\n";
    let examined = tempfile::tempdir().unwrap();
    fork(second.path(), examined.path()).unwrap();
    run(
        &resolved(&repo, &sealed),
        examined.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    let predecessor: Value =
        serde_json::from_slice(&std::fs::read(repo.root.join("eval/predecessor.json")).unwrap())
            .unwrap();
    assert_eq!(predecessor["adapter_digest"], champion.as_str());
    let g2 = release(examined.path());
    assert_eq!(
        g2["promoted"], "lora-r8",
        "a verified Promote against the champion promotes"
    );
}
