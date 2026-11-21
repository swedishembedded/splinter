// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec of `experiences compare-arms` against golden values computed
//! independently (`tests/fixtures/conformance/arm_compare.json`, written by
//! `generate_arm_compare.py` beside it in exact rational arithmetic).

use super::*;

#[derive(Deserialize)]
struct Golden {
    reference: String,
    k: Vec<usize>,
    level: f64,
    arms: BTreeMap<String, GoldenArm>,
    contrasts: BTreeMap<String, GoldenContrast>,
}

#[derive(Deserialize)]
struct GoldenArm {
    trials: usize,
    graded: usize,
    passed: usize,
    timeouts: usize,
    timeouts_passed: usize,
    infra_errors: usize,
    tasks: usize,
    pass_at_k: BTreeMap<String, GoldenPassAtK>,
}

#[derive(Deserialize)]
struct GoldenPassAtK {
    value: f64,
    tasks: usize,
}

#[derive(Deserialize)]
struct GoldenContrast {
    paired_tasks: usize,
    arm_pass_at_1: f64,
    reference_pass_at_1: f64,
    difference: f64,
    better: usize,
    worse: usize,
    ties: usize,
    families: usize,
    p_one_sided: f64,
    p_two_sided: f64,
    p_one_sided_holm: f64,
    p_two_sided_holm: f64,
    task_cdf: Vec<(f64, f64)>,
    family_cdf: Vec<(f64, f64)>,
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/conformance")
}

fn golden() -> Golden {
    let text = std::fs::read_to_string(fixtures().join("arm_compare.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// A seed no test depends on, printed so a failure can be replayed.
fn drawn_seed() -> Seed {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    println!("seed {seed}");
    Seed::new(seed)
}

/// Resamples, and how far in CDF mass a bound may sit from the exact
/// quantile: five standard errors of an empirical 2.5% quantile.
const RESAMPLES: usize = 20_000;
const CDF_TOLERANCE: f64 = 0.006;

fn is_quantile(cdf: &[(f64, f64)], bound: f64, q: f64) -> Result<(), String> {
    let at = cdf
        .iter()
        .position(|(x, _)| (x - bound).abs() < 1e-9)
        .ok_or_else(|| format!("{bound} is no value the resampled mean can take"))?;
    let below = if at == 0 { 0.0 } else { cdf[at - 1].1 };
    if cdf[at].1 + CDF_TOLERANCE < q || below - CDF_TOLERANCE > q {
        return Err(format!(
            "{bound} is not the {q} quantile: CDF {below} below it, {} at it",
            cdf[at].1
        ));
    }
    Ok(())
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}

/// The first field of `report` that differs from the golden values.
fn check(report: &ArmComparison, golden: &Golden) -> Result<(), String> {
    for arm in &report.arms {
        let want = &golden.arms[&arm.arm];
        let counts = [
            ("trials", arm.trials, want.trials),
            ("graded", arm.graded, want.graded),
            ("passed", arm.passed, want.passed),
            ("timeouts", arm.timeouts, want.timeouts),
            ("timeouts passed", arm.timeouts_passed, want.timeouts_passed),
            ("infrastructure errors", arm.infra_errors, want.infra_errors),
            ("tasks", arm.tasks, want.tasks),
        ];
        for (field, got, want) in counts {
            if got != want {
                return Err(format!("arm {}: {field} {got}, exactly {want}", arm.arm));
            }
        }
        for pk in &arm.pass_at_k {
            let want = &want.pass_at_k[&pk.k.to_string()];
            if pk.tasks != want.tasks || pk.value.is_none_or(|v| !close(v, want.value)) {
                return Err(format!(
                    "arm {}: pass@{} {:?} over {} tasks, exactly {} over {}",
                    arm.arm, pk.k, pk.value, pk.tasks, want.value, want.tasks
                ));
            }
        }
    }
    for c in &report.contrasts {
        let want = &golden.contrasts[&c.arm];
        let counts = [
            ("paired tasks", c.paired_tasks, want.paired_tasks),
            ("better", c.better, want.better),
            ("worse", c.worse, want.worse),
            ("ties", c.ties, want.ties),
            ("families", c.families.unwrap_or(0), want.families),
        ];
        for (field, got, want) in counts {
            if got != want {
                return Err(format!("contrast {}: {field} {got}, exactly {want}", c.arm));
            }
        }
        let values = [
            ("arm pass@1", c.arm_pass_at_1, want.arm_pass_at_1),
            (
                "reference pass@1",
                c.reference_pass_at_1,
                want.reference_pass_at_1,
            ),
            ("difference", c.difference, want.difference),
            ("one-sided p", Some(c.p_one_sided), want.p_one_sided),
            ("two-sided p", Some(c.p_two_sided), want.p_two_sided),
            (
                "one-sided Holm p",
                Some(c.p_one_sided_holm),
                want.p_one_sided_holm,
            ),
            (
                "two-sided Holm p",
                Some(c.p_two_sided_holm),
                want.p_two_sided_holm,
            ),
        ];
        for (field, got, want) in values {
            if got.is_none_or(|g| !close(g, want)) {
                return Err(format!(
                    "contrast {}: {field} {got:?}, exactly {want}",
                    c.arm
                ));
            }
        }
        let tail = (1.0 - golden.level) / 2.0;
        for (name, bounds, cdf) in [
            ("task", c.task_interval, &want.task_cdf),
            ("family", c.family_interval, &want.family_cdf),
        ] {
            let bounds = bounds.ok_or_else(|| format!("contrast {}: no {name} interval", c.arm))?;
            is_quantile(cdf, bounds.low, tail)
                .and_then(|()| is_quantile(cdf, bounds.high, 1.0 - tail))
                .map_err(|e| format!("contrast {}: {name} interval: {e}", c.arm))?;
        }
    }
    Ok(())
}

fn request(golden: &Golden) -> CompareRequest {
    CompareRequest {
        reference: golden.reference.clone(),
        k: golden.k.clone(),
        resamples: RESAMPLES,
        level: golden.level,
        seed: drawn_seed(),
        families: BTreeMap::new(),
    }
}

#[test]
fn the_report_is_the_golden_comparison_of_the_trials() {
    let golden = golden();
    let trials = read_trials_jsonl(&fixtures().join("arm_compare.jsonl")).unwrap();
    let report = compare_arms(&trials, &request(&golden)).unwrap();
    assert_eq!(report.arms.len(), 3);
    assert_eq!(
        report
            .contrasts
            .iter()
            .map(|c| c.arm.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    check(&report, &golden).unwrap();
}

/// The check catches defects of the report (applied to a correct one here;
/// the same defects in the code are the source mutants of this module).
#[test]
fn the_golden_check_catches_a_biased_pass_at_k_a_one_sided_two_sided_test_and_no_holm() {
    let golden = golden();
    let trials = read_trials_jsonl(&fixtures().join("arm_compare.jsonl")).unwrap();
    let good = compare_arms(&trials, &request(&golden)).unwrap();
    let mut mutants: Vec<(&str, ArmComparison)> = Vec::new();
    let mut any_of = good.clone();
    for arm in &mut any_of.arms {
        for pk in arm.pass_at_k.iter_mut().filter(|pk| pk.k > 1) {
            pk.value = Some(1.0);
        }
    }
    mutants.push(("pass@k as any of the attempts", any_of));
    let mut one_as_two = good.clone();
    for c in &mut one_as_two.contrasts {
        c.p_two_sided = c.p_one_sided;
    }
    mutants.push(("one-sided passed off as two-sided", one_as_two));
    let mut no_holm = good.clone();
    for c in &mut no_holm.contrasts {
        c.p_one_sided_holm = c.p_one_sided;
    }
    mutants.push(("no Holm adjustment", no_holm));
    for (name, mutant) in mutants {
        assert!(check(&mutant, &golden).is_err(), "not caught: {name}");
    }
}

fn write_result(job: &Path, trial: &str, task: &str, exception: Option<&str>, reward: Option<f64>) {
    let dir = job.join(trial);
    std::fs::create_dir_all(&dir).unwrap();
    let result = serde_json::json!({
        "task_name": task,
        "trial_name": trial,
        "exception_info": exception.map(|e| serde_json::json!({"exception_type": e})),
        "verifier_result": reward.map(|r| serde_json::json!({"rewards": {"reward": r}})),
    });
    std::fs::write(dir.join("result.json"), result.to_string()).unwrap();
}

/// A Harbor trial passes on reward exactly 1; an agent timeout that was
/// graded is a graded timeout; another exception, or no reward, is an
/// infrastructure error and in no denominator.
#[test]
fn harbor_trials_are_read_as_graded_timeouts_or_infrastructure_errors() {
    let scratch = tempfile::tempdir().unwrap();
    let job = scratch.path().join("job");
    write_result(&job, "a__1", "org/t1", None, Some(1.0));
    write_result(&job, "a__2", "org/t1", None, Some(0.5));
    write_result(&job, "b__1", "org/t2", Some("AgentTimeoutError"), Some(1.0));
    write_result(&job, "b__2", "org/t2", Some("AgentTimeoutError"), None);
    write_result(
        &job,
        "c__1",
        "org/t3",
        Some("EnvironmentStartError"),
        Some(0.0),
    );
    std::fs::create_dir_all(job.join("not-a-trial")).unwrap();
    let trials = read_harbor_job("arm", &job).unwrap();
    let read: Vec<(&str, Status, bool)> = trials
        .iter()
        .map(|t| (t.task.as_str(), t.status, t.passed))
        .collect();
    assert_eq!(
        read,
        [
            ("org/t1", Status::Graded, true),
            ("org/t1", Status::Graded, false),
            ("org/t2", Status::Timeout, true),
            ("org/t2", Status::InfraError, false),
            ("org/t3", Status::InfraError, false),
        ]
    );
    assert!(
        read_harbor_job("arm", scratch.path()).is_err(),
        "no trial is refused"
    );
}

/// A trial a person typed into (bench's `human_guided` play mode) is never
/// read into an arm: its pass is not the agent's.
#[test]
fn a_human_guided_trial_is_refused_in_an_arm() {
    let scratch = tempfile::tempdir().unwrap();
    let job = scratch.path().join("job");
    write_result(&job, "a__1", "org/t1", None, Some(1.0));
    let path = job.join("a__1").join("result.json");
    let mut result: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    result["config"] =
        serde_json::json!({"agent": {"kwargs": {"human_guided": true, "seat": "play"}}});
    std::fs::write(&path, result.to_string()).unwrap();
    let refused = read_harbor_job("arm", &job).unwrap_err().to_string();
    assert!(refused.contains("human-guided"), "{refused}");
}

#[test]
fn a_malformed_trials_line_or_an_unusable_request_is_refused_by_name() {
    let scratch = tempfile::tempdir().unwrap();
    for (line, why) in [
        (r#"{"task": "t", "arm": "a"}"#, "says whether it passed"),
        (r#"{"task": "t", "arm": "a", "reward": 2}"#, "not in [0, 1]"),
        (
            r#"{"task": "t", "arm": "a", "pass": true, "reward": 1}"#,
            "not both",
        ),
        (
            r#"{"task": "t", "arm": "a", "pass": true, "score": 1}"#,
            "unknown field",
        ),
    ] {
        let path = scratch.path().join("t.jsonl");
        std::fs::write(&path, format!("{line}\n")).unwrap();
        let error = read_trials_jsonl(&path).unwrap_err().to_string();
        assert!(error.contains("line 1") && error.contains(why), "{error}");
    }
    let trial = |arm: &str, family: &str| Trial {
        arm: arm.into(),
        task: "t".into(),
        family: Some(family.into()),
        status: Status::Graded,
        passed: true,
    };
    let mut asked = request(&golden());
    asked.k = vec![1];
    assert!(
        compare_arms(&[trial("base", "f")], &asked).is_err(),
        "no other arm"
    );
    assert!(
        compare_arms(&[trial("x", "f"), trial("y", "f")], &asked).is_err(),
        "no reference"
    );
    assert!(
        compare_arms(&[trial("base", "f"), trial("a", "g")], &asked).is_err(),
        "a task in two families"
    );
    asked.k = vec![0];
    assert!(compare_arms(&[trial("base", "f"), trial("a", "f")], &asked).is_err());
}
