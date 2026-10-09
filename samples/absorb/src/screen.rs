// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements screening of fact pools against a model's own
// answers so that what is measured was unknown to it, for its clients. If
// your team needs expertise in designing measurements a learning system
// cannot game, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `facts screen`: each candidate fact put to the day-0 policy six times
//! (greedy, then five draws), every answer graded by both checks, each fact
//! classed, and the roles' quotas filled in a fixed order.
//!
//! A fact is consistently wrong only when neither check passes on any of the
//! six answers, the judge measures precise on that fact's own hard negatives
//! (its day-0 wrong answers, a corrupted statement, the true statement and a
//! reworded one), and all four of its probes also fail at day 0: a probe
//! succeeds only with its keys, the judge's yes, and no repeat of the wrong
//! claim the day-0 answers made. The policy answers everything first and the
//! judge grades afterwards, so only one base is on the device at a time.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Context as _;
use serde::Serialize;

use crate::answers::{collect, Collected, Plan, SAMPLES};
use crate::facts::{corrupt, Fact, Manifest};
use crate::grading::{classify, grade_keys, trace, wrong_entities, Grade, Judge};
use crate::keys::{missing, Key};
use crate::policy::{Answer, Decoding, Policy, SAMPLE_TEMPERATURE};
use crate::probes::{Probe, ProbeKind};
use crate::roles::{select, Class, Quotas, Role, Screened};
use crate::runtime;

/// Where the screening report is written.
const REPORT: &str = "screening.json";

/// The precision a fact's own hard negatives must show the judge has.
pub const MIN_FACT_PRECISION: f64 = 0.95;

/// What `facts screen` needs.
pub struct Request {
    /// The run's output directory.
    pub out: PathBuf,
    /// The model store, when not the configuration's.
    pub models: Option<PathBuf>,
    /// The policy's model reference.
    pub policy: String,
    /// The judge's model reference.
    pub judge: String,
    /// Screen only this many facts of the pool, in the fixed order of a hash
    /// of the seed and the fact's id; all of them when absent.
    pub candidates: Option<usize>,
    /// Quotas to fill in place of the manifest's, for a smaller run; they
    /// replace the manifest's and are recorded in it.
    pub quotas: Option<Quotas>,
}

/// One graded answer of the report.
#[derive(Serialize)]
struct Graded {
    n: usize,
    decoding: Decoding,
    answer: Option<String>,
    conclusion: String,
    seconds: f64,
    #[serde(flatten)]
    grade: Grade,
    passes: bool,
}

/// One item the judge was measured on for a fact.
#[derive(Serialize)]
struct Control {
    what: &'static str,
    text: String,
    should_pass: bool,
    judge: Option<bool>,
}

/// One probe's day-0 result.
#[derive(Serialize)]
struct ProbeResult {
    kind: ProbeKind,
    question: String,
    answer: Option<String>,
    missing_keys: Vec<String>,
    judge: Option<bool>,
    repeats_wrong_claim: bool,
    succeeds: bool,
}

/// One screened fact of the report.
#[derive(Serialize)]
struct Entry {
    id: String,
    family: String,
    candidate_role: Role,
    question: String,
    statement: String,
    keys: Vec<String>,
    class: Class,
    discarded_because: Option<String>,
    passed: usize,
    role: Option<Role>,
    day: Option<usize>,
    answers: Vec<Graded>,
    wrong_claim: Vec<String>,
    judge_controls: Vec<Control>,
    judge_precision: Option<f64>,
    probes: Vec<ProbeResult>,
}

/// What the screening found.
#[derive(Debug, Serialize)]
pub struct Summary {
    /// Facts put to the policy.
    pub screened: usize,
    /// How many came out in each class.
    pub by_class: BTreeMap<String, usize>,
    /// How many filled each role.
    pub by_role: BTreeMap<String, usize>,
    /// Seconds the policy took to answer, then the judge to grade.
    pub answer_seconds: u64,
    /// Seconds spent grading.
    pub grade_seconds: u64,
}

/// The facts to screen, in a fixed order that depends on the seed and the
/// ids alone.
fn in_screening_order(manifest: &Manifest, candidates: Option<usize>) -> Vec<&Fact> {
    let mut facts: Vec<&Fact> = manifest.facts.iter().collect();
    facts.sort_by_key(|f| {
        let mut input = manifest.seed.to_le_bytes().to_vec();
        input.extend_from_slice(b"screen:");
        input.extend_from_slice(f.id.as_bytes());
        (blake3::hash(&input).as_bytes()[..8].to_vec(), f.id.clone())
    });
    facts.truncate(candidates.unwrap_or(usize::MAX));
    facts
}

/// The solver identity a judged control is given: no model's.
const CONTROL_SOLVER: &str = "absorb/control";

/// Grades one fact: its six answers, and for a fact that looks wrong its
/// judge controls and probes.
struct Grading<'a> {
    ctx: &'a splinter_sdk::Context,
    judge: &'a Judge,
    solver: &'a str,
    collected: &'a Collected,
    donors: &'a [Key],
}

impl Grading<'_> {
    fn verdict(
        &self,
        solver: &str,
        question: &str,
        statement: &str,
        text: &str,
    ) -> anyhow::Result<Option<bool>> {
        self.judge
            .verdict(self.ctx, solver, question, statement, text)
    }

    fn entry(&self, fact: &Fact) -> anyhow::Result<Entry> {
        let mut graded = Vec::new();
        for n in 0..=SAMPLES {
            let line = self
                .collected
                .lines
                .iter()
                .find(|l| l.fact == fact.id && l.n == n)
                .with_context(|| format!("fact {} has no answer #{n}", fact.id))?;
            graded.push(self.answer(fact, n, line.decoding, &line.answer)?);
        }
        let grades: Vec<Grade> = graded.iter().map(|g| g.grade.clone()).collect();
        let mut class = classify(&grades);
        let mut entry = Entry {
            id: fact.id.clone(),
            family: fact.family.clone(),
            candidate_role: fact.candidate_role,
            question: fact.question.clone(),
            statement: fact.statement.clone(),
            keys: fact.keys.iter().map(|k| k.text.clone()).collect(),
            class,
            discarded_because: None,
            passed: graded.iter().filter(|g| g.passes).count(),
            role: None,
            day: None,
            answers: graded,
            wrong_claim: Vec::new(),
            judge_controls: Vec::new(),
            judge_precision: None,
            probes: Vec::new(),
        };
        if class == Class::ConsistentlyWrong && !fact.candidate_role.wants_wrong() {
            class = Class::Discarded;
            entry.class = class;
            entry.discarded_because = Some(
                "wrong at day 0, but its family is a candidate for a role of facts the policy knows".into(),
            );
        } else if class == Class::ConsistentlyWrong {
            if let Some(why) = self.probe_and_calibrate(fact, &mut entry)? {
                class = Class::Discarded;
                entry.class = class;
                entry.discarded_because = Some(why);
            }
        }
        eprintln!(
            "fact {}: {}/{} pass both checks, {class:?}{}",
            fact.id,
            entry.passed,
            entry.answers.len(),
            entry
                .discarded_because
                .as_deref()
                .map_or(String::new(), |w| format!(" ({w})"))
        );
        Ok(entry)
    }

    fn answer(
        &self,
        fact: &Fact,
        n: usize,
        decoding: Decoding,
        answer: &Answer,
    ) -> anyhow::Result<Graded> {
        let grade = match answer.text.as_deref() {
            Some(text) => {
                let mut grade = grade_keys(text, &fact.keys);
                grade.judge = self.verdict(self.solver, &fact.question, &fact.statement, text)?;
                grade
            }
            None => Grade {
                missing_keys: fact.keys.iter().map(|k| k.text.clone()).collect(),
                judge: Some(false),
            },
        };
        Ok(Graded {
            n,
            decoding,
            answer: answer.text.clone(),
            conclusion: answer.conclusion.clone(),
            seconds: answer.seconds,
            passes: grade.passes(),
            grade,
        })
    }

    /// Measures the judge on the fact's own hard negatives and the probes at
    /// day 0. Returns why the fact must be discarded, if it must.
    fn probe_and_calibrate(
        &self,
        fact: &Fact,
        entry: &mut Entry,
    ) -> anyhow::Result<Option<String>> {
        if let Some(why) = self.collected.unwritable.get(&fact.id) {
            return Ok(Some(format!(
                "the generator could not write admissible items for it: {why}"
            )));
        }
        let Some(reworded) = self.collected.reworded.get(&fact.id) else {
            return Ok(Some(
                "no reworded statement was made for the judge to be measured on".into(),
            ));
        };
        let day_zero: Vec<&str> = entry
            .answers
            .iter()
            .filter_map(|g| g.answer.as_deref())
            .collect();
        entry.wrong_claim = wrong_entities(&day_zero, &fact.question, &fact.statement);

        let mut controls = vec![
            ("true statement", fact.statement.clone(), true),
            ("reworded statement", reworded.clone(), true),
        ];
        if let Some(false_one) = corrupt(&fact.statement, &fact.keys, self.donors) {
            controls.push(("corrupted statement", false_one, false));
        }
        for g in &entry.answers {
            if let Some(text) = &g.answer {
                controls.push(("day-0 answer", text.clone(), false));
            }
        }
        let mut right = 0usize;
        for (what, text, should_pass) in controls {
            let judge = self.verdict(CONTROL_SOLVER, &fact.question, &fact.statement, &text)?;
            right += usize::from(judge == Some(should_pass));
            entry.judge_controls.push(Control {
                what,
                text,
                should_pass,
                judge,
            });
        }
        let precision = right as f64 / entry.judge_controls.len() as f64;
        entry.judge_precision = Some(precision);
        if precision < MIN_FACT_PRECISION {
            return Ok(Some(format!(
                "the judge is right on {right} of {} of this fact's own hard negatives",
                entry.judge_controls.len()
            )));
        }

        let probes: Vec<&Probe> = self
            .collected
            .probes
            .iter()
            .filter(|p| p.fact == fact.id)
            .collect();
        if probes.len() != ProbeKind::ALL.len() {
            return Ok(Some(format!(
                "{} probes were written, not {}",
                probes.len(),
                ProbeKind::ALL.len()
            )));
        }
        for probe in probes {
            let line = self
                .collected
                .probe_lines
                .iter()
                .find(|l| l.fact == fact.id && l.kind == probe.kind)
                .with_context(|| {
                    format!(
                        "probe {:?} of fact {} was not answered",
                        probe.kind, fact.id
                    )
                })?;
            let text = line.answer.text.clone();
            let lacking: Vec<String> = match &text {
                Some(t) => missing(t, &probe.keys)
                    .into_iter()
                    .map(|k| k.text.clone())
                    .collect(),
                None => probe.keys.iter().map(|k| k.text.clone()).collect(),
            };
            let judge = match &text {
                Some(t) => self.verdict(self.solver, &probe.question, &fact.statement, t)?,
                None => Some(false),
            };
            let repeats = text
                .as_deref()
                .is_some_and(|t| trace(t, &entry.wrong_claim));
            let succeeds = lacking.is_empty() && judge == Some(true) && !repeats;
            entry.probes.push(ProbeResult {
                kind: probe.kind,
                question: probe.question.clone(),
                answer: text,
                missing_keys: lacking,
                judge,
                repeats_wrong_claim: repeats,
                succeeds,
            });
        }
        Ok(entry
            .probes
            .iter()
            .find(|p| p.succeeds)
            .map(|p| format!("its {:?} probe already succeeds at day 0", p.kind)))
    }
}

/// Screens the pool; see the module documentation.
///
/// # Errors
/// A model cannot run, the judge is not precise enough, or the screened
/// facts cannot fill the roles' quotas (the report and the classes are
/// written first, so a larger `--candidates` continues from them).
pub fn run(request: &Request) -> anyhow::Result<Summary> {
    let mut manifest = Manifest::read(&request.out)?;
    anyhow::ensure!(
        manifest.probes.is_none(),
        "the probes are sealed: the screening is closed"
    );
    if let Some(quotas) = request.quotas {
        manifest.quotas = quotas;
    }
    let splinter = runtime::open(&request.out, request.models.as_ref())?;
    let ctx = splinter.context();
    let facts: Vec<Fact> = in_screening_order(&manifest, request.candidates)
        .into_iter()
        .cloned()
        .collect();

    // Phase one: the policy answers, and the generator writes what it must.
    let started = std::time::Instant::now();
    let policy_ref = runtime::model_ref(&request.policy)?;
    let policy = Policy::load(&ctx, &policy_ref, &manifest.persona)?;
    let generator = ctx.model(&runtime::model_ref(&manifest.generator)?)?;
    let collected = collect(
        &ctx,
        &policy,
        &Plan {
            out: &request.out,
            facts: &facts,
            persona: &manifest.persona,
            generator: &generator,
        },
    )?;
    let answer_seconds = started.elapsed().as_secs();
    let solver = policy.identity().to_string();
    drop(policy);

    // Phase two: the judge, measured on controls, then grading.
    let started = std::time::Instant::now();
    let judge_ref = runtime::model_ref(&request.judge)?;
    let calibrated = Judge::calibrate(&ctx, &judge_ref, &manifest.task_set)?;
    eprintln!(
        "judge {}: {} controls, precision pass {:?} fail {:?}, abstains {:?}",
        calibrated.calibration.judge,
        calibrated.calibration.controls,
        calibrated.calibration.precision_pass,
        calibrated.calibration.precision_fail,
        calibrated.calibration.abstain_rate
    );
    manifest.judge = Some(calibrated.calibration.clone());
    let donors: Vec<Key> = manifest.facts.iter().flat_map(|f| f.keys.clone()).collect();
    let grading = Grading {
        ctx: &ctx,
        judge: &calibrated.judge,
        solver: &solver,
        collected: &collected,
        donors: &donors,
    };
    let mut entries = Vec::new();
    for fact in &facts {
        entries.push(grading.entry(fact)?);
    }
    let grade_seconds = started.elapsed().as_secs();

    // Roles: filled in a fixed order, at most one fact of a family, or the
    // pool is too small.
    let screened: Vec<Screened> = entries
        .iter()
        .filter_map(|e| facts.iter().find(|f| f.id == e.id).map(|f| (e, f)))
        .map(|(e, f)| Screened {
            id: e.id.clone(),
            family: f.family.clone(),
            entities: f.entities(),
            candidate: e.candidate_role,
            class: e.class,
        })
        .collect();
    let placed = select(manifest.seed, &manifest.quotas, &screened);
    for entry in &entries {
        if let Some(fact) = manifest.facts.iter_mut().find(|f| f.id == entry.id) {
            fact.class = Some(entry.class);
        }
    }
    if let Ok(placed) = &placed {
        for p in placed {
            if let Some(fact) = manifest.facts.iter_mut().find(|f| f.id == p.id) {
                fact.role = Some(p.role);
                fact.day = p.day;
            }
            if let Some(entry) = entries.iter_mut().find(|e| e.id == p.id) {
                entry.role = Some(p.role);
                entry.day = p.day;
            }
        }
    }
    let mut by_class: BTreeMap<String, usize> = BTreeMap::new();
    for e in &entries {
        *by_class.entry(format!("{:?}", e.class)).or_default() += 1;
    }
    let mut by_role: BTreeMap<String, usize> = BTreeMap::new();
    for e in entries.iter().filter_map(|e| e.role) {
        *by_role.entry(e.name().to_string()).or_default() += 1;
    }
    let summary = Summary {
        screened: entries.len(),
        by_class,
        by_role,
        answer_seconds,
        grade_seconds,
    };
    let report = serde_json::json!({
        "policy": solver,
        "persona": manifest.persona,
        "decoding": {"greedy": 1, "sampled": SAMPLES, "temperature": SAMPLE_TEMPERATURE},
        "judge": calibrated.calibration,
        "judge_misjudged_controls": calibrated.misjudged,
        "seed": manifest.seed,
        "quotas": manifest.quotas,
        "summary": summary,
        "roles_filled": placed.is_ok(),
        "shortfall": placed.as_ref().err().map(ToString::to_string),
        "facts": entries,
    });
    std::fs::write(
        request.out.join(REPORT),
        serde_json::to_string_pretty(&report)? + "\n",
    )?;
    manifest.write(&request.out)?;
    placed.map_err(|short| {
        anyhow::anyhow!(
            "{short} (screening report: {})",
            request.out.join(REPORT).display()
        )
    })?;
    Ok(summary)
}
