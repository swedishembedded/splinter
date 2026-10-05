// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-assisted harmonisation of survey and
// cohort records for its clients. If your team needs expertise in pooling
// heterogeneous health data under auditable rules, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Intake measured against a harmonisation already checked by hand.
//!
//! Every exam concept in [`EXAM`] is a set of NHANES variables, each with a
//! factor and a valid range a reviewer checked against the codebooks. That
//! is ground truth for the intake agent: given one cycle's codebook entry
//! for one of those variables, and the concepts with the values the *other*
//! variables (or, for a variable never renamed, the other cycles) take, does
//! a proposal land on the same concept and factor, and do the admission
//! rules let the right mappings through and stop the wrong ones?
//!
//! The admission rules are measured without a model: the true mapping must
//! be admitted, and each corruption of it - the unit factor off by ten,
//! refusal codes read as values, the variable assigned to another concept -
//! counts as caught when it is rejected. With a served model, its proposals
//! are scored against the truth as well.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Serialize;
use splinter_sdk::agent::mapper::SvenMapper;
use splinter_sdk::agent::solve::Model;
use splinter_sdk::knowledge::codebook::{parse_html, VariableDoc};
use splinter_sdk::knowledge::harmonize::{
    admit, non_measurement_codes, ConceptSpec, MappingProposal, MappingProposer, Reference,
};

use crate::concepts::{Numeric, EXAM};
use crate::nhanes::{mortality, mortality_file, Cycle, CYCLES};

/// How long one proposal may take, corrections included.
const CALL_DEADLINE: Duration = Duration::from_secs(600);

/// One cycle's variable for one concept, with its true mapping.
struct Case {
    cycle: u16,
    concept: usize,
    doc: VariableDoc,
    values: Vec<Option<f64>>,
    truth: MappingProposal,
}

/// Every variable's codebook entry in the cycle directory, the first file in
/// name order winning, as for the data.
fn codebooks(dir: &Path) -> Result<HashMap<String, VariableDoc>> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("htm")))
        .collect();
    files.sort();
    let mut out = HashMap::new();
    for path in files {
        let html = String::from_utf8_lossy(
            &std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
        )
        .into_owned();
        for doc in parse_html(&html) {
            out.entry(doc.name.clone()).or_insert(doc);
        }
    }
    Ok(out)
}

/// Every case, over the participants eligible for mortality linkage - the
/// cohort the hand mapping was written for (its valid ranges are adults').
fn cases(nhanes: &Path, mortality_dir: &Path) -> Result<Vec<Case>> {
    let mut out = Vec::new();
    for (start, suffix) in CYCLES {
        let cycle = Cycle::load(nhanes, start, suffix)?;
        let (linked, _) = mortality(&mortality_file(mortality_dir, start))?;
        let cohort: Vec<u64> = cycle
            .participants()
            .into_iter()
            .filter(|s| linked.get(s).is_some_and(|m| m.eligible))
            .collect();
        let docs = codebooks(&nhanes.join(start.to_string()))?;
        for (k, n) in EXAM.iter().enumerate() {
            let Some(&(var, factor)) = n.vars.iter().find(|(v, _)| cycle.column(v).is_some())
            else {
                continue;
            };
            let values: Vec<Option<f64>> = cohort.iter().map(|s| cycle.num(var, *s)).collect();
            let Some(doc) = docs.get(var) else { continue };
            let truth = MappingProposal {
                concept: n.name.into(),
                factor,
                valid_min: n.valid.0,
                valid_max: n.valid.1,
                missing_codes: non_measurement_codes(doc),
                quote: if doc.label.is_empty() {
                    doc.name.clone()
                } else {
                    doc.label.clone()
                },
            };
            out.push(Case {
                cycle: start,
                concept: k,
                doc: doc.clone(),
                values,
                truth,
            });
        }
    }
    Ok(out)
}

fn reference_of(values: &mut [f64]) -> Option<Reference> {
    if values.len() < 10 {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let q = |p: f64| values[((values.len() - 1) as f64 * p).round() as usize];
    Some(Reference {
        q05: q(0.05),
        median: q(0.5),
        q95: q(0.95),
    })
}

/// The concepts as offered for `case`: each with the values of every other
/// case of it, converted by the truth - and for the case's own concept,
/// only the cases under another variable name, or (never renamed) from
/// other cycles, so the case is never its own reference.
fn concepts_for(all: &[Case], case: &Case) -> Vec<ConceptSpec> {
    EXAM.iter()
        .enumerate()
        .map(|(k, n): (usize, &Numeric)| {
            let others = all.iter().filter(|c| c.concept == k);
            let renamed = all
                .iter()
                .any(|c| c.concept == k && c.doc.name != case.doc.name);
            let mut values: Vec<f64> = others
                .filter(|c| {
                    k != case.concept
                        || if renamed {
                            c.doc.name != case.doc.name
                        } else {
                            c.cycle != case.cycle
                        }
                })
                .flat_map(|c| {
                    let t = &c.truth;
                    c.values
                        .iter()
                        .flatten()
                        .map(move |x| x * t.factor)
                        .filter(move |x| (t.valid_min..=t.valid_max).contains(x))
                })
                .collect();
            ConceptSpec {
                name: n.name.into(),
                unit: n.unit.into(),
                description: n.name.replace('_', " "),
                reference: reference_of(&mut values),
            }
        })
        .collect()
}

/// How often one kind of mapping was admitted.
#[derive(Default, Serialize)]
struct Tally {
    tried: usize,
    admitted: usize,
}

impl Tally {
    fn add(&mut self, admitted: bool) {
        self.tried += 1;
        self.admitted += usize::from(admitted);
    }
}

#[derive(Default, Serialize)]
struct RuleReport {
    cases: usize,
    true_mapping: Tally,
    factor_times_ten: Tally,
    factor_over_ten: Tally,
    codes_read_as_values: Tally,
    other_concept: Tally,
    /// The true mappings rejected, and why: each is a rule too strict for
    /// real data, or a truth that is wrong.
    rejected_truths: Vec<String>,
    /// The unit-factor corruptions admitted: errors the rules missed.
    admitted_unit_errors: Vec<String>,
}

fn rules(all: &[Case]) -> RuleReport {
    let mut r = RuleReport {
        cases: all.len(),
        ..Default::default()
    };
    for case in all {
        let concepts = concepts_for(all, case);
        let check = |p: &MappingProposal| admit(p, &case.doc, &case.values, &concepts);
        let t = &case.truth;
        let a = check(t);
        r.true_mapping.add(a.admitted);
        if !a.admitted {
            r.rejected_truths.push(format!(
                "{} {} -> {}: {}",
                case.cycle,
                case.doc.name,
                t.concept,
                a.reasons.join("; ")
            ));
        }
        for (scale, tally) in [
            (10.0, &mut r.factor_times_ten),
            (0.1, &mut r.factor_over_ten),
        ] {
            let a = check(&MappingProposal {
                factor: t.factor * scale,
                ..t.clone()
            });
            tally.add(a.admitted);
            if a.admitted {
                r.admitted_unit_errors.push(format!(
                    "{} {} -> {} x{scale}: converted {:?}",
                    case.cycle, case.doc.name, t.concept, a.converted
                ));
            }
        }
        if let Some(top) = t.missing_codes.iter().copied().reduce(f64::max) {
            let leaky = MappingProposal {
                missing_codes: vec![],
                valid_max: t.valid_max.max(top * t.factor + 1.0),
                ..t.clone()
            };
            r.codes_read_as_values.add(check(&leaky).admitted);
        }
        for n in EXAM.iter().filter(|n| n.name != t.concept) {
            r.other_concept.add(
                check(&MappingProposal {
                    concept: n.name.into(),
                    valid_min: n.valid.0,
                    valid_max: n.valid.1,
                    ..t.clone()
                })
                .admitted,
            );
        }
    }
    r
}

#[derive(Serialize)]
struct Scored {
    cycle: u16,
    variable: String,
    truth: MappingProposal,
    proposal: Option<MappingProposal>,
    error: Option<String>,
    concept_right: bool,
    factor_right: bool,
    admitted: bool,
    reasons: Vec<String>,
}

/// Every case's proposal, the model asked once per distinct codebook entry
/// (most variables are documented identically in every cycle). The model is
/// shown the concepts' names, units and descriptions but not their reference
/// distributions: it maps from the codebook alone, and the references stay
/// with the admission rules, which run per case on that case's own values.
fn propose_all(all: &[Case], model: Model) -> Result<Vec<Scored>> {
    let mapper = SvenMapper::new(model, CALL_DEADLINE);
    let runtime = tokio::runtime::Runtime::new().context("starting the async runtime")?;
    let offered: Vec<ConceptSpec> = concepts_for(all, &all[0])
        .into_iter()
        .map(|c| ConceptSpec {
            reference: None,
            ..c
        })
        .collect();
    let mut asked: HashMap<String, Result<MappingProposal, String>> = HashMap::new();
    let mut out = Vec::new();
    for case in all {
        let key = case.doc.full_text();
        let got = match asked.get(&key) {
            Some(g) => g.clone(),
            None => {
                let g = runtime.block_on(mapper.propose(&case.doc, &offered));
                eprintln!(
                    "asked about {} ({} distinct so far)",
                    case.doc.name,
                    asked.len() + 1
                );
                asked.insert(key, g.clone());
                g
            }
        };
        let t = &case.truth;
        let scored = match got {
            Ok(p) => {
                let a = admit(&p, &case.doc, &case.values, &concepts_for(all, case));
                Scored {
                    cycle: case.cycle,
                    variable: case.doc.name.clone(),
                    concept_right: p.concept == t.concept,
                    factor_right: p.concept == t.concept
                        && (p.factor / t.factor - 1.0).abs() < 0.01,
                    admitted: a.admitted,
                    reasons: a.reasons,
                    truth: t.clone(),
                    proposal: Some(p),
                    error: None,
                }
            }
            Err(e) => Scored {
                cycle: case.cycle,
                variable: case.doc.name.clone(),
                truth: t.clone(),
                proposal: None,
                error: Some(e),
                concept_right: false,
                factor_right: false,
                admitted: false,
                reasons: vec![],
            },
        };
        eprintln!(
            "{} {:<10} concept {} factor {} admitted {}",
            scored.cycle,
            scored.variable,
            scored.concept_right,
            scored.factor_right,
            scored.admitted
        );
        out.push(scored);
    }
    Ok(out)
}

/// Where a model to propose with is served.
pub struct Served {
    /// Its OpenAI-compatible address.
    pub base_url: String,
    /// The key it accepts.
    pub api_key: String,
    /// The model's name.
    pub model: String,
}

/// Measure the admission rules on every case under `nhanes` (cohort from
/// the linkage files in `mortality_dir`), and a served
/// model's proposals when one is given; reports go to `out`.
pub fn intake(
    nhanes: &Path,
    mortality_dir: &Path,
    out: &Path,
    served: Option<&Served>,
) -> Result<()> {
    std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let all = cases(nhanes, mortality_dir)?;
    let report = rules(&all);
    println!(
        "{} cases: true mapping admitted {}/{}; wrongly admitted: factor x10 {}/{}, factor /10 {}/{}, codes as values {}/{}, other concept {}/{}",
        report.cases,
        report.true_mapping.admitted,
        report.true_mapping.tried,
        report.factor_times_ten.admitted,
        report.factor_times_ten.tried,
        report.factor_over_ten.admitted,
        report.factor_over_ten.tried,
        report.codes_read_as_values.admitted,
        report.codes_read_as_values.tried,
        report.other_concept.admitted,
        report.other_concept.tried,
    );
    for r in &report.rejected_truths {
        println!("  true mapping rejected: {r}");
    }
    for r in &report.admitted_unit_errors {
        println!("  unit error admitted: {r}");
    }
    std::fs::write(
        out.join("rules.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    let Some(s) = served else { return Ok(()) };
    let loaded =
        splinter_sdk::model::selection::served_model(&s.base_url, &s.api_key, &s.model, Some(0.0))?;
    let model = Model::new(loaded.provider(), loaded.identity());
    let identity = model.identity.replace('/', "_");
    let scored = propose_all(&all, model)?;
    let n = scored.len();
    let count = |f: fn(&Scored) -> bool| scored.iter().filter(|s| f(s)).count();
    println!(
        "{}: concept right {}/{n}, factor right {}/{n}, admitted {}/{n}, admitted and wrong {}/{n}",
        s.model,
        count(|s| s.concept_right),
        count(|s| s.factor_right),
        count(|s| s.admitted),
        count(|s| s.admitted && !s.factor_right),
    );
    let lines: Vec<String> = scored
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<_, _>>()?;
    std::fs::write(
        out.join(format!("proposals-{identity}.jsonl")),
        lines.join("\n") + "\n",
    )?;
    Ok(())
}
