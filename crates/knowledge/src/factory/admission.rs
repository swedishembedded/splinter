// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that decide whether a
// training environment may be trained on, for its clients. If your team
// needs expertise in validating generated environments and their verifiers
// before a campaign spends compute on them, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Admission: what an environment must show before a campaign may train on
//! it.
//!
//! An environment is sampled (train and dev instances drawn from a recorded
//! seed) and every sampled instance is held to the gates below. The gates
//! are the bench's own, restated for training environments, plus the two a
//! training campaign adds: an adversarial attempt on the verifier and a
//! measured base pass-rate band.
//!
//! | Gate | Meaning |
//! |---|---|
//! | `structural` | the folder is a valid artifact: split train or dev, id matches, the solver is shown neither tests nor solution |
//! | `reference-passes` | the reference solution, run in the workspace, is graded solved |
//! | `nop-fails` | doing nothing is graded, and not solved (a crash is not a grade) |
//! | `robust` | grading is repeatable on one state; the reference is repeatable; reset restores the unsolved state; a missing answer is graded, not a crash |
//! | `reproducible` | the same request gives a byte-identical folder and the same rewards; a plugin's second oracle agrees |
//! | `novel` | instances differ from each other, dev from train, and from anything already trained on |
//! | `red-team` | no attack in [`attack`](super::attack) earns reward |
//! | `base-band` | a base policy's measured pass rate lies in the band a campaign can learn from |
//!
//! The status that follows is [`Status::Admitted`] only when all pass.
//! A beaten verifier is [`Status::Quarantined`]: never trained on, and kept
//! on record with the attack that beat it. An environment whose gates pass
//! but whose base pass rate was not measured is [`Status::PendingBand`], and
//! [`AdmissionReport::may_enter_campaign`] is false for it.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_core::rng::SplitMix64;
use splinter_core::verifiability::{Split, VerifierClass};

use crate::tasks::dedup::Seen;

use super::artifact::{environment_id, Instance, SCHEMA_VERSION};
use super::attack::{generic_attacks, leaks, run_attacks, AttackKind, AttackOutcome};
use super::band::{judge_band, BandMeasurement, BandPolicy, BandResult, BandStatus};
use super::host::{Host, HostError, Session, Verdict};
use super::plugin::{resolve_difficulty, InstanceRequest, Plugin, PluginError, Reference};

/// How an environment is sampled and held.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdmissionPolicy {
    /// How many train and how many dev instances are drawn and tested.
    pub instances_per_split: usize,
    /// The largest share of instances a blind attack (a guess, a scrambled
    /// answer, another instance's answer) may beat before the verifier is
    /// quarantined.
    pub max_blind_rate: f64,
    /// The base pass rates a campaign can learn from.
    pub band: BandPolicy,
}

impl Default for AdmissionPolicy {
    fn default() -> Self {
        Self {
            instances_per_split: 4,
            max_blind_rate: 0.1,
            band: BandPolicy::default(),
        }
    }
}

/// One gate's verdict.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateResult {
    /// The gate's name.
    pub name: String,
    /// Whether it held for every sampled instance.
    pub passed: bool,
    /// What was seen, naming the instance (by split and seed) that failed.
    pub detail: String,
}

/// An attack that earned reward.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Beaten {
    /// The split of the instance attacked.
    pub split: Split,
    /// Its seed.
    pub seed: u64,
    /// The attack's name.
    pub attack: String,
    /// What the verifier said.
    pub detail: String,
}

/// Where an environment stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    /// Every gate passed and the base pass rate is in its band.
    Admitted,
    /// Every gate passed; the base pass rate has not been measured.
    PendingBand,
    /// A gate failed or the base pass rate is outside its band.
    Rejected,
    /// An attack earned reward: the verifier cannot be trusted.
    Quarantined,
}

/// What admission found out about an environment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdmissionReport {
    /// The artifact schema version.
    pub schema_version: u32,
    /// The environment's id.
    pub environment: String,
    /// The plugin and its version.
    pub plugin: String,
    /// The plugin's version.
    pub plugin_version: String,
    /// The difficulty knobs.
    pub difficulty: BTreeMap<String, u32>,
    /// How its rewards are established.
    pub verifier_class: VerifierClass,
    /// What a verdict on it must say beside its number.
    pub label: Option<String>,
    /// The seed the sample was drawn from.
    pub admission_seed: u64,
    /// The seeds of the train instances tested.
    pub train_seeds: Vec<u64>,
    /// The seeds of the dev instances tested.
    pub dev_seeds: Vec<u64>,
    /// The gates, in order.
    pub gates: Vec<GateResult>,
    /// How many attacks were run in all.
    pub attacks_run: usize,
    /// The attacks that earned reward.
    pub beaten: Vec<Beaten>,
    /// The share of instances a blind attack beat.
    pub blind_rate: f64,
    /// The measured base pass rate, when there is one.
    pub band: Option<BandResult>,
    /// The policy the band was held to.
    pub band_policy: BandPolicy,
    /// Where the environment stands.
    pub status: Status,
    /// Why, in words.
    pub reasons: Vec<String>,
}

impl AdmissionReport {
    /// Whether a campaign may train on this environment.
    #[must_use]
    pub fn may_enter_campaign(&self) -> bool {
        self.status == Status::Admitted
    }
}

/// Why admission could not be carried out. A failing environment is a
/// report, not an error.
#[derive(Debug, thiserror::Error)]
pub enum AdmitError {
    /// The plugin refused the request.
    #[error(transparent)]
    Plugin(#[from] PluginError),
    /// A script could not be run.
    #[error(transparent)]
    Host(#[from] HostError),
}

/// A sampled instance with what running it taught.
struct Sample {
    instance: Instance,
    reference: Reference,
    request: InstanceRequest,
}

fn verdict_text(verdict: &Verdict) -> String {
    match verdict {
        Verdict::Graded(v) => format!("{v:?}"),
        Verdict::NoReward { exit, tail } => format!("no reward ({exit}): {tail}"),
        Verdict::Malformed(why) => format!("malformed reward: {why}"),
    }
}

fn label(split: Split, seed: u64) -> String {
    format!("{split}-{seed}")
}

/// Seeds for the train and dev samples, distinct from each other.
fn draw_seeds(admission_seed: u64, per_split: usize) -> (Vec<u64>, Vec<u64>) {
    let mut rng = SplitMix64::new(admission_seed);
    let mut used = BTreeSet::new();
    let mut take = |n: usize| {
        let mut seeds = Vec::with_capacity(n);
        while seeds.len() < n {
            let seed = rng.next_u64();
            if used.insert(seed) {
                seeds.push(seed);
            }
        }
        seeds
    };
    let train = take(per_split);
    let dev = take(per_split);
    (train, dev)
}

/// Collects one gate's failures.
struct GateBuilder {
    name: &'static str,
    failures: Vec<String>,
    ok: String,
}

impl GateBuilder {
    fn new(name: &'static str, ok: &str) -> Self {
        Self {
            name,
            failures: Vec::new(),
            ok: ok.to_string(),
        }
    }

    fn fail(&mut self, split: Split, seed: u64, why: impl std::fmt::Display) {
        self.failures.push(format!("{}: {why}", label(split, seed)));
    }

    fn fail_general(&mut self, why: &str) {
        self.failures.push(why.to_string());
    }

    fn finish(self) -> GateResult {
        GateResult {
            name: self.name.into(),
            passed: self.failures.is_empty(),
            detail: if self.failures.is_empty() {
                self.ok
            } else {
                self.failures.join("; ")
            },
        }
    }
}

fn run_reference(
    host: &Host,
    instance: &Instance,
) -> Result<(Session, Reference, Verdict), HostError> {
    let session = host.open(instance)?;
    let before = session.workspace()?;
    let solved = session.solve()?;
    let after = session.workspace()?;
    let verdict = if solved.succeeded() {
        session.verify()?
    } else {
        Verdict::NoReward {
            exit: format!("solve exited {:?}", solved.exit_code),
            tail: solved.stderr,
        }
    };
    Ok((session, Reference::between(&before, &after), verdict))
}

/// Samples `plugin` at `difficulty`, holds every sampled instance to the
/// gates and the red team, and reports. `known` holds the versions of
/// instances already trained on, which a new one must not repeat. The
/// instances are returned with the report so that a caller may store them.
///
/// `admission_seed` is the caller's drawn, recorded seed; the instance seeds
/// follow from it.
pub fn admit(
    plugin: &dyn Plugin,
    difficulty: &BTreeMap<String, u32>,
    host: &Host,
    policy: &AdmissionPolicy,
    admission_seed: u64,
    known: &BTreeSet<Digest>,
    band: Option<&BandMeasurement>,
) -> Result<(AdmissionReport, Vec<Instance>), AdmitError> {
    let difficulty = resolve_difficulty(plugin.id(), plugin.knobs(), difficulty)?;
    let environment = environment_id(plugin.id(), plugin.version(), &difficulty);
    let (train, dev) = draw_seeds(admission_seed, policy.instances_per_split);

    let mut samples = Vec::new();
    for (split, seeds) in [(Split::Train, &train), (Split::Dev, &dev)] {
        for &seed in seeds {
            let request = InstanceRequest {
                split,
                seed,
                difficulty: difficulty.clone(),
            };
            let instance = plugin.generate(&request)?;
            samples.push(Sample {
                instance,
                reference: Reference::default(),
                request,
            });
        }
    }

    let mut structural = GateBuilder::new(
        "structural",
        "valid artifacts, hidden material kept from the solver",
    );
    let mut reference_passes = GateBuilder::new(
        "reference-passes",
        "the reference is graded solved on every instance",
    );
    let mut nop_fails = GateBuilder::new(
        "nop-fails",
        "doing nothing is graded and fails on every instance",
    );
    let mut robust = GateBuilder::new(
        "robust",
        "grading and the reference are repeatable; reset restores; a missing answer is graded",
    );
    let mut reproducible = GateBuilder::new(
        "reproducible",
        "the same request gives the same folder and rewards",
    );
    let mut novel = GateBuilder::new(
        "novel",
        "instances differ from each other and from what was trained on",
    );

    for sample in &mut samples {
        let (split, seed) = (sample.request.split, sample.request.seed);
        check_structure(&mut structural, plugin, sample, &environment);

        // nop
        let nop = host.open(&sample.instance)?.verify()?;
        if !matches!(nop, Verdict::Graded(_)) {
            nop_fails.fail(
                split,
                seed,
                format!(
                    "the verifier did not grade an empty attempt: {}",
                    verdict_text(&nop)
                ),
            );
        } else if nop.solved() {
            nop_fails.fail(split, seed, "an untouched workspace is graded solved");
        }

        // the reference
        let (session, reference, verdict) = run_reference(host, &sample.instance)?;
        if !verdict.solved() {
            reference_passes.fail(
                split,
                seed,
                format!("the reference is not solved: {}", verdict_text(&verdict)),
            );
        }
        sample.reference = reference;
        let solved_state = session.workspace_digest()?;
        check_robustness(&mut robust, plugin, sample, &session)?;

        // reproducibility
        let again = plugin.generate(&sample.request)?;
        if again.version() != sample.instance.version() {
            reproducible.fail(
                split,
                seed,
                "generating the same request twice gave different folders",
            );
        }
        let (second, second_reference, second_verdict) = run_reference(host, &sample.instance)?;
        if second_verdict != verdict
            || second_reference != sample.reference
            || second.workspace_digest()? != solved_state
        {
            reproducible.fail(
                split,
                seed,
                "running the reference twice gave different states or rewards",
            );
        }
        if let Some(Err(why)) = plugin.cross_check(&sample.request, host) {
            reproducible.fail(split, seed, format!("second oracle disagrees: {why}"));
        }
    }

    check_novelty(&mut novel, plugin, &samples, known);

    // The red team. Instances are attacked in parallel, each on its own
    // session; results come back in sample order.
    let references: Vec<Reference> = samples.iter().map(|s| s.reference.clone()).collect();
    let attacked = attack_all(plugin, host, &samples, &references)?;
    let mut attacks_run = 0;
    let mut beaten = Vec::new();
    let mut blind_beaten: BTreeSet<(Split, u64)> = BTreeSet::new();
    for (sample, (static_leaks, outcomes)) in samples.iter().zip(attacked) {
        let (split, seed) = (sample.request.split, sample.request.seed);
        for found in static_leaks {
            beaten.push(Beaten {
                split,
                seed,
                attack: format!("leak:static:{found}"),
                detail: "the reference answer stands in what the solver is shown".into(),
            });
        }
        attacks_run += outcomes.len();
        for outcome in outcomes.into_iter().filter(|o| o.beaten) {
            if outcome.kind == AttackKind::Blind {
                blind_beaten.insert((split, seed));
            }
            beaten.push(Beaten {
                split,
                seed,
                attack: outcome.name,
                detail: outcome.detail,
            });
        }
    }
    let blind_rate = blind_beaten.len() as f64 / samples.len().max(1) as f64;
    let hard_beaten = beaten.iter().any(|b| {
        !b.attack.starts_with("constant:")
            && !b.attack.starts_with("format-only")
            && !b.attack.starts_with("cross-instance")
    });
    let quarantined = hard_beaten || blind_rate > policy.max_blind_rate;
    let red_team = GateResult {
        name: "red-team".into(),
        passed: !quarantined,
        detail: if quarantined {
            format!(
                "{} attack(s) earned reward in {attacks_run} run; blind attacks beat {:.0}% of instances (limit {:.0}%): {}",
                beaten.len(),
                blind_rate * 100.0,
                policy.max_blind_rate * 100.0,
                beaten.iter().take(4).map(|b| format!("{} on {}", b.attack, label(b.split, b.seed))).collect::<Vec<_>>().join(", ")
            )
        } else {
            format!("{attacks_run} attacks run, none beat the verifier beyond the blind-rate limit (blind rate {:.0}%)", blind_rate * 100.0)
        },
    };

    let mut gates = vec![
        structural.finish(),
        reference_passes.finish(),
        nop_fails.finish(),
        robust.finish(),
        reproducible.finish(),
        novel.finish(),
        red_team,
    ];
    let band_result = band.and_then(BandMeasurement::result);
    let (band_gate, band_status) = judge_band(band, band_result.as_ref(), policy.band);
    let hygiene_ok = gates.iter().all(|g| g.passed);
    if let Some(gate) = band_gate {
        gates.push(gate);
    }
    let mut reasons: Vec<String> = gates
        .iter()
        .filter(|g| !g.passed)
        .map(|g| format!("{}: {}", g.name, g.detail))
        .collect();
    let status = if quarantined {
        Status::Quarantined
    } else if !hygiene_ok {
        Status::Rejected
    } else {
        match band_status {
            BandStatus::Unmeasured => {
                reasons
                    .push("base pass rate not measured: not admissible to a campaign yet".into());
                Status::PendingBand
            }
            BandStatus::Outside => Status::Rejected,
            BandStatus::Inside => Status::Admitted,
        }
    };

    let report = AdmissionReport {
        schema_version: SCHEMA_VERSION,
        environment,
        plugin: plugin.id().into(),
        plugin_version: plugin.version().into(),
        difficulty,
        verifier_class: plugin.verifier_class(),
        label: plugin.verifier_class().label().map(ToString::to_string),
        admission_seed,
        train_seeds: train,
        dev_seeds: dev,
        gates,
        attacks_run,
        beaten,
        blind_rate,
        band: band_result,
        band_policy: policy.band,
        status,
        reasons,
    };
    Ok((report, samples.into_iter().map(|s| s.instance).collect()))
}

type Attacked = (Vec<String>, Vec<AttackOutcome>);

/// Every sample's static leaks and attack outcomes, the samples attacked on
/// as many threads as the machine has (at most eight).
fn attack_all(
    plugin: &dyn Plugin,
    host: &Host,
    samples: &[Sample],
    references: &[Reference],
) -> Result<Vec<Attacked>, HostError> {
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .clamp(1, 8);
    let attack_one = |index: usize| -> Result<Attacked, HostError> {
        let sample = &samples[index];
        let siblings: Vec<Reference> = references
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != index)
            .map(|(_, r)| r.clone())
            .collect();
        let mut attacks = generic_attacks(&sample.instance, &sample.reference, &siblings);
        attacks.extend(plugin.attacks(&sample.instance, &sample.reference));
        let outcomes = run_attacks(host, &sample.instance, &attacks)?;
        Ok((leaks(&sample.instance, &sample.reference), outcomes))
    };
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<Option<Result<Attacked, HostError>>> =
        (0..samples.len()).map(|_| None).collect();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let sender = sender.clone();
            let (next, attack_one) = (&next, &attack_one);
            scope.spawn(move || loop {
                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if index >= samples.len() {
                    break;
                }
                if sender.send((index, attack_one(index))).is_err() {
                    break;
                }
            });
        }
        drop(sender);
        for (index, result) in receiver {
            results[index] = Some(result);
        }
    });
    results
        .into_iter()
        .map(|r| r.unwrap_or_else(|| Ok((Vec::new(), Vec::new()))))
        .collect()
}

fn check_structure(
    gate: &mut GateBuilder,
    plugin: &dyn Plugin,
    sample: &Sample,
    environment: &str,
) {
    let (split, seed) = (sample.request.split, sample.request.seed);
    let meta = sample.instance.meta();
    match Instance::from_folder(sample.instance.folder().clone()) {
        Err(e) => gate.fail(split, seed, e),
        Ok(read) if read.meta() != meta => {
            gate.fail(split, seed, "task.toml does not read back as written")
        }
        Ok(_) => {}
    }
    if meta.split != split || meta.seed != seed {
        gate.fail(
            split,
            seed,
            "task.toml records another split or seed than was asked",
        );
    }
    if meta.environment != environment
        || meta.plugin != plugin.id()
        || meta.plugin_version != plugin.version()
    {
        gate.fail(split, seed, "task.toml names another environment or plugin");
    }
    if meta.verifier_class != plugin.verifier_class() {
        gate.fail(
            split,
            seed,
            "task.toml records another verifier class than the plugin's",
        );
    }
    if sample.instance.instruction().trim().is_empty() {
        gate.fail(split, seed, "the instruction is empty");
    }
    let view = sample.instance.solver_view();
    if view
        .paths()
        .any(|p| p.starts_with("tests/") || p.starts_with("solution/"))
    {
        gate.fail(split, seed, "the solver is shown hidden material");
    }
}

fn check_robustness(
    gate: &mut GateBuilder,
    plugin: &dyn Plugin,
    sample: &Sample,
    session: &Session,
) -> Result<(), HostError> {
    let (split, seed) = (sample.request.split, sample.request.seed);
    let first = session.verify()?;
    if session.verify()? != first {
        gate.fail(
            split,
            seed,
            "grading the same state twice gave different rewards",
        );
    }
    if session.solve()?.succeeded() && !session.verify()?.solved() {
        gate.fail(
            split,
            seed,
            "running the reference a second time on its own result breaks it",
        );
    }
    for (name, outcome) in plugin.extra_gates(&sample.instance, session) {
        if let Err(why) = outcome {
            gate.fail(split, seed, format!("{name}: {why}"));
        }
    }
    // A missing answer is graded, not a crash.
    let missing = sample
        .reference
        .changed
        .keys()
        .filter(|p| session.remove(p))
        .count();
    if missing > 0 {
        let verdict = session.verify()?;
        if !matches!(verdict, Verdict::Graded(_)) || verdict.solved() {
            gate.fail(
                split,
                seed,
                format!(
                    "with the answer removed the verifier said {}",
                    verdict_text(&verdict)
                ),
            );
        }
    }
    // Reset restores the unsolved state.
    if session.has_reset() {
        session.reset()?;
        if session.verify()?.solved() {
            gate.fail(split, seed, "after reset the workspace still grades solved");
        }
    }
    Ok(())
}

fn check_novelty(
    gate: &mut GateBuilder,
    plugin: &dyn Plugin,
    samples: &[Sample],
    known: &BTreeSet<Digest>,
) {
    let mut seen = Seen::new(3, 0.9);
    let mut versions: BTreeSet<Digest> = BTreeSet::new();
    let mut by_split: BTreeMap<Split, BTreeSet<Digest>> = BTreeMap::new();
    for sample in samples {
        let (split, seed) = (sample.request.split, sample.request.seed);
        let version = sample.instance.version();
        if known.contains(&version) {
            gate.fail(split, seed, "this instance was already trained on");
        }
        if !versions.insert(version.clone()) {
            gate.fail(split, seed, "the same folder as another sampled instance");
        }
        let text = plugin.novelty_text(&sample.instance);
        if let Some(repeat) = seen.repeats(&text) {
            gate.fail(
                split,
                seed,
                format!("repeats an earlier instance ({repeat:?})"),
            );
        }
        seen.admit(&text);
        by_split
            .entry(split)
            .or_default()
            .insert(digest_of_text(&text));
    }
    if let (Some(train), Some(dev)) = (by_split.get(&Split::Train), by_split.get(&Split::Dev)) {
        if train.intersection(dev).next().is_some() {
            gate.fail_general("a dev instance repeats a train instance");
        }
    }
}

fn digest_of_text(text: &str) -> Digest {
    Digest::of(text.as_bytes())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;

    #[test]
    fn train_and_dev_seeds_are_distinct_and_follow_from_the_admission_seed() {
        let (train, dev) = draw_seeds(9, 4); // seed: only distinctness and determinism are asserted
        assert_eq!(draw_seeds(9, 4), (train.clone(), dev.clone()));
        let all: BTreeSet<_> = train.iter().chain(&dev).collect();
        assert_eq!(all.len(), 8);
        assert_ne!(draw_seeds(10, 4).0, train);
    }
}
