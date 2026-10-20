// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning that keeps accumulating
// capabilities generation after generation, for its clients. If your team
// needs expertise in continual learning that consolidates into the weights,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a release belongs to a generation, and the base a policy runs on
//! follows its release.
//!
//! * A release records the generation it belongs to; one recorded before
//!   generations existed belongs to generation 0, on the configured base.
//! * A consolidation (a full checkpoint with a generation mark) is the base
//!   of the generation it opens: the policy pinned to it runs the checkpoint
//!   with no adapter, and an adapter released in that generation is pinned
//!   to the same base. The base is a directory laid out beside the
//!   configured base's tokenizer files, with the checkpoint's weights linked
//!   in, so brain loads it as it loads any base.
//! * A full checkpoint with no generation mark is not a policy.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::json;
use splinter_core::claim::ClaimId;
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_core::terms::{Distribution, Terms};
use splinter_core::training::{Regime, TrainingSummary};
use splinter_eval::gate::{Check, GateConfig, GateReport};
use splinter_orchestrator::generation::{Consolidation, ConsolidationMode, GenerationMark};
use splinter_orchestrator::releases::{
    Provenance, ReleaseGate, ReleaseManifest, ReleasedArtifact, RELEASE_FORMAT,
};
use splinter_orchestrator::{Config, Context};
use splinter_store::artifacts::ArtifactSpec;
use splinter_store::lineage::DatasetLineage;
use splinter_store::StateRoot;

struct Scratch(PathBuf);

impl Scratch {
    fn new(test: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "splinter-generations-{test}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn context(scratch: &Scratch) -> Context {
    let base = scratch.0.join("models/base");
    std::fs::create_dir_all(&base).unwrap();
    std::fs::write(base.join("model.safetensors"), b"root weights").unwrap();
    std::fs::write(
        base.join("config.json"),
        br#"{"max_position_embeddings": 4096}"#,
    )
    .unwrap();
    std::fs::write(base.join("tokenizer.json"), b"{}").unwrap();
    let config = Config {
        state_root: StateRoot::new(scratch.0.join("state")),
        model_store: scratch.0.join("models"),
        policy_base: base,
        policy_context_tokens: None,
        openrouter_api_key: None,
        brain_api_key: None,
        allow_remote: false,
        command_env: BTreeMap::new(),
        working_dir: scratch.0.clone(),
        brain_binary: None,
        front_door_model: None,
        assistant_model: None,
        judge_model: None,
        bf16_base: false,
        default_budget: None,
        remote_concurrency: 4,
        solve_deadline: splinter_orchestrator::config::DEFAULT_SOLVE_DEADLINE,
        min_calibration_controls: 2,
        workspace: Default::default(),
        thinking: false,
        served_temperature: None,
        served_top_p: None,
        served_top_k: None,
        served_seed: None,
        build: Default::default(),
    };
    Context::new(config, false).unwrap()
}

fn keep(ctx: &Context, bytes: &[u8]) -> Digest {
    let path = ctx.root().work().join(format!("keep-{}", bytes.len()));
    std::fs::create_dir_all(ctx.root().work()).unwrap();
    std::fs::write(&path, bytes).unwrap();
    ctx.artifacts()
        .put_file(
            &path,
            &ArtifactSpec::new("spec", "generations")
                .with_extension(".safetensors")
                .with_sha256(),
        )
        .unwrap()
        .digest
}

/// Records candidate `name`, continuing `parent`: what a release needs
/// before it is stored.
fn record(ctx: &Context, name: &str, parent: Option<&ReleaseId>) {
    let dataset = Digest::of(format!("dataset of {name}").as_bytes());
    ctx.workspace()
        .record_dataset(
            &dataset,
            &DatasetLineage {
                recipe: json!({ "view": "spec" }),
                records: 1,
            },
            &[],
        )
        .unwrap();
    ctx.workspace()
        .record_external_candidate(
            name,
            std::slice::from_ref(&dataset),
            "sft",
            parent.map(|p| &p.0),
        )
        .unwrap();
}

fn manifest(
    ctx: &Context,
    name: &str,
    artifact: ReleasedArtifact,
    parent: Option<&ReleaseId>,
    generation: Option<GenerationMark>,
) -> ReleaseManifest {
    let why = "written by the generations spec";
    record(ctx, name, parent);
    ReleaseManifest {
        format: RELEASE_FORMAT.into(),
        artifact,
        parent: parent.cloned(),
        candidate: name.into(),
        datasets: Vec::<DatasetId>::new(),
        replay: None,
        rehearsal: None,
        training: TrainingSummary {
            from: "local:base".into(),
            steps: 1,
            rank: 4,
            records: 1,
            regime: Regime::Sft,
            base_score: None,
            tuned_score: None,
            preference: None,
            curve: None,
            record: json!({ "trainer": "spec" }),
            terms: None,
        },
        gate: ReleaseGate::Llm {
            report: Box::new(GateReport::new(
                GateConfig::default(),
                Check::unmeasured(why),
                Check::unmeasured(why),
                Check::unmeasured(why),
                Check::unmeasured(why),
            )),
        },
        metrics: None,
        terms: Terms::unknown("spec"),
        distribution: Distribution::Restricted,
        provenance: Provenance::default(),
        generation,
        created_at: ctx.clock().utc_now(),
    }
}

fn adapter(
    ctx: &Context,
    name: &str,
    parent: Option<&ReleaseId>,
    generation: Option<GenerationMark>,
) -> ReleaseId {
    let bytes = format!("adapter {name}");
    let artifact = ReleasedArtifact::Adapter {
        base_model: "base".into(),
        base_digest: Digest::sha256_of(b"base"),
        adapter_digest: Digest::sha256_of(bytes.as_bytes()),
        adapter_artifact: keep(ctx, bytes.as_bytes()),
    };
    let m = manifest(ctx, name, artifact, parent, generation);
    ctx.releases().put(&m).unwrap().id
}

fn consolidation(ctx: &Context, name: &str, parent: &ReleaseId, mark: GenerationMark) -> ReleaseId {
    let bytes = format!("consolidated {name}");
    let artifact = ReleasedArtifact::FullCheckpoint {
        architecture: "qwen3".into(),
        checkpoint_digest: Digest::sha256_of(bytes.as_bytes()),
        checkpoint_artifact: keep(ctx, bytes.as_bytes()),
    };
    let m = manifest(ctx, name, artifact, Some(parent), Some(mark));
    ctx.releases().put(&m).unwrap().id
}

fn claim(n: u8) -> ClaimId {
    ClaimId(Digest::of(&[n]))
}

#[test]
fn a_release_without_a_generation_mark_is_generation_zero_on_the_configured_base() {
    let scratch = Scratch::new("zero");
    let ctx = context(&scratch);
    let id = adapter(&ctx, "first", None, None);
    ctx.releases()
        .move_alias("default", None, &id, "2026-10-01T00:00:00.000Z")
        .unwrap();

    let stored = ctx.releases().get(&id).unwrap();
    assert_eq!(stored.epoch(), 0);
    assert_eq!(stored.base_release(), None);
    let pin = ctx.policy_pin("default").unwrap().unwrap();
    assert_eq!((pin.generation, pin.base), (0, None));
    assert!(pin.adapter.is_some());
    assert_eq!(
        ctx.policy_base("default").unwrap(),
        ctx.config().policy_base,
        "generation 0 runs the configured base"
    );
}

#[test]
fn a_consolidation_is_the_base_of_the_generation_it_opens() {
    let scratch = Scratch::new("opens");
    let ctx = context(&scratch);
    let first = adapter(&ctx, "first", None, None);
    let closing = vec![claim(1), claim(2)];
    let opened = consolidation(
        &ctx,
        "consolidated",
        &first,
        GenerationMark {
            epoch: 1,
            base: None,
            consolidation: Some(Consolidation {
                mode: ConsolidationMode::Full,
                from: first.clone(),
                closing: closing.clone(),
                retained: Vec::new(),
                rebuilt: false,
            }),
        },
    );
    ctx.releases()
        .move_alias("default", None, &opened, "2026-10-01T00:00:00.000Z")
        .unwrap();

    let stored = ctx.releases().get(&opened).unwrap();
    assert_eq!(stored.epoch(), 1);
    assert_eq!(
        stored.base_release(),
        Some(opened.clone()),
        "a consolidation is its own generation's base"
    );
    let consolidation = stored
        .manifest
        .generation
        .as_ref()
        .and_then(|g| g.consolidation.as_ref())
        .unwrap();
    assert_eq!(consolidation.closing, closing);

    // The policy pinned to it runs the checkpoint, with no adapter, from a
    // directory brain loads like any base: the weights linked in, the
    // configured base's config and tokenizer beside them.
    let pin = ctx.policy_pin("default").unwrap().unwrap();
    assert_eq!(pin.generation, 1);
    assert_eq!(pin.adapter, None);
    let dir = pin.base.clone().expect("a generation 1 base");
    assert_eq!(
        std::fs::read(dir.join("model.safetensors")).unwrap(),
        b"consolidated consolidated"
    );
    assert_eq!(
        std::fs::read(dir.join("config.json")).unwrap(),
        std::fs::read(ctx.config().policy_base.join("config.json")).unwrap()
    );
    assert!(dir.join("tokenizer.json").is_file());
    assert_eq!(ctx.policy_base("default").unwrap(), dir);
    assert_eq!(
        ctx.generation_base(&opened).unwrap(),
        dir,
        "laying the base out again finds it"
    );
}

#[test]
fn an_adapter_released_in_a_generation_sits_on_that_generations_base() {
    let scratch = Scratch::new("adapter");
    let ctx = context(&scratch);
    let first = adapter(&ctx, "first", None, None);
    let opened = consolidation(
        &ctx,
        "consolidated",
        &first,
        GenerationMark {
            epoch: 1,
            base: None,
            consolidation: None,
        },
    );
    let second = adapter(
        &ctx,
        "second",
        Some(&opened),
        Some(GenerationMark {
            epoch: 1,
            base: Some(opened.clone()),
            consolidation: None,
        }),
    );
    ctx.releases()
        .move_alias("default", None, &second, "2026-10-01T00:00:00.000Z")
        .unwrap();

    let stored = ctx.releases().get(&second).unwrap();
    assert_eq!(
        (stored.epoch(), stored.base_release()),
        (1, Some(opened.clone()))
    );
    let pin = ctx.policy_pin("default").unwrap().unwrap();
    assert_eq!(pin.generation, 1);
    assert!(pin.adapter.is_some());
    assert_eq!(pin.base, Some(ctx.generation_base(&opened).unwrap()));
}

#[test]
fn a_full_checkpoint_with_no_generation_is_not_a_policy() {
    let scratch = Scratch::new("head");
    let ctx = context(&scratch);
    let bytes = b"a timeline head";
    let artifact = ReleasedArtifact::FullCheckpoint {
        architecture: "timeline-v1".into(),
        checkpoint_digest: Digest::sha256_of(bytes),
        checkpoint_artifact: keep(&ctx, bytes),
    };
    let m = manifest(&ctx, "head", artifact, None, None);
    let id = ctx.releases().put(&m).unwrap().id;
    ctx.releases()
        .move_alias("default", None, &id, "2026-10-01T00:00:00.000Z")
        .unwrap();
    let refused = ctx.policy_pin("default").unwrap_err().to_string();
    assert!(refused.contains("not an adapter"), "{refused}");
}
