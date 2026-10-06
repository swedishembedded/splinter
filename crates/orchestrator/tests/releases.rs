// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements immutable, content-addressed model
// releases with full lineage, for its clients. If your team needs
// expertise in model release management, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: the release store enforces what a release may be, whoever asks.
//!
//! * An unrestricted manifest is refused when its terms do not allow every
//!   use, whatever the pipeline decided; a restricted one is stored with the
//!   restriction recorded.
//! * A release stored as `splinter-release-v2` or `-v3`, before terms or
//!   artifact kinds were recorded, still loads, as an adapter release, with
//!   unknown terms and restricted where it never recorded any, and still
//!   verifies against its id.
//! * A full checkpoint is a release like an adapter: stored immutably under
//!   its digest, refused when the file is not the one the manifest names, and
//!   moved between by alias compare-and-set, with its history to roll back
//!   along.
#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_core::terms::{Distribution, Permission, Terms, UsagePolicy};
use splinter_core::training::{Regime, TrainingSummary};
use splinter_eval::gate::{Check, GateConfig, GateReport};
use splinter_eval::metric_gate::Evidence;
use splinter_orchestrator::releases::{
    ArtifactKind, Provenance, ReleaseGate, ReleaseManifest, ReleaseStore, ReleasedArtifact,
    MANIFEST_V2, MANIFEST_V3, RELEASE_FORMAT,
};
use splinter_orchestrator::OrchestratorError;
use splinter_store::artifacts::{ArtifactSpec, ArtifactStore};
use splinter_store::lineage::DatasetLineage;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

struct Fixture {
    workspace: Workspace,
    artifacts: ArtifactStore,
    store: ReleaseStore,
}

fn fixture(name: &str) -> Fixture {
    let root = StateRoot::new(
        std::env::temp_dir().join(format!("splinter-releases-{name}-{}", std::process::id())),
    );
    let _ = std::fs::remove_dir_all(root.path());
    let workspace = Workspace::at(&root);
    Fixture {
        artifacts: ArtifactStore::new(&workspace, &root),
        store: ReleaseStore::new(&workspace, &root),
        workspace,
    }
}

/// Records candidate `name`, trained on a dataset of its own and continuing
/// `parent`.
fn record(f: &Fixture, name: &str, parent: Option<&ReleaseId>) -> DatasetId {
    let dataset = Digest::of(format!("dataset of {name}").as_bytes());
    f.workspace
        .record_dataset(
            &dataset,
            &DatasetLineage {
                recipe: json!({ "view": "spec" }),
                records: 1,
            },
            &[],
        )
        .unwrap();
    f.workspace
        .record_candidate(
            &json!({ "candidate": name }),
            name,
            std::slice::from_ref(&dataset),
            "sft",
            parent.map(|p| &p.0),
        )
        .unwrap();
    DatasetId(dataset)
}

/// Keeps `bytes` as an artifact with its SHA-256; returns its address.
fn keep(f: &Fixture, bytes: &[u8]) -> Digest {
    f.artifacts
        .put_bytes(
            bytes,
            &ArtifactSpec::new("release", "spec")
                .with_extension(".bin")
                .with_sha256(),
        )
        .unwrap()
        .digest
}

fn unmeasured_gate() -> ReleaseGate {
    let why = "written by the release store spec";
    ReleaseGate::Llm {
        report: GateReport::new(
            GateConfig::default(),
            Check::unmeasured(why),
            Check::unmeasured(why),
            Check::unmeasured(why),
            Check::unmeasured(why),
        ),
    }
}

fn manifest_of(
    f: &Fixture,
    name: &str,
    parent: Option<&ReleaseId>,
    artifact: ReleasedArtifact,
    terms: Terms,
    distribution: Distribution,
) -> ReleaseManifest {
    ReleaseManifest {
        format: RELEASE_FORMAT.into(),
        artifact,
        parent: parent.cloned(),
        candidate: name.into(),
        datasets: vec![record(f, name, parent)],
        replay: None,
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
            terms: Some(terms.clone()),
        },
        gate: unmeasured_gate(),
        metrics: None,
        terms,
        distribution,
        provenance: Provenance::default(),
        created_at: "2026-09-30T12:00:00.000Z".into(),
    }
}

/// An adapter release of candidate `name`.
fn adapter(f: &Fixture, name: &str, terms: Terms, distribution: Distribution) -> ReleaseManifest {
    let bytes = format!("adapter of {name}");
    let artifact = ReleasedArtifact::Adapter {
        base_model: "Qwen/Qwen3-0.6B".into(),
        base_digest: Digest::sha256_of(b"base"),
        adapter_digest: Digest::sha256_of(bytes.as_bytes()),
        adapter_artifact: keep(f, bytes.as_bytes()),
    };
    manifest_of(f, name, None, artifact, terms, distribution)
}

/// A full-checkpoint release of candidate `name`, continuing `parent`.
fn checkpoint(f: &Fixture, name: &str, parent: Option<&ReleaseId>) -> ReleaseManifest {
    let bytes = format!("checkpoint of {name}");
    let artifact = ReleasedArtifact::FullCheckpoint {
        architecture: "timeline-v1".into(),
        checkpoint_digest: Digest::sha256_of(bytes.as_bytes()),
        checkpoint_artifact: keep(f, bytes.as_bytes()),
    };
    let terms = UsagePolicy::Redistributable.terms("open");
    manifest_of(f, name, parent, artifact, terms, Distribution::Unrestricted)
}

#[test]
fn the_store_refuses_an_unrestricted_release_the_terms_do_not_allow() {
    let f = fixture("refuse");
    for (name, label) in [
        ("research", UsagePolicy::ResearchOnly),
        ("noncommercial", UsagePolicy::Noncommercial),
        ("dua", UsagePolicy::RestrictedDua),
        ("unknown", UsagePolicy::Unknown),
    ] {
        let unrestricted = adapter(
            &f,
            name,
            label.terms(label.as_str()),
            Distribution::Unrestricted,
        );
        match f.store.put(&unrestricted) {
            Err(OrchestratorError::Refused(why)) => {
                assert!(why.contains("cannot be unrestricted"), "{label:?}: {why}");
            }
            other => panic!("{label:?} must be refused as unrestricted, got {other:?}"),
        }
        assert!(f.store.of_candidate(name).unwrap().is_none());

        let restricted = ReleaseManifest {
            distribution: Distribution::Restricted,
            ..unrestricted
        };
        let stored = f.store.put(&restricted).unwrap();
        assert_eq!(stored.manifest.distribution, Distribution::Restricted);
        assert_eq!(stored.manifest.terms, restricted.terms);
    }
    let open = adapter(
        &f,
        "open",
        UsagePolicy::Redistributable.terms("open"),
        Distribution::Unrestricted,
    );
    assert_eq!(
        f.store.put(&open).unwrap().manifest.distribution,
        Distribution::Unrestricted
    );
}

/// The flat shape of an adapter release in `format`, as the previous
/// versions of this crate stored it.
fn legacy(current: &ReleaseManifest, format: &str) -> serde_json::Value {
    let mut old = serde_json::to_value(current).unwrap();
    let fields = old.as_object_mut().unwrap();
    let ReleasedArtifact::Adapter { .. } = &current.artifact else {
        panic!("only adapters had a legacy shape");
    };
    let artifact = fields.remove("artifact").unwrap();
    for key in [
        "base_model",
        "base_digest",
        "adapter_digest",
        "adapter_artifact",
    ] {
        fields.insert(key.into(), artifact[key].clone());
    }
    let gate = fields.remove("gate").unwrap();
    fields.insert("gate".into(), gate["report"].clone());
    fields.remove("metrics");
    fields.remove("provenance");
    fields.insert("format".into(), json!(format));
    if format == MANIFEST_V2 {
        fields.remove("terms");
        fields.remove("distribution");
        fields["training"].as_object_mut().unwrap().remove("terms");
    }
    old
}

#[test]
fn releases_stored_before_terms_and_kinds_were_recorded_still_load() {
    let f = fixture("legacy");
    let current = adapter(
        &f,
        "old",
        UsagePolicy::ResearchOnly.terms("cohort"),
        Distribution::Restricted,
    );
    let ReleasedArtifact::Adapter { adapter_digest, .. } = &current.artifact else {
        unreachable!("built as an adapter");
    };

    // splinter-release-v2: no terms. Unknown, restricted.
    let id = f
        .workspace
        .record_release(&legacy(&current, MANIFEST_V2), "release", "old", None)
        .unwrap();
    let loaded = f
        .store
        .get(&ReleaseId(id))
        .expect("an old manifest still verifies against its id and loads");
    assert_eq!(loaded.manifest.format, MANIFEST_V2);
    assert_eq!(loaded.manifest.artifact.kind(), ArtifactKind::Adapter);
    assert_eq!(loaded.manifest.artifact.content_digest(), adapter_digest);
    assert_eq!(loaded.manifest.distribution, Distribution::Restricted);
    assert_eq!(loaded.manifest.terms.training, Permission::Unknown);
    assert!(loaded
        .manifest
        .terms
        .permits_unrestricted_release()
        .is_err());
    assert!(loaded.manifest.gate.llm().is_some());
    assert!(loaded.adapter().is_ok());

    // splinter-release-v3: terms, adapter only. Terms kept.
    let id = f
        .workspace
        .record_release(&legacy(&current, MANIFEST_V3), "release", "old", None)
        .unwrap();
    let loaded = f.store.get(&ReleaseId(id)).unwrap();
    assert_eq!(loaded.manifest.format, MANIFEST_V3);
    assert_eq!(loaded.manifest.terms, current.terms);
    assert_eq!(loaded.manifest.artifact, current.artifact);
}

#[test]
fn only_the_current_format_is_written() {
    let f = fixture("format");
    let mut stale = adapter(&f, "stale", Terms::unknown("x"), Distribution::Restricted);
    stale.format = MANIFEST_V3.into();
    assert!(matches!(
        f.store.put(&stale),
        Err(OrchestratorError::Refused(why)) if why.contains(RELEASE_FORMAT)
    ));
}

#[test]
fn a_full_checkpoint_is_stored_and_read_back_by_its_digest() {
    let f = fixture("checkpoint");
    let mut manifest = checkpoint(&f, "risk-1", None);
    manifest.provenance = Provenance {
        dataset_snapshots: vec![Digest::of(b"snapshot")],
        training_config: Some(Digest::of(b"config")),
        evaluation_splits: vec![Digest::of(b"split")],
        calibration: Some(Digest::of(b"calibration")),
        brain_commit: Some("3f019a23".into()),
        splinter_commit: Some("05f13d4".into()),
    };
    let mut evidence = Evidence::default();
    evidence.values.insert("slope".into(), 0.97);
    evidence
        .intervals
        .insert("ibs_diff".into(), (-0.004, -0.001));
    manifest.metrics = Some(evidence);

    let stored = f.store.put(&manifest).unwrap();
    assert_eq!(stored.manifest, manifest, "everything recorded reads back");
    assert_eq!(
        stored.manifest.artifact.kind(),
        ArtifactKind::FullCheckpoint
    );
    assert_eq!(
        std::fs::read(&stored.artifact).unwrap(),
        b"checkpoint of risk-1"
    );
    assert!(
        matches!(stored.adapter(), Err(OrchestratorError::Refused(why)) if why.contains("not an adapter")),
        "a checkpoint has no adapter to load on a base"
    );
    // Immutable: the same manifest is the same release, never a second one.
    assert_eq!(f.store.put(&manifest).unwrap().id, stored.id);
    assert_eq!(f.store.list().unwrap(), vec![stored.id.clone()]);
    assert_eq!(
        f.store.of_candidate("risk-1").unwrap().unwrap().id,
        stored.id
    );
}

#[test]
fn a_file_that_is_not_the_one_the_manifest_names_is_refused() {
    let f = fixture("mismatch");
    let mut wrong = checkpoint(&f, "risk-1", None);
    wrong.artifact = ReleasedArtifact::FullCheckpoint {
        architecture: "timeline-v1".into(),
        checkpoint_digest: Digest::sha256_of(b"some other checkpoint"),
        checkpoint_artifact: keep(&f, b"checkpoint of risk-1"),
    };
    match f.store.put(&wrong) {
        Err(OrchestratorError::Refused(why)) => {
            assert!(why.contains("SHA-256") && why.contains("risk-1"), "{why}");
        }
        other => panic!("a digest mismatch must be refused, got {other:?}"),
    }
    assert!(f.store.list().unwrap().is_empty());

    let mut nameless = checkpoint(&f, "risk-2", None);
    let ReleasedArtifact::FullCheckpoint {
        checkpoint_digest,
        checkpoint_artifact,
        ..
    } = nameless.artifact.clone()
    else {
        unreachable!("built as a checkpoint");
    };
    nameless.artifact = ReleasedArtifact::FullCheckpoint {
        architecture: " ".into(),
        checkpoint_digest,
        checkpoint_artifact,
    };
    assert!(matches!(
        f.store.put(&nameless),
        Err(OrchestratorError::Refused(why)) if why.contains("architecture")
    ));
}

#[test]
fn an_alias_moves_by_compare_and_set_between_checkpoints_and_rolls_back() {
    let f = fixture("alias");
    let at = "2026-09-30T12:00:00.000Z";
    let first = f.store.put(&checkpoint(&f, "risk-1", None)).unwrap().id;
    f.store.move_alias("risk", None, &first, at).unwrap();
    let second = f
        .store
        .put(&checkpoint(&f, "risk-2", Some(&first)))
        .unwrap()
        .id;
    assert_eq!(
        f.store.get(&second).unwrap().manifest.parent,
        Some(first.clone())
    );

    // A move decided against a champion that is not the alias's is refused.
    let stale = f.store.move_alias("risk", None, &second, at);
    assert!(matches!(stale, Err(OrchestratorError::Refused(why)) if why.contains("decide again")));
    assert_eq!(f.store.alias("risk").unwrap(), Some(first.clone()));

    f.store
        .move_alias("risk", Some(&first), &second, at)
        .unwrap();
    assert_eq!(f.store.alias("risk").unwrap(), Some(second.clone()));

    // Rolling back is moving to the release this one continued.
    let parent = f.store.get(&second).unwrap().manifest.parent.unwrap();
    f.store
        .move_alias("risk", Some(&second), &parent, at)
        .unwrap();
    assert_eq!(f.store.alias("risk").unwrap(), Some(first.clone()));
    let history: Vec<ReleaseId> = f
        .store
        .alias_history("risk")
        .unwrap()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(history, vec![first.clone(), second, first]);
    assert_eq!(f.store.lineage(&history[1]).unwrap().len(), 2);
}
