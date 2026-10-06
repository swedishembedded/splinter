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
//! * A release stored as `splinter-release-v2`, before terms were recorded,
//!   still loads, with unknown terms and as restricted, and still verifies
//!   against its id.
#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::terms::{Distribution, Permission, Terms, UsagePolicy};
use splinter_core::training::{Regime, TrainingSummary};
use splinter_eval::gate::{Check, GateConfig, GateReport};
use splinter_orchestrator::releases::{ReleaseManifest, ReleaseStore, MANIFEST_V2, RELEASE_FORMAT};
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

/// A recorded candidate `name` and an adapter artifact for it; the manifest
/// of its release.
fn manifest(f: &Fixture, name: &str, terms: Terms, distribution: Distribution) -> ReleaseManifest {
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
            None,
        )
        .unwrap();
    let bytes = format!("adapter of {name}");
    let kept = f
        .artifacts
        .put_bytes(
            bytes.as_bytes(),
            &ArtifactSpec::new("adapter", "spec")
                .with_extension(".safetensors")
                .with_sha256(),
        )
        .unwrap();
    let unmeasured = "written by the release store spec";
    ReleaseManifest {
        format: RELEASE_FORMAT.into(),
        base_model: "Qwen/Qwen3-0.6B".into(),
        base_digest: Digest::sha256_of(b"base"),
        adapter_digest: Digest::sha256_of(bytes.as_bytes()),
        adapter_artifact: kept.digest,
        parent: None,
        candidate: name.into(),
        datasets: vec![DatasetId(dataset)],
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
        gate: GateReport::new(
            GateConfig::default(),
            Check::unmeasured(unmeasured),
            Check::unmeasured(unmeasured),
            Check::unmeasured(unmeasured),
            Check::unmeasured(unmeasured),
        ),
        terms,
        distribution,
        created_at: "2026-09-30T12:00:00.000Z".into(),
    }
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
        let unrestricted = manifest(
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
    let open = manifest(
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

#[test]
fn a_release_stored_before_terms_were_recorded_still_loads_as_restricted_and_unknown() {
    let f = fixture("v2");
    let current = manifest(&f, "old", Terms::unknown("x"), Distribution::Restricted);
    // What the previous release of this crate stored: the same manifest
    // without the two fields terms added, under the old format.
    let mut old = serde_json::to_value(&current).unwrap();
    let fields = old.as_object_mut().unwrap();
    fields.remove("terms");
    fields.remove("distribution");
    fields.insert("format".into(), json!(MANIFEST_V2));
    fields["training"].as_object_mut().unwrap().remove("terms");
    let id = f
        .workspace
        .record_release(&old, "release", "old", None)
        .unwrap();

    let loaded = f
        .store
        .get(&splinter_core::release::ReleaseId(id))
        .expect("an old manifest still verifies against its id and loads");
    assert_eq!(loaded.manifest.format, MANIFEST_V2);
    assert_eq!(loaded.manifest.candidate, "old");
    assert_eq!(loaded.manifest.distribution, Distribution::Restricted);
    assert_eq!(loaded.manifest.terms.training, Permission::Unknown);
    assert!(loaded
        .manifest
        .terms
        .permits_unrestricted_release()
        .is_err());
    assert_eq!(loaded.manifest.adapter_digest, current.adapter_digest);
}

#[test]
fn only_the_current_format_is_written() {
    let f = fixture("format");
    let mut stale = manifest(&f, "stale", Terms::unknown("x"), Distribution::Restricted);
    stale.format = MANIFEST_V2.into();
    assert!(matches!(
        f.store.put(&stale),
        Err(OrchestratorError::Refused(why)) if why.contains(RELEASE_FORMAT)
    ));
}
