// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning campaigns whose every generation is
// held to one promotion gate, for its clients. If your team needs expertise
// in continual training of agent models, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Which release of a run `champion:RUN` names: a run of several
//! generations holds one release stage per generation (`release`,
//! `release-g2`, ...), and the champion it leaves in use is the one its
//! recipe's `outputs.release` names, else its only release, else the newest
//! promotion. A release is named explicitly as `champion:RUN#STAGE`; a run of
//! several releases that promoted nothing has no champion to name implicitly.

use std::path::Path;

use serde_json::json;
use splinter_core::digest::Digest;
use splinter_factory::champion;
use splinter_factory::recipe::ModelRef;
use splinter_store::factory::products::{ProductManifest, MANIFEST_SCHEMA, VIEW_MANIFEST};

/// A committed release view at `run/key`: promoted `candidate` (or kept
/// one), its adapter's bytes `adapter`, committed at `created_at`.
fn release(run: &Path, key: &str, promoted: Option<&str>, adapter: &[u8], created_at: &str) {
    let view = run.join(key);
    std::fs::create_dir_all(view.join("champion")).unwrap();
    std::fs::write(view.join("champion/adapter.safetensors"), adapter).unwrap();
    let digest = Digest::sha256_of_reader(adapter).unwrap();
    let record = json!({
        "schema": "splinter.release/1", "promote": true, "promoted": promoted,
        "candidates": [], "lineage": [],
        "champion": {
            "base": "brain:Qwen/Qwen3-4B", "adapter": "champion/adapter.safetensors",
            "adapter_digest": digest, "candidate": promoted.unwrap_or("lora-g0"),
        },
    });
    std::fs::write(view.join("release.json"), record.to_string()).unwrap();
    let manifest = ProductManifest {
        schema: MANIFEST_SCHEMA.into(),
        stage: Digest::of(key.as_bytes()),
        name: key.into(),
        files: Vec::new(),
        created_at: created_at.into(),
    };
    std::fs::write(
        view.join(VIEW_MANIFEST),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}

fn recipe(run: &Path, outputs: &str) {
    std::fs::write(
        run.join("recipe.yaml"),
        format!(
            "schema: splinter.recipe/v2\nname: loop\nstages:\n\
             - id: release\n  uses: promotion\n  inputs: {{ evaluations: [] }}\n  with: {{ promote: true }}\n\
             - id: release-g2\n  uses: promotion\n  inputs: {{ evaluations: [] }}\n  with: {{ promote: true }}\n\
             {outputs}"
        ),
    )
    .unwrap();
}

fn load(repository: &Path, reference: &str) -> Result<champion::Champion, String> {
    champion::load(repository, &reference.parse::<ModelRef>().unwrap())
}

#[test]
fn a_run_of_two_generations_names_the_newest_promoted_release_unless_a_stage_is_named() {
    let repository = tempfile::tempdir().unwrap();
    let run = repository.path().join("runs/loop");
    std::fs::create_dir_all(&run).unwrap();
    recipe(&run, "");
    release(
        &run,
        "release",
        Some("lora-g1"),
        b"g1",
        "2026-11-26T10:00:00Z",
    );
    release(
        &run,
        "release-g2",
        Some("lora-g2"),
        b"g2",
        "2026-11-27T10:00:00Z",
    );

    let newest = load(repository.path(), "champion:runs/loop").unwrap();
    assert_eq!(newest.record.candidate.as_deref(), Some("lora-g2"));
    assert_eq!(
        std::fs::read(newest.adapter.unwrap()).unwrap(),
        b"g2",
        "generation 2's adapter"
    );

    let first = load(repository.path(), "champion:runs/loop#release").unwrap();
    assert_eq!(first.record.candidate.as_deref(), Some("lora-g1"));
    assert_ne!(
        first.digest,
        load(repository.path(), "champion:runs/loop")
            .unwrap()
            .digest
    );

    // The recipe's declared release wins over the newest one.
    recipe(&run, "outputs:\n  release: $stages.release.release\n");
    let declared = load(repository.path(), "champion:runs/loop").unwrap();
    assert_eq!(declared.record.candidate.as_deref(), Some("lora-g1"));

    let missing = load(repository.path(), "champion:runs/loop#release-g9").unwrap_err();
    assert!(missing.contains("release-g9"), "{missing}");
}

#[test]
fn a_run_of_releases_that_promoted_nothing_names_no_champion_unless_one_is_named() {
    let repository = tempfile::tempdir().unwrap();
    let run = repository.path().join("runs/loop");
    std::fs::create_dir_all(&run).unwrap();
    recipe(&run, "");
    release(&run, "release", None, b"kept", "2026-11-26T10:00:00Z");
    release(&run, "release-g2", None, b"kept", "2026-11-27T10:00:00Z");

    let refused = load(repository.path(), "champion:runs/loop").unwrap_err();
    assert!(
        refused.contains("promoted") && refused.contains("#"),
        "the refusal says why and how to name a release: {refused}"
    );
    // Named, a release's kept champion is what it leaves in use.
    let kept = load(repository.path(), "champion:runs/loop#release-g2").unwrap();
    assert_eq!(kept.record.candidate.as_deref(), Some("lora-g0"));

    // A run of one release leaves what that release kept.
    std::fs::remove_dir_all(run.join("release-g2")).unwrap();
    let only = load(repository.path(), "champion:runs/loop").unwrap();
    assert_eq!(only.release, run.join("release"));
}

#[test]
fn a_run_with_one_release_stage_keeps_the_champion_digest_it_had() {
    let repository = tempfile::tempdir().unwrap();
    let run = repository.path().join("runs/g1");
    std::fs::create_dir_all(&run).unwrap();
    recipe(&run, "");
    release(
        &run,
        "release",
        Some("lora-g1"),
        b"g1",
        "2026-11-26T10:00:00Z",
    );
    let named = load(repository.path(), "champion:runs/g1").unwrap();
    // The digest is the release record's bytes and the adapter's digest, as
    // before a release could be chosen: a run that pinned it still resumes.
    let mut expected = std::fs::read(run.join("release/release.json")).unwrap();
    expected.extend_from_slice(
        Digest::sha256_of_reader(&b"g1"[..])
            .unwrap()
            .as_str()
            .as_bytes(),
    );
    assert_eq!(named.digest, Digest::of(&expected));
}
