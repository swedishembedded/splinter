// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements lossless trajectory intake that keeps what
// an agent recorded and refuses what it cannot represent, for its clients. If
// your team needs expertise in training-data provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: importing an ATIF session loses nothing silently. A structural key
//! the importer does not know is refused by its JSON path; the digest of the
//! source bytes and the number of image segments left out are recorded in the
//! origin; and namespaced `extra` records a producer attached (`sven.gate`,
//! `sven.memory`) are kept verbatim and addressed by their digest.

use serde_json::{json, Value};
use splinter_core::clock::FixedClock;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::source::Origin;
use splinter_knowledge::capture::{capture_session, SessionRefusal, PASS_THROUGH_RECORDS};

fn clock() -> FixedClock {
    FixedClock::new("2026-10-15T08:00:00.000Z")
}

fn session() -> Value {
    json!({
        "schema_version": "ATIF-v1.7",
        "session_id": "s1",
        "agent": {"name": "scripted", "version": "1"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "Which port does the dashboard use?"},
            {"step_id": 2, "source": "agent", "message": "Port 8080."}
        ]
    })
}

fn raw(value: &Value) -> anyhow::Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}

fn session_origin(value: &Value) -> anyhow::Result<Origin> {
    Ok(capture_session(&raw(value)?, &clock())?
        .source()
        .origin
        .clone())
}

#[test]
fn an_unknown_step_key_is_refused_by_name() -> anyhow::Result<()> {
    let mut value = session();
    value["steps"][1]["mystery"] = json!(true);
    let refused = capture_session(&raw(&value)?, &clock()).err();
    assert!(
        matches!(&refused, Some(SessionRefusal::UnknownKey { path }) if path == "steps[1].mystery"),
        "{refused:?}"
    );

    let mut nested = session();
    nested["steps"][1]["tool_calls"] =
        json!([{"tool_call_id": "c1", "function_name": "shell", "arguments": {}, "stray": 1}]);
    let refused = capture_session(&raw(&nested)?, &clock()).err();
    assert!(
        matches!(&refused, Some(SessionRefusal::UnknownKey { path }) if path == "steps[1].tool_calls[0].stray"),
        "{refused:?}"
    );

    // `extra` is the format's own room for producer metadata: anything goes.
    let mut free = session();
    free["extra"] = json!({"anything": {"goes": 1}});
    free["steps"][1]["extra"] = json!({"anything": 2});
    capture_session(&raw(&free)?, &clock())?;
    Ok(())
}

#[test]
fn the_source_bytes_digest_is_recorded() -> anyhow::Result<()> {
    let bytes = raw(&session())?;
    let captured = capture_session(&bytes, &clock())?;
    let Origin::Session { source_sha256, .. } = &captured.source().origin else {
        anyhow::bail!("not a session origin");
    };
    assert_eq!(source_sha256.as_ref(), Some(&Digest::sha256_of(&bytes)));

    let reformatted = serde_json::to_vec_pretty(&session())?;
    assert_ne!(reformatted, bytes);
    let other = capture_session(&reformatted, &clock())?;
    assert_ne!(
        other.source().id,
        captured.source().id,
        "other bytes are other evidence, even when they parse alike"
    );
    Ok(())
}

#[test]
fn gate_and_memory_records_survive_import() -> anyhow::Result<()> {
    let gate = json!({
        "schema": "sven.gate.v1",
        "decisions": [{
            "decision_id": "d-1", "call_id": "c1", "capability": "ExecuteShell",
            "policy": "requires_approval", "decision": "approved", "decided_by": "human"
        }]
    });
    let memory = json!({
        "schema": "sven.memory.v1",
        "retrievals": [{
            "call_id": "c1", "query": "dashboard port", "space_ids": ["space-1"],
            "generation_id": "gen-a",
            "results": [{
                "unit_id": "unit-7", "revision": "sha256:abc",
                "anchor": {"heading_path": ["Ports"], "block_ordinal": 2, "text_fingerprint": "f00d"},
                "score": 0.83, "rank": 1, "inserted": true
            }]
        }]
    });
    let mut value = session();
    value["steps"][1]["tool_calls"] =
        json!([{"tool_call_id": "c1", "function_name": "shell", "arguments": {}}]);
    value["steps"][1]["extra"] = json!({
        "sven": {"gate": gate, "memory": memory, "other": {"kept": "in the trajectory only"}}
    });

    let captured = capture_session(&raw(&value)?, &clock())?;
    let Origin::Session { extra_records, .. } = &captured.source().origin else {
        anyhow::bail!("not a session origin");
    };
    let found: Vec<(Option<u64>, &str)> = extra_records
        .iter()
        .map(|r| (r.step_id, r.path.as_str()))
        .collect();
    assert_eq!(found, [(Some(2), "sven.gate"), (Some(2), "sven.memory")]);
    for (record, expected) in extra_records.iter().zip([&gate, &memory]) {
        assert_eq!(&record.record, expected, "kept verbatim");
        assert_eq!(record.digest, Digest::of(&canonical_json(expected)?));
    }
    assert_eq!(PASS_THROUGH_RECORDS, ["sven.gate", "sven.memory"]);

    // The same trajectory without records has no records and an origin that
    // names none.
    let Origin::Session { extra_records, .. } = session_origin(&session())? else {
        anyhow::bail!("not a session origin");
    };
    assert!(extra_records.is_empty());
    Ok(())
}

#[test]
fn dropped_image_segments_are_counted() -> anyhow::Result<()> {
    let mut value = session();
    value["steps"][0]["message"] = json!([
        {"type": "text", "text": "Which port does this dashboard use?"},
        {"type": "image", "source": {"media_type": "image/png", "path": "shot.png"}}
    ]);
    let Origin::Session { dropped_images, .. } = session_origin(&value)? else {
        anyhow::bail!("not a session origin");
    };
    assert_eq!(dropped_images, 1);
    let Origin::Session { dropped_images, .. } = session_origin(&session())? else {
        anyhow::bail!("not a session origin");
    };
    assert_eq!(dropped_images, 0);
    Ok(())
}
