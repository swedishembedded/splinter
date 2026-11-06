// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that reach their model
// only through a serving API, for its clients. If your team needs expertise
// in routing agent traffic across inference servers, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a local model reference is requested from a brain endpoint by the
//! id the endpoint serves it under. A base is served as `<org>/<model>`, an
//! adapter is found by the digest of its file, a reference the endpoint does
//! not list is refused by name, and a conversation is pinned to one endpoint
//! by a stable hash of its id.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use futures::StreamExt;
use splinter_model::endpoint::BrainEndpoints;
use splinter_model::local::{Sampling, AGENT_SAMPLING};
use splinter_model::scripted_endpoint::{Card, ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::{LocalWeights, ModelSelection, PolicyError};
use sven_sdk::model::{CompletionRequest, Message, ResponseEvent};

fn weights(base: &str, adapter: Option<&Path>) -> LocalWeights {
    LocalWeights {
        base: PathBuf::from(base),
        adapter: adapter.map(Path::to_path_buf),
        context_tokens: None,
    }
}

fn endpoints(urls: &[String]) -> BrainEndpoints {
    BrainEndpoints::new(urls.to_vec(), SCRIPTED_KEY).unwrap()
}

/// The digest brain reports for the file at `path`.
fn digest_of(path: &Path) -> String {
    splinter_core::digest::Digest::sha256_of_reader(std::fs::File::open(path).unwrap())
        .unwrap()
        .to_string()
}

fn adapter_file(dir: &Path, content: &[u8]) -> PathBuf {
    let path = dir.join("adapter.brain.safetensors");
    std::fs::write(&path, content).unwrap();
    path
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(future)
}

/// Everything a model streams back for `prompt`, as text.
fn ask(loaded: &splinter_model::LoadedModel, prompt: &str) -> String {
    let provider = loaded.provider();
    let request = CompletionRequest {
        messages: vec![Message::user(prompt)],
        stream: true,
        ..CompletionRequest::default()
    };
    let mut text = String::new();
    block_on(async {
        let mut stream = provider.complete(request).await.unwrap();
        while let Some(event) = stream.next().await {
            if let ResponseEvent::TextDelta(delta) = event.unwrap() {
                text.push_str(&delta);
            }
        }
    });
    text
}

#[test]
fn a_base_is_requested_by_its_org_and_model_name_and_answers_through_the_endpoint() {
    let server = ScriptedEndpoint::start(vec![Card::base("Qwen/Qwen3.8-27B-FP8")])
        .with_reply(|seen| format!("served by {}", seen.model));
    let brain = endpoints(&[server.url()]);
    let selection = ModelSelection::Local(weights("/models/Qwen/Qwen3.8-27B-FP8", None));

    let loaded = selection.load_served(&brain).unwrap();

    assert_eq!(
        loaded.identity(),
        "brain/Qwen/Qwen3.8-27B-FP8",
        "records name the model as an in-process run does"
    );
    assert_eq!(ask(&loaded, "hello"), "served by Qwen/Qwen3.8-27B-FP8");
    assert_eq!(server.seen()[0].bearer, SCRIPTED_KEY);
}

#[test]
fn a_model_the_endpoint_does_not_list_is_refused_by_name_with_what_it_serves() {
    let server = ScriptedEndpoint::start(vec![Card::base("Qwen/Qwen3-0.6B")]);
    let brain = endpoints(&[server.url()]);
    let selection = ModelSelection::Local(weights("/models/Qwen/Qwen3.8-27B-FP8", None));

    let refused = selection.load_served(&brain).unwrap_err();

    let text = refused.to_string();
    assert!(matches!(refused, PolicyError::NotServed { .. }), "{text}");
    assert!(
        text.contains("Qwen/Qwen3.8-27B-FP8"),
        "names the model: {text}"
    );
    assert!(text.contains(&server.url()), "names the endpoint: {text}");
    assert!(
        text.contains("Qwen/Qwen3-0.6B"),
        "lists what it serves: {text}"
    );
    assert!(server.seen().is_empty(), "nothing was asked of the model");
}

#[test]
fn an_adapter_is_found_by_the_digest_of_its_file_whatever_id_it_is_served_under() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = adapter_file(dir.path(), b"candidate-1");
    let server = ScriptedEndpoint::start(vec![
        Card::base("Qwen/Qwen3-0.6B"),
        Card::adapter(
            "Qwen/Qwen3-0.6B:splinter:candidate:v1",
            &digest_of(&adapter),
        ),
    ])
    .with_reply(|seen| seen.model.clone());
    let brain = endpoints(&[server.url()]);
    let selection = ModelSelection::Local(weights("/models/Qwen/Qwen3-0.6B", Some(&adapter)));

    let loaded = selection.load_served(&brain).unwrap();

    assert_eq!(loaded.identity(), "brain/Qwen/Qwen3-0.6B+adapter.brain");
    assert_eq!(
        ask(&loaded, "hi"),
        "Qwen/Qwen3-0.6B:splinter:candidate:v1",
        "the adapter's own card answers, not the base"
    );
}

#[test]
fn an_adapter_the_endpoint_does_not_serve_is_refused_by_its_digest_and_how_to_serve_it() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = adapter_file(dir.path(), b"candidate-2");
    let server = ScriptedEndpoint::start(vec![Card::base("Qwen/Qwen3-0.6B")]);
    let brain = endpoints(&[server.url()]);
    let selection = ModelSelection::Local(weights("/models/Qwen/Qwen3-0.6B", Some(&adapter)));

    let text = selection.load_served(&brain).unwrap_err().to_string();

    assert!(
        text.contains(&digest_of(&adapter)),
        "names the digest: {text}"
    );
    assert!(text.contains("--adapter"), "says how to serve it: {text}");
    assert!(
        text.contains("--brain-adapter-manifest"),
        "says how splinter can offer it: {text}"
    );
}

#[test]
fn an_adapter_is_offered_through_the_manifest_the_endpoint_follows_and_waited_for() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = adapter_file(dir.path(), b"candidate-3");
    let manifest = dir.path().join("release.json");
    let server = ScriptedEndpoint::start(vec![Card::base("Qwen/Qwen3-0.6B")])
        .following(manifest.clone(), "Qwen/Qwen3-0.6B:splinter:candidate:v3");
    let brain = endpoints(&[server.url()]).with_adapter_manifest(manifest.clone());
    let selection = ModelSelection::Local(weights("/models/Qwen/Qwen3-0.6B", Some(&adapter)));

    let loaded = selection.load_served(&brain).unwrap();

    let published: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    assert_eq!(published["digest"], digest_of(&adapter));
    assert_eq!(
        ask(&loaded, "hi"),
        "an answer",
        "the offered adapter answers"
    );
    assert_eq!(
        server.seen()[0].model,
        "Qwen/Qwen3-0.6B:splinter:candidate:v3"
    );
}

#[test]
fn sampling_rides_every_request_and_a_seed_pins_the_attempt_to_one_endpoint() {
    let first = ScriptedEndpoint::start(vec![Card::base("Qwen/Qwen3-0.6B")]);
    let second = ScriptedEndpoint::start(vec![Card::base("Qwen/Qwen3-0.6B")]);
    let brain = endpoints(&[first.url(), second.url()]);
    let selection = ModelSelection::Local(weights("/models/Qwen/Qwen3-0.6B", None));
    let loaded = selection.load_served(&brain).unwrap();

    let sampling = |seed| Sampling {
        seed: Some(seed),
        temperature: 0.7,
        top_k: 20,
        ..AGENT_SAMPLING
    };
    let mut asked_of = Vec::new();
    for seed in 0..40u64 {
        let provider = loaded.resampled(sampling(seed)).unwrap().unwrap();
        let request = || CompletionRequest {
            messages: vec![Message::user("q")],
            stream: true,
            ..CompletionRequest::default()
        };
        // Two requests of one attempt: the same endpoint answers both.
        for _ in 0..2 {
            block_on(async {
                let mut stream = provider.complete(request()).await.unwrap();
                while stream.next().await.is_some() {}
            });
        }
        asked_of.push(seed);
    }

    let (a, b) = (first.seen(), second.seen());
    assert_eq!(a.len() + b.len(), 80);
    assert!(
        !a.is_empty() && !b.is_empty(),
        "attempts spread over both endpoints: {} and {}",
        a.len(),
        b.len()
    );
    for seed in asked_of {
        let on_first = a.iter().filter(|s| s.seed == Some(seed)).count();
        let on_second = b.iter().filter(|s| s.seed == Some(seed)).count();
        assert!(
            (on_first, on_second) == (2, 0) || (on_first, on_second) == (0, 2),
            "seed {seed} was split across endpoints: {on_first}/{on_second}"
        );
    }
    let any = a.first().unwrap();
    assert!(
        (any.temperature.unwrap() - 0.7).abs() < 1e-6,
        "{:?}",
        any.temperature
    );
    assert_eq!(any.top_k, Some(20));
}

#[test]
fn the_endpoint_for_an_id_is_the_same_every_time_and_every_endpoint_gets_some() {
    let brain = BrainEndpoints::new(
        vec![
            "http://127.0.0.1:1/v1".into(),
            "http://127.0.0.1:2/v1".into(),
            "http://127.0.0.1:3/v1".into(),
        ],
        "k",
    )
    .unwrap();
    // A fixed id must keep landing on its endpoint across runs and releases,
    // or a conversation's prefix is cold on the endpoint it moved to.
    assert_eq!(
        brain.pin(b"task-7/attempt-0"),
        brain.pin(b"task-7/attempt-0")
    );
    let hit: std::collections::BTreeSet<usize> = (0..200)
        .map(|n| brain.pin(format!("attempt-{n}").as_bytes()))
        .collect();
    assert_eq!(hit.len(), 3);
    assert!(hit.iter().all(|index| *index < 3));
}

#[test]
fn the_key_is_the_one_word_of_its_file_and_an_empty_file_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("key");
    std::fs::write(&file, "sk-brain-abc\n").unwrap();
    assert_eq!(BrainEndpoints::read_key(&file).unwrap(), "sk-brain-abc");
    std::fs::write(&file, "  \n").unwrap();
    let text = BrainEndpoints::read_key(&file).unwrap_err().to_string();
    assert!(text.contains(&file.display().to_string()), "{text}");
    assert!(BrainEndpoints::read_key(&dir.path().join("absent")).is_err());
}

#[test]
fn an_endpoint_url_must_be_an_http_base_url_and_at_least_one_is_needed() {
    assert!(BrainEndpoints::new(Vec::new(), "k").is_err());
    let odd = BrainEndpoints::new(vec!["ftp://brain.example/v1".into()], "k")
        .unwrap_err()
        .to_string();
    assert!(odd.contains("ftp://brain.example/v1"), "{odd}");
    assert!(BrainEndpoints::new(vec!["https://brain.example/v1".into()], "k").is_ok());
    let trimmed = BrainEndpoints::new(vec!["http://127.0.0.1:8805/v1/".into()], "k").unwrap();
    assert_eq!(trimmed.urls(), ["http://127.0.0.1:8805/v1"]);
}

#[test]
fn the_width_asked_of_endpoints_is_the_configured_one_else_a_stated_default_per_endpoint() {
    let one = BrainEndpoints::new(vec!["http://127.0.0.1:1/v1".into()], "k").unwrap();
    let two = BrainEndpoints::new(
        vec![
            "http://127.0.0.1:1/v1".into(),
            "http://127.0.0.1:2/v1".into(),
        ],
        "k",
    )
    .unwrap();
    assert_eq!(
        one.attempt_concurrency(),
        splinter_model::endpoint::DEFAULT_ATTEMPTS_PER_ENDPOINT
    );
    assert_eq!(
        two.attempt_concurrency(),
        2 * splinter_model::endpoint::DEFAULT_ATTEMPTS_PER_ENDPOINT
    );
    assert_eq!(one.with_attempt_concurrency(48).attempt_concurrency(), 48);
}
