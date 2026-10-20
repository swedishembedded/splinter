// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Spec: the agent a sample measures is offered the tools the task needs.
//!
//! An embedded sven engine offers no built-in tools unless the application
//! asks for them, and under its default deferred disclosure a model whose
//! first request carries only the `tools` tool cannot even look at the
//! workspace. Every episode of such an arm fails the same way for a reason
//! that has nothing to do with the model, and the score reads as a model that
//! cannot use tools. So the first request of an episode is checked at the wire
//! - through the same recorder the measured arms use - against a stub server.

// A test file: its fixture helpers panic on a broken fixture, which is the
// failure report.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::hash::BuildHasher;
use std::time::Duration;

use splinter_tool_syntax::{run_agent, Bounds, Endpoint, Recorder, ServedModel};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// A server that answers every request with a one-word streamed completion.
async fn stub_model() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address").to_string();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                // Drain the request; the body is not interpreted.
                let mut sink = vec![0u8; 1 << 16];
                let _ = tokio::time::timeout(
                    std::time::Duration::from_millis(300),
                    socket.read(&mut sink),
                )
                .await;
                let chunk = r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
                let body = format!("data: {chunk}\n\ndata: [DONE]\n\n");
                let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
                let _ = socket.write_all(head.as_bytes()).await;
                let _ = socket.write_all(body.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    address
}

#[tokio::test]
async fn the_first_request_carries_the_tools_the_task_needs_and_the_serving_mode() {
    let upstream = stub_model().await;
    let recorder = Recorder::start(&upstream).await.expect("recorder");
    let model = ServedModel::qwen3();
    let base_url = recorder.base_url();
    // A seed drawn for this run: the claim holds for every seed.
    let seed = std::hash::RandomState::new().hash_one(0_u8);

    let scratch = std::env::temp_dir().join(format!("tool-syntax-agent-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).expect("scratch");
    std::env::set_current_dir(&scratch).expect("cwd");

    run_agent(
        "say done",
        &Endpoint {
            model: &model,
            base_url: Some(&base_url),
            api_key: Some("not-checked"),
            seed,
        },
        Bounds {
            tool_calls: 30,
            deadline: Duration::from_secs(60),
        },
    )
    .await
    .expect("the agent answers");

    let schemas = recorder.tool_schemas().await;
    let names: Vec<&str> = schemas
        .iter()
        .filter_map(|t| t.pointer("/function/name").and_then(|n| n.as_str()))
        .collect();
    for needed in ["shell", "read_file", "write_file"] {
        assert!(
            names.contains(&needed),
            "the first request must offer {needed}; it offered {names:?}"
        );
    }

    // Trained for no-think mode and served in it: the request must say so,
    // and carry the episode's seed.
    let request = &recorder.captured().await[0];
    assert_eq!(
        request["chat_template_kwargs"]["enable_thinking"],
        serde_json::json!(false),
        "{request}"
    );
    assert_eq!(request["seed"], serde_json::json!(seed), "{request}");
    std::fs::remove_dir_all(&scratch).ok();
}
