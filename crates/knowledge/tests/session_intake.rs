// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements session intake that grounds every learned
// fact in exact, traceable bytes, for its clients. If your team needs
// expertise in training-data provenance or secret hygiene, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: an ATIF session is captured as a source whose parts are its user
//! and agent steps, addressable by step id; the same bytes are the same
//! source; a trajectory the training projection refuses is refused with the
//! reason; and no secret in any message or tool output survives into a part.

use atif::{
    AgentProfile, ObservationEntry, StepObservation, StepOrigin, ToolInvocation, TraceStep,
    Trajectory,
};
use serde_json::json;
use splinter_core::clock::FixedClock;
use splinter_core::source::Origin;
use splinter_knowledge::capture::{capture_session, SessionRefusal};
use splinter_knowledge::session::{part_name, Role, SessionView};
use splinter_store::sources::SourceStore;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

const API_KEY: &str = "sk-proj-4f9a8c7b2e1d3f6a5b4c3d2e1f0a9b8c";
const PASSWORD: &str = "hunter-two";
const PEM_BODY: &str = "MIIEowIBAAKCAQEAneutralneutralneutral";

fn clock() -> FixedClock {
    FixedClock::new("2026-10-01T08:00:00.000Z")
}

fn store() -> anyhow::Result<(SourceStore, tempfile::TempDir)> {
    let dir = tempfile::tempdir()?;
    let store = SourceStore::new(&Workspace::at(&StateRoot::new(dir.path().join("state"))));
    Ok((store, dir))
}

fn trajectory(steps: Vec<TraceStep>) -> Trajectory {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
    t.session_id = Some("session-one".into());
    t.steps = steps;
    t
}

fn bytes(t: &Trajectory) -> anyhow::Result<Vec<u8>> {
    Ok(serde_json::to_vec(t)?)
}

fn user(step: u64, text: &str) -> TraceStep {
    TraceStep::new(step, StepOrigin::User, text)
}

fn agent(step: u64, text: &str) -> TraceStep {
    TraceStep::new(step, StepOrigin::Agent, text)
}

fn tool_step(step: u64, arguments: serde_json::Value, observation: &str) -> TraceStep {
    let mut s = agent(step, "");
    s.tool_calls = Some(vec![
        ToolInvocation::new("c1", "shell").with_arguments(arguments)
    ]);
    s.observation = Some(StepObservation::single(ObservationEntry::for_call(
        "c1",
        observation,
    )));
    s
}

fn correction_session() -> Trajectory {
    trajectory(vec![
        user(1, "Which port does the Tessera dashboard listen on?"),
        agent(2, "The Tessera dashboard listens on port 8080."),
        user(
            3,
            "No, that is wrong. It listens on port 9090 since the March move.",
        ),
        agent(4, "Thanks, port 9090 it is."),
    ])
}

#[test]
fn a_valid_session_is_recorded_once_and_its_steps_are_addressable() -> anyhow::Result<()> {
    let (store, dir) = store()?;
    let raw = bytes(&correction_session())?;

    let first = capture_session(&raw, &clock())?;
    assert!(!store.contains(&first.source().id)?);
    store.put_source(&first)?;
    let again = capture_session(&raw, &clock())?;
    assert_eq!(
        again.source().id,
        first.source().id,
        "the same bytes, the same source"
    );
    assert!(store.contains(&again.source().id)?);
    store.put_source(&again)?;
    assert_eq!(store.list()?.len(), 1);

    let view = SessionView::of(&first)?;
    let step3 = view.step(3).ok_or_else(|| anyhow::anyhow!("no step 3"))?;
    assert!(step3
        .message
        .as_ref()
        .is_some_and(|m| m.text.contains("port 9090")));
    assert_eq!(
        step3.message.as_ref().map(|m| m.name.as_str()),
        Some(part_name(3, Role::User).as_str())
    );
    let bytes = store.read_part(&first.source().id, &part_name(2, Role::Agent))?;
    assert_eq!(bytes, b"The Tessera dashboard listens on port 8080.");
    assert!(view.step(9).is_none());
    drop(dir);
    Ok(())
}

#[test]
fn a_trajectory_the_projection_refuses_is_refused_with_the_reason() -> anyhow::Result<()> {
    let mixed = trajectory(vec![
        user(1, "Restart the service."),
        tool_step(2, json!({"cmd": "systemctl restart svc"}), "ok"),
        user(3, "Thanks. Now check status."),
        agent(4, "Fine."),
    ]);
    let refused = capture_session(&bytes(&mixed)?, &clock()).err();
    assert!(
        matches!(refused, Some(SessionRefusal::Unprojectable(_))),
        "{refused:?}"
    );
    assert!(refused.is_some_and(|r| r.to_string().contains("step 2")));

    let no_user = trajectory(vec![agent(1, "Hello.")]);
    assert!(matches!(
        capture_session(&bytes(&no_user)?, &clock()).err(),
        Some(SessionRefusal::NoUserStep)
    ));

    let mut bad_version = correction_session();
    bad_version.schema_version = "v1".into();
    assert!(matches!(
        capture_session(&bytes(&bad_version)?, &clock()).err(),
        Some(SessionRefusal::Invalid(_))
    ));
    assert!(matches!(
        capture_session(b"not json at all", &clock()).err(),
        Some(SessionRefusal::NotATrajectory(_))
    ));
    Ok(())
}

#[test]
fn a_secret_in_any_message_or_tool_output_is_removed_before_storage() -> anyhow::Result<()> {
    // The marker lines are assembled so that this file holds no key block.
    let pem = format!(
        "-----BEGIN {0} KEY-----\n{PEM_BODY}\n-----END {0} KEY-----",
        "RSA PRIVATE"
    );
    let session = trajectory(vec![
        user(
            1,
            &format!("Use my key {API_KEY} to log in to the registry."),
        ),
        tool_step(
            2,
            json!({"cmd": "login", "env": {"REGISTRY_PASSWORD": PASSWORD}}),
            &format!("logged in\npassword={PASSWORD}\n{pem}"),
        ),
        agent(3, &format!("Done; the key {API_KEY} worked.")),
    ]);
    let raw = bytes(&session)?;
    assert!(
        String::from_utf8_lossy(&raw).contains(API_KEY),
        "fixture holds the secret"
    );

    let (store, dir) = store()?;
    let captured = capture_session(&raw, &clock())?;
    let id = store.put_source(&captured)?;
    let source = store.get_source(&id)?;
    for part in &source.parts {
        let text = String::from_utf8_lossy(&store.read_blob(&part.content)?).into_owned();
        for secret in [API_KEY, PASSWORD, PEM_BODY] {
            assert!(!text.contains(secret), "{secret} in part {}", part.name);
        }
    }
    let Origin::Session { redactions, .. } = &source.origin else {
        anyhow::bail!("not a session origin");
    };
    let steps: Vec<(Option<u64>, &str)> = redactions
        .iter()
        .map(|r| (r.step, r.kind.as_str()))
        .collect();
    assert!(steps.contains(&(Some(1), "token")), "{steps:?}");
    assert!(steps.contains(&(Some(2), "private_key")), "{steps:?}");
    assert!(steps.contains(&(Some(2), "credential")), "{steps:?}");
    assert!(steps.contains(&(Some(3), "token")), "{steps:?}");
    let origin = serde_json::to_string(&source.origin)?;
    assert!(!origin.contains(API_KEY) && !origin.contains(PASSWORD));

    let view = SessionView::of(&captured)?;
    let shown = format!("{:?}", view.steps().collect::<Vec<_>>());
    assert!(!shown.contains(API_KEY) && !shown.contains(PASSWORD) && !shown.contains(PEM_BODY));
    drop(dir);
    Ok(())
}
