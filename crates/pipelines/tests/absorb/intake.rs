// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Intake: sessions recorded once, refusals named, secrets gone.

use splinter_core::source::Origin;
use splinter_orchestrator::runs;
use splinter_pipelines::sessions::{intake, IntakeRequest, DEFAULT_MAX_SESSION_BYTES};
use splinter_pipelines::sources;

use crate::fixtures::{deploy, mixed, port_correction, with_secret, world, Outcome, API_KEY};

fn request(paths: &[std::path::PathBuf]) -> IntakeRequest<'_> {
    IntakeRequest {
        paths,
        max_bytes: DEFAULT_MAX_SESSION_BYTES,
    }
}

#[test]
fn a_session_is_recorded_once_and_the_same_file_again_is_a_no_op() -> Outcome {
    let w = world(|_| String::new())?;
    let path = w.write("monday.atif.json", &port_correction())?;
    let paths = [path];

    let first = runs::record(&w.ctx, "session add", &paths, |_| {
        intake(&w.ctx, &request(&paths))
    })?;
    assert_eq!((first.report.sessions.len(), first.report.new), (1, 1));
    assert!(first.report.refused.is_empty());
    assert_eq!(first.report.sessions[0].steps, 4);

    let again = intake(&w.ctx, &request(&paths))?;
    assert_eq!((again.sessions.len(), again.new), (1, 0));
    assert_eq!(again.sessions[0].source, first.report.sessions[0].source);
    assert_eq!(sources::list(&w.ctx)?.sources.len(), 1, "recorded once");

    let listed = runs::list(&w.ctx)?;
    assert!(listed.runs.iter().any(|r| r.command == "session add"));
    Ok(())
}

#[test]
fn a_directory_is_taken_whole_and_a_refused_session_is_named_with_its_reason() -> Outcome {
    let w = world(|_| String::new())?;
    w.write("a-correction.atif.json", &port_correction())?;
    w.write("b-deploy.atif.json", &deploy())?;
    w.write("c-mixed.atif.json", &mixed())?;
    std::fs::write(w.sessions_dir().join("d-broken.atif.json"), b"{ not json")?;
    std::fs::write(w.sessions_dir().join("notes.txt"), b"not a session")?;
    let paths = [w.sessions_dir()];

    let report = intake(&w.ctx, &request(&paths))?;
    assert_eq!(report.sessions.len(), 2, "{report:#?}");
    assert_eq!(report.refused.len(), 2, "{report:#?}");
    let mixed_reason = report
        .refused
        .iter()
        .find(|r| r.path.ends_with("c-mixed.atif.json"))
        .map(|r| r.reason.as_str());
    assert!(
        mixed_reason.is_some_and(|r| r.contains("step 2") && r.contains("tool call")),
        "{mixed_reason:?}"
    );
    assert!(report
        .refused
        .iter()
        .any(|r| r.path.ends_with("d-broken.atif.json") && r.reason.contains("not an ATIF")));
    assert_eq!(
        sources::list(&w.ctx)?.sources.len(),
        2,
        "only the valid ones"
    );
    Ok(())
}

#[test]
fn nothing_to_take_is_refused() -> Outcome {
    let w = world(|_| String::new())?;
    std::fs::create_dir_all(w.sessions_dir())?;
    let paths = [w.sessions_dir()];
    assert!(intake(&w.ctx, &request(&paths)).is_err());
    let missing = [w.dir.path().join("nowhere.atif.json")];
    let error = intake(&w.ctx, &request(&missing))
        .err()
        .map(|e| e.to_string());
    assert!(error.is_some_and(|e| e.contains("nowhere.atif.json")));
    Ok(())
}

#[test]
fn a_secret_pasted_in_a_session_is_not_stored() -> Outcome {
    let w = world(|_| String::new())?;
    let path = w.write("secret.atif.json", &with_secret())?;
    let paths = [path];
    let report = intake(&w.ctx, &request(&paths))?;
    assert_eq!(report.redactions, 1);

    let store = w.ctx.sources();
    let source = store.get_source(&report.sessions[0].source)?;
    for part in &source.parts {
        let text = String::from_utf8_lossy(&store.read_blob(&part.content)?).into_owned();
        assert!(!text.contains(API_KEY), "{API_KEY} in part {}", part.name);
    }
    let Origin::Session { redactions, .. } = &source.origin else {
        anyhow::bail!("not a session");
    };
    assert_eq!(redactions.len(), 1);
    assert_eq!(report.sessions[0].redactions, *redactions);
    Ok(())
}
