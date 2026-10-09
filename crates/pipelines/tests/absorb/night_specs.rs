// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The composition: a night of sessions becomes a candidate, and a release
//! when the declared rule on counts is met.

use splinter_core::claim::ClaimId;
use splinter_orchestrator::runs;
use splinter_pipelines::absorb::kit::kept;
use splinter_pipelines::absorb::{absorb, AbsorbRequest};
use splinter_pipelines::claims::ledger;
use splinter_pipelines::lineage::{lineage, Direction, LineageRequest, NodeKind};

use crate::night::{
    night, serve_release, session_at, statement, statement_at, Learner, Night, Outcome,
};

const EXTRACT: &str = "List what the person taught";
const VARIANTS: &str = "differently worded questions about one fact";
const FORMS: &str = "Write the fact in other forms";

fn live_claims(w: &Night) -> anyhow::Result<Vec<ClaimId>> {
    Ok(ledger(&w.ctx)?.live.into_iter().map(|l| l.claim).collect())
}

#[test]
fn a_night_teaches_every_claim_from_many_records_and_releases_what_it_answers() -> Outcome {
    let w = night("night-releases");
    let dir = w.sessions("s", &[0, 1, 2])?;
    let learner = Learner::default();
    let done = absorb(&w.ctx, &w.request(dir), &learner)?;
    let r = &done.report;
    assert_eq!(r.stopped, None, "{r:#?}");
    assert_eq!((r.live, r.pending.len()), (3, 3));

    // Each claim: its question and eight paraphrases answered by the
    // teacher, the hindsight dialogue, eight forms; four paraphrases kept
    // out of training.
    let kits = kept(&w.ctx)?;
    assert_eq!(kits.len(), 3);
    for kit in kits.values() {
        assert_eq!(kit.records.len(), 18, "{:#?}", r.kits);
        assert_eq!(kit.stopping.len(), 4);
    }
    let data = r
        .dataset
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no dataset"))?;
    assert_eq!((data.trained, data.held_out), (54, 12));
    assert_eq!(data.claims.len(), 3, "no claim is held out");

    // The agent's wrong reply and its capitulation are never trained on.
    let text = std::fs::read_to_string(&data.path)?;
    assert!(
        !text.contains("8080") && !text.contains("You are right"),
        "{text}"
    );
    assert!(
        text.contains("Svc0 - what port is it on?"),
        "the hindsight question"
    );

    // Retrained from the base, and released under the declared rule.
    let candidate = r
        .candidate
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no candidate"))?;
    assert_eq!(candidate.parent, None);
    let gate = r.gate.as_ref().ok_or_else(|| anyhow::anyhow!("no gate"))?;
    assert!(gate.passed, "{gate:#?}");
    assert_eq!((gate.answered, gate.gained.len()), (3, 3));
    let release = r
        .release
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no release"))?;
    assert_eq!(r.absorbed.len(), 3);

    // The ledger says which release first absorbed each claim.
    let lines = ledger(&w.ctx)?.live;
    assert!(lines
        .iter()
        .all(|l| l.absorbed_by.as_ref() == Some(&release)));
    // A stopping paraphrase traces to its claim and the person's words.
    let stopping = kits
        .values()
        .next()
        .and_then(|k| k.stopping.first())
        .ok_or_else(|| anyhow::anyhow!("no stopping paraphrase"))?;
    let up = lineage(
        &w.ctx,
        &LineageRequest {
            id: stopping.to_string(),
            direction: Direction::Up,
            depth: None,
        },
    )?;
    let kinds: Vec<NodeKind> = up.nodes.iter().map(|n| n.kind).collect();
    assert!(
        kinds.contains(&NodeKind::Claim) && kinds.contains(&NodeKind::Span),
        "{kinds:?}"
    );
    assert!(runs::list(&w.ctx)?
        .runs
        .iter()
        .any(|run| run.command == "absorb"));
    Ok(())
}

#[test]
fn a_dry_run_ends_at_the_rulings_and_trains_nothing() -> Outcome {
    let w = night("night-dry");
    let dir = w.sessions("s", &[0, 1])?;
    let learner = Learner::default();
    let request = AbsorbRequest {
        dry_run: true,
        ..w.request(dir)
    };
    let r = absorb(&w.ctx, &request, &learner)?.report;
    assert!(
        r.stopped.as_deref().is_some_and(|s| s.contains("dry run")),
        "{:?}",
        r.stopped
    );
    assert_eq!(
        r.claims.as_ref().map(|c| c.admitted),
        Some(2),
        "{:#?}",
        r.claims
    );
    assert!(r.kits.is_none() && r.candidate.is_none());
    assert!(kept(&w.ctx)?.is_empty() && w.asked(VARIANTS) == 0);
    assert!(learner
        .plans
        .lock()
        .map_err(|_| anyhow::anyhow!("poisoned"))?
        .is_empty());
    assert!(r.finished());
    Ok(())
}

#[test]
fn no_release_stops_at_the_candidate_and_absorbs_nothing() -> Outcome {
    let w = night("night-candidate");
    let dir = w.sessions("s", &[0, 1])?;
    let learner = Learner::default();
    let request = AbsorbRequest {
        no_release: true,
        ..w.request(dir.clone())
    };
    let r = absorb(&w.ctx, &request, &learner)?.report;
    assert!(r.candidate.is_some() && r.gate.is_none() && r.release.is_none());
    assert!(w.ctx.releases().list()?.is_empty());
    assert!(ledger(&w.ctx)?.live.iter().all(|l| l.absorbed_by.is_none()));

    // Run again: what was kept is not asked for again.
    let asked = (w.asked(EXTRACT), w.asked(VARIANTS), w.asked(FORMS));
    let again = absorb(&w.ctx, &w.request(dir), &learner)?.report;
    assert!(again.release.is_some(), "{again:#?}");
    assert_eq!(asked, (w.asked(EXTRACT), w.asked(VARIANTS), w.asked(FORMS)));
    assert_eq!(
        again.kits.as_ref().map(|k| (k.built, k.reused)),
        Some((0, 2))
    );
    Ok(())
}

#[test]
fn a_candidate_the_gate_refuses_stays_on_record_and_the_release_stays_in_use() -> Outcome {
    let w = night("night-refused");
    let dir = w.sessions("s", &[0, 1])?;
    let learner = Learner::default();
    learner
        .forgets
        .lock()
        .map_err(|_| anyhow::anyhow!("poisoned"))?
        .extend([0, 1]);
    let r = absorb(&w.ctx, &w.request(dir), &learner)?.report;
    let gate = r.gate.as_ref().ok_or_else(|| anyhow::anyhow!("no gate"))?;
    assert!(!gate.passed && !gate.improvement.passed, "{gate:#?}");
    assert_eq!(gate.unanswered.len(), 2);
    assert!(r.release.is_none() && w.ctx.releases().alias("default")?.is_none());
    let candidate = r
        .candidate
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no candidate"))?;
    assert!(splinter_pipelines::train::load_candidate(&w.ctx, &candidate.candidate).is_ok());
    assert!(r
        .stopped
        .as_deref()
        .is_some_and(|s| s.contains("stays in use")));
    assert!(!r.finished());
    Ok(())
}

#[test]
fn the_next_night_retrains_from_the_base_on_every_live_claim() -> Outcome {
    let w = night("night-two");
    let learner = Learner::default();
    let first = absorb(&w.ctx, &w.request(w.sessions("one", &[0, 1])?), &learner)?.report;
    let release = first
        .release
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no release"))?;
    serve_release(&w.ctx, &release);

    let second = absorb(&w.ctx, &w.request(w.sessions("two", &[2])?), &learner)?.report;
    let gate = second
        .gate
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no gate"))?;
    assert!(gate.passed, "{gate:#?}");
    assert_eq!(live_claims(&w)?.len(), 3);
    let data = second
        .dataset
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no dataset"))?;
    assert_eq!(
        data.claims.len(),
        3,
        "the earlier claims are in tonight's set too"
    );
    let plans = learner
        .plans
        .lock()
        .map_err(|_| anyhow::anyhow!("poisoned"))?;
    assert_eq!(
        plans[1].parent, None,
        "trained from the base, not the release"
    );
    // Only the new claim is credited to the new release.
    assert_eq!(second.absorbed.len(), 1);
    let new_release = second
        .release
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no release"))?;
    let lines = ledger(&w.ctx)?.live;
    assert_eq!(
        lines
            .iter()
            .filter(|l| l.absorbed_by.as_ref() == Some(&release))
            .count(),
        2
    );
    assert_eq!(
        lines
            .iter()
            .filter(|l| l.absorbed_by.as_ref() == Some(&new_release))
            .count(),
        1
    );
    assert_eq!(
        w.ctx.releases().get(&new_release)?.manifest.parent,
        Some(release)
    );
    Ok(())
}

#[test]
fn a_claim_regressed_by_the_candidate_fails_retention_with_no_tolerance() -> Outcome {
    let w = night("night-regress");
    let learner = Learner::default();
    let first = absorb(&w.ctx, &w.request(w.sessions("one", &[0, 1])?), &learner)?.report;
    serve_release(
        &w.ctx,
        &first
            .release
            .clone()
            .ok_or_else(|| anyhow::anyhow!("no release"))?,
    );
    // Tonight's training loses service 0.
    learner
        .forgets
        .lock()
        .map_err(|_| anyhow::anyhow!("poisoned"))?
        .push(0);
    let second = absorb(&w.ctx, &w.request(w.sessions("two", &[2])?), &learner)?.report;
    let gate = second
        .gate
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no gate"))?;
    assert!(
        gate.improvement.passed && !gate.retention.passed,
        "{gate:#?}"
    );
    assert_eq!(gate.regressed.len(), 1);
    assert!(second.release.is_none());
    Ok(())
}

#[test]
fn a_sealed_probe_in_a_record_refuses_that_record_and_names_the_probe() -> Outcome {
    let w = night("night-sealed");
    let dir = w.sessions("s", &[0, 1])?;
    let probes = w.scratch.0.join("probes.jsonl");
    std::fs::write(
        &probes,
        "{\"name\": \"p-bound\", \"question\": \"What port is Svc1 bound to these days?\"}\n",
    )?;
    let learner = Learner::default();
    let request = AbsorbRequest {
        sealed_probes: vec![probes],
        no_release: true,
        ..w.request(dir)
    };
    let r = absorb(&w.ctx, &request, &learner)?.report;
    let data = r
        .dataset
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no dataset"))?;
    assert!(
        data.refused.iter().all(|x| x.leak.probe == "p-bound") && !data.refused.is_empty(),
        "{:#?}",
        data.refused
    );
    let text = std::fs::read_to_string(&data.path)?;
    let found: Vec<&str> = text
        .lines()
        .filter(|l| l.contains("Svc1 bound to these days"))
        .collect();
    assert!(
        found.is_empty(),
        "{:#?}\n{}",
        data.refused,
        found.join("\n")
    );
    Ok(())
}

#[test]
fn a_superseded_claim_drops_out_of_the_next_nights_set() -> Outcome {
    let w = night("night-supersede");
    let learner = Learner::default();
    let first = absorb(&w.ctx, &w.request(w.sessions("one", &[0, 1])?), &learner)?.report;
    serve_release(
        &w.ctx,
        &first
            .release
            .clone()
            .ok_or_else(|| anyhow::anyhow!("no release"))?,
    );

    // Service 0 moved to another port.
    let dir = w.scratch.0.join("two");
    std::fs::create_dir_all(&dir)?;
    std::fs::write(
        dir.join("moved.atif.json"),
        serde_json::to_vec(&session_at(0, 9100))?,
    )?;
    let second = absorb(&w.ctx, &w.request(dir), &learner)?.report;
    let gate = second
        .gate
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no gate"))?;
    assert!(gate.passed, "{gate:#?}");
    assert_eq!(second.claims.as_ref().map(|c| c.superseded), Some(1));
    let data = second
        .dataset
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no dataset"))?;
    let text = std::fs::read_to_string(&data.path)?;
    assert!(!text.contains(&statement(0)) && text.contains("Svc0 uses port 9100"));
    assert!(
        text.contains(&statement(1)),
        "the claim that stands is still taught"
    );
    assert_eq!(live_claims(&w)?.len(), 2);
    assert_eq!(ledger(&w.ctx)?.superseded.len(), 1);
    Ok(())
}

/// A night of the one session `trajectory`, in a directory of its own.
fn one_session(
    w: &Night,
    name: &str,
    trajectory: &atif::Trajectory,
) -> anyhow::Result<std::path::PathBuf> {
    let dir = w.scratch.0.join(name);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("day.atif.json"), serde_json::to_vec(trajectory)?)?;
    Ok(dir)
}

fn on_day(name: &str, n: usize, port: usize) -> atif::Trajectory {
    let mut t = session_at(n, port);
    t.session_id = Some(name.into());
    t
}

#[test]
fn a_claim_corrected_back_and_forth_leaves_the_last_one_live_with_every_ruling_kept() -> Outcome {
    let w = night("night-canary");
    let learner = Learner::default();
    let mut last = None;
    // Day 1 says port 9000, day 2 says 9100, day 3 says 9000 again.
    for (day, port) in [("day1", 9000), ("day2", 9100), ("day3", 9000)] {
        let dir = one_session(&w, day, &on_day(day, 0, port))?;
        let report = absorb(&w.ctx, &w.request(dir), &learner)?.report;
        let gate = report
            .gate
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no gate on {day}: {report:#?}"))?;
        assert!(gate.passed, "{day}: {gate:#?}");
        serve_release(
            &w.ctx,
            report
                .release
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("no release on {day}"))?,
        );
        last = Some(report);
    }
    let live = ledger(&w.ctx)?;
    assert_eq!(live.live.len(), 1);
    assert_eq!(live.live[0].statement, statement_at(0, 9000));
    assert_eq!(
        live.live[0].session,
        last.ok_or_else(|| anyhow::anyhow!("no night"))?
            .intake
            .ok_or_else(|| anyhow::anyhow!("no intake"))?
            .sessions[0]
            .source
    );
    assert_eq!(live.superseded.len(), 2, "day 1 and day 2 stay, replaced");
    assert_eq!(w.ctx.claims().entries()?.len(), 3, "every ruling is kept");
    Ok(())
}

#[test]
fn a_fact_said_again_is_recorded_as_reinforced_and_stays_one_live_claim() -> Outcome {
    let w = night("night-reinforced");
    let learner = Learner::default();
    let first = absorb(
        &w.ctx,
        &w.request(one_session(&w, "day1", &on_day("day1", 0, 9000))?),
        &learner,
    )?
    .report;
    serve_release(
        &w.ctx,
        first
            .release
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no release"))?,
    );
    let again = absorb(
        &w.ctx,
        &w.request(one_session(&w, "day2", &on_day("day2", 0, 9000))?),
        &learner,
    )?
    .report;
    assert_eq!(
        again.claims.as_ref().map(|c| c.reinforced),
        Some(1),
        "{again:#?}"
    );
    let after = ledger(&w.ctx)?;
    assert_eq!(after.live.len(), 1);
    assert_eq!(after.live[0].reinforced, 1);
    assert_eq!(
        after.live[0].session,
        first
            .intake
            .ok_or_else(|| anyhow::anyhow!("no intake"))?
            .sessions[0]
            .source,
        "the first claim stays the live one"
    );
    assert!(after.refused.is_empty());
    Ok(())
}
