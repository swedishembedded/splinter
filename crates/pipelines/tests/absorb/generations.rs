// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a policy learns in generations.
//!
//! * Within a generation the adapter is trained again from the generation's
//!   base on the generation's live claims.
//! * A consolidation takes the generation into the weights: a full
//!   checkpoint, gated against the release in use, is the base of the next
//!   generation, whose claim set starts empty. `full` trains every weight,
//!   `merge` folds the adapter in and trains nothing.
//! * The claims a base holds are the next generation's retention: never taught
//!   again, replayed in a bounded sample, and asked at every release, where
//!   losing one refuses the candidate.
//! * `absorb` consolidates a generation once its claims hold more records
//!   than the threshold, and not at it.
//! * A claim the weights hold cannot be deleted from them: forgetting one
//!   trains the base again from the last that did not hold it.
//!
//! Every release records its generation and the base it sits on. The models
//! are scripted and training is a test double; nothing here uses a device.

use std::collections::BTreeSet;

use splinter_core::claim::ClaimId;
use splinter_core::release::ReleaseId;
use splinter_orchestrator::generation::ConsolidationMode;
use splinter_orchestrator::releases::ArtifactKind;
use splinter_pipelines::absorb::consolidate::{
    consolidate, ConsolidateRequest, ConsolidationPolicy,
};
use splinter_pipelines::absorb::generation::{
    consolidation_due, plan_rebuild, replay_budget, retention_sample, Link,
};
use splinter_pipelines::absorb::kit::kept;
use splinter_pipelines::absorb::{absorb, AbsorbRequest, Absorbed};
use splinter_pipelines::claims::{forget, ledger};

use crate::gen_world::{answers_of, serve_release, Smith};
use crate::night::{night, Night, Outcome};

fn set(claims: &[ClaimId]) -> BTreeSet<ClaimId> {
    claims.iter().cloned().collect()
}

fn all_live(w: &Night) -> anyhow::Result<Vec<ClaimId>> {
    Ok(ledger(&w.ctx)?.live.into_iter().map(|l| l.claim).collect())
}

fn consolidation_request() -> ConsolidateRequest {
    ConsolidateRequest {
        rehearsal_share: 0.0,
        gate: splinter_eval::gate::GateConfig {
            serve_startup_secs: 30,
            ..splinter_eval::gate::GateConfig::default()
        },
        ..ConsolidateRequest::default()
    }
}

/// A night on `ns`, never consolidating on its own.
fn night_on(w: &Night, smith: &Smith, name: &str, ns: &[usize]) -> anyhow::Result<Absorbed> {
    let dir = w.sessions(name, ns)?;
    let request = AbsorbRequest {
        consolidation: ConsolidationPolicy {
            after_records: None,
            ..ConsolidationPolicy::default()
        },
        ..w.request(dir)
    };
    let done = absorb(&w.ctx, &request, smith)?.report;
    if let Some(id) = &done.release {
        serve_release(&w.ctx, id);
    }
    Ok(done)
}

fn release_of(done: &Absorbed) -> anyhow::Result<ReleaseId> {
    done.release
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no release: {:#?}", done.gate))
}

#[test]
fn consolidating_makes_a_checkpoint_the_base_of_the_next_generation() -> Outcome {
    let w = night("generation-rollover");
    let smith = Smith::default();
    let first = release_of(&night_on(&w, &smith, "one", &[0, 1, 2])?)?;
    let stored = w.ctx.releases().get(&first)?;
    assert_eq!((stored.epoch(), stored.base_release()), (0, None));

    let recorded = consolidate(&w.ctx, &consolidation_request(), &smith)?;
    let done = recorded.report;
    let consolidated = done
        .consolidated
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("nothing consolidated: {done:#?}"))?;
    let release = done
        .release()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no release: {:#?}", consolidated.gate))?;
    serve_release(&w.ctx, &release);

    // A full checkpoint, the base of the generation it opens.
    let made = w.ctx.releases().get(&release)?;
    assert_eq!(made.manifest.artifact.kind(), ArtifactKind::FullCheckpoint);
    assert_eq!(
        (made.epoch(), made.base_release()),
        (1, Some(release.clone()))
    );
    assert_eq!(made.manifest.parent, Some(first.clone()));
    let mark = made.manifest.generation.as_ref().unwrap();
    assert_eq!(mark.base, None, "trained from the configured base");
    let took = mark.consolidation.as_ref().unwrap();
    assert_eq!(took.mode, ConsolidationMode::Full);
    assert_eq!(took.from, first);
    assert_eq!(
        (took.closing.len(), took.retained.len(), took.rebuilt),
        (3, 0, false)
    );
    // Every weight trained, on the records of the three claims, from the
    // base the adapter had.
    let plan = Smith::last(&smith.fulls);
    assert_eq!(plan.base, w.ctx.config().policy_base);
    assert_eq!(
        splinter_store::runs::read_run(w.ctx.workspace(), &recorded.run)?.seeds["seed"],
        plan.seed,
        "the run states the seed the full tune was handed"
    );
    assert_eq!(smith.merges.lock().unwrap().len(), 0);
    let data = consolidated.dataset.as_ref().unwrap();
    assert_eq!(
        (data.claims.len(), data.trained, data.held_out),
        (3, 54, 12)
    );
    assert_eq!(
        consolidated.replayed_records, 0,
        "the base holds no claim yet"
    );
    let gate = consolidated.gate.as_ref().unwrap();
    assert!(gate.passed && gate.gained.is_empty(), "{gate:#?}");
    assert_eq!(gate.claims.len(), 3);

    // The alias moved to it; the claims it holds are what a later generation
    // retains, in the ledger.
    assert_eq!(w.ctx.releases().alias("default")?, Some(release.clone()));
    let held = w.ctx.claims().trained_on(&release)?.unwrap_or_default();
    assert_eq!(held.len(), 3);
    assert_eq!(
        held.iter().collect::<BTreeSet<_>>(),
        all_live(&w)?.iter().collect::<BTreeSet<_>>()
    );
    let pin = w.ctx.policy_pin("default")?.unwrap();
    assert_eq!((pin.generation, pin.adapter), (1, None));
    // It answers what the adapter did.
    assert_eq!(answers_of(&w.ctx, &release).len(), 3);
    Ok(())
}

#[test]
fn the_next_generation_starts_empty_and_keeps_what_the_base_holds() -> Outcome {
    let w = night("generation-retention");
    let smith = Smith::default();
    night_on(&w, &smith, "one", &[0, 1, 2])?;
    let done = consolidate(&w.ctx, &consolidation_request(), &smith)?.report;
    let base = done.release().cloned().unwrap();
    serve_release(&w.ctx, &base);

    let second = night_on(&w, &smith, "two", &[3, 4])?;
    assert_eq!(second.stopped, None, "{second:#?}");
    let generation = second.generation.as_ref().unwrap();
    assert_eq!((generation.epoch, generation.held.len()), (1, 3));
    assert_eq!(second.live, 5);

    // The adapter is trained on the checkpoint, not on the configured base,
    // on the new claims' records beside a bounded sample of the held ones.
    let plan = Smith::last(&smith.adapters);
    assert_eq!(plan.base, w.ctx.generation_base(&base)?);
    assert_ne!(plan.base, w.ctx.config().policy_base);
    let data = second.dataset.as_ref().unwrap();
    assert_eq!(
        data.claims.len(),
        2,
        "only the generation's own claims are taught"
    );
    let text = std::fs::read_to_string(&data.path)?;
    let own = text.matches("Svc3 uses port").count() + text.matches("Svc4 uses port").count();
    let held = ["Svc0 uses port", "Svc1 uses port", "Svc2 uses port"]
        .iter()
        .map(|s| text.matches(s).count())
        .sum::<usize>();
    assert!(own > 0 && held > 0, "own {own}, held {held}");
    assert!(second.replayed_records > 0, "the held claims are replayed");
    assert!(
        second.replayed_records <= data.trained / 3 + 1,
        "bounded: {} of {}",
        second.replayed_records,
        data.trained
    );

    // The gate asks all five claims; the release records its generation and
    // base, trained on the two new claims alone.
    let gate = second.gate.as_ref().unwrap();
    assert!(gate.passed, "{gate:#?}");
    assert_eq!((gate.claims.len(), gate.gained.len()), (5, 2));
    let release = release_of(&second)?;
    let stored = w.ctx.releases().get(&release)?;
    assert_eq!(
        (stored.epoch(), stored.base_release()),
        (1, Some(base.clone()))
    );
    assert_eq!(stored.manifest.parent, Some(base.clone()));
    assert_eq!(
        w.ctx
            .claims()
            .trained_on(&release)?
            .unwrap_or_default()
            .len(),
        2
    );
    assert_eq!(
        answers_of(&w.ctx, &release).len(),
        5,
        "base and adapter together"
    );
    Ok(())
}

#[test]
fn a_candidate_that_loses_a_claim_the_base_holds_is_refused() -> Outcome {
    let w = night("generation-regression");
    let smith = Smith::default();
    night_on(&w, &smith, "one", &[0, 1, 2])?;
    let base = consolidate(&w.ctx, &consolidation_request(), &smith)?
        .report
        .release()
        .cloned()
        .unwrap();
    serve_release(&w.ctx, &base);

    smith.drops.lock().unwrap().push(1);
    let second = night_on(&w, &smith, "two", &[3, 4])?;
    let gate = second.gate.as_ref().unwrap();
    assert!(!gate.passed && second.release.is_none(), "{gate:#?}");
    assert_eq!(gate.regressed.len(), 1);
    assert!(!gate.retention.passed);
    assert_eq!(
        w.ctx.releases().alias("default")?,
        Some(base),
        "the base stays in use"
    );
    Ok(())
}

#[test]
fn a_merge_folds_the_adapter_in_and_trains_nothing() -> Outcome {
    let w = night("generation-merge");
    let smith = Smith::default();
    let first = release_of(&night_on(&w, &smith, "one", &[0, 1, 2])?)?;

    let request = ConsolidateRequest {
        mode: ConsolidationMode::Merge,
        ..consolidation_request()
    };
    let done = consolidate(&w.ctx, &request, &smith)?.report;
    let release = done.release().cloned().ok_or_else(|| {
        anyhow::anyhow!(
            "no release: {:#?}",
            done.consolidated.as_ref().map(|c| &c.gate)
        )
    })?;
    assert!(smith.fulls.lock().unwrap().is_empty(), "nothing is trained");
    let merges = smith.merges.lock().unwrap().clone();
    assert_eq!(merges.len(), 1);
    assert_eq!(merges[0].0, w.ctx.config().policy_base);
    assert_eq!(merges[0].1, w.ctx.releases().get(&first)?.artifact);

    let made = w.ctx.releases().get(&release)?;
    let took = made
        .manifest
        .generation
        .as_ref()
        .and_then(|g| g.consolidation.as_ref());
    assert_eq!(took.map(|t| t.mode), Some(ConsolidationMode::Merge));
    assert_eq!(made.epoch(), 1);
    assert_eq!(
        made.manifest.datasets,
        w.ctx.releases().get(&first)?.manifest.datasets
    );
    assert_eq!(
        w.ctx
            .claims()
            .trained_on(&release)?
            .unwrap_or_default()
            .len(),
        3
    );
    // The merged weights alone answer what the adapter did.
    assert_eq!(answers_of(&w.ctx, &release), answers_of(&w.ctx, &first));
    Ok(())
}

#[test]
fn a_generation_is_consolidated_above_the_threshold_and_not_at_it() -> Outcome {
    // Three claims hold 54 trained records.
    for (threshold, due) in [(54, false), (53, true)] {
        let w = night(&format!("generation-trigger-{threshold}"));
        let smith = Smith::default();
        let dir = w.sessions("s", &[0, 1, 2])?;
        let request = AbsorbRequest {
            consolidation: ConsolidationPolicy {
                after_records: Some(threshold),
                ..ConsolidationPolicy::default()
            },
            ..w.request(dir)
        };
        let done = absorb(&w.ctx, &request, &smith)?.report;
        assert_eq!(
            done.consolidation.is_some(),
            due,
            "threshold {threshold}: {done:#?}"
        );
        let head = w.ctx.releases().alias("default")?.unwrap();
        let kind = w.ctx.releases().get(&head)?.manifest.artifact.kind();
        assert_eq!(
            kind,
            if due {
                ArtifactKind::FullCheckpoint
            } else {
                ArtifactKind::Adapter
            },
            "threshold {threshold}"
        );
        assert_eq!(smith.fulls.lock().unwrap().len(), usize::from(due));
    }
    assert!(!consolidation_due(54, 54) && consolidation_due(55, 54));
    Ok(())
}

#[test]
fn forgetting_a_claim_the_weights_hold_trains_the_base_again_without_it() -> Outcome {
    let w = night("generation-forget");
    let smith = Smith::default();
    night_on(&w, &smith, "one", &[0, 1, 2])?;
    let first_base = consolidate(&w.ctx, &consolidation_request(), &smith)?
        .report
        .release()
        .cloned()
        .unwrap();
    serve_release(&w.ctx, &first_base);
    let second = night_on(&w, &smith, "two", &[3, 4])?;
    release_of(&second)?;
    let before = smith.fulls.lock().unwrap().len();

    // Claim 1 is in the weights of the checkpoint. It is forgotten.
    let ruled = ledger(&w.ctx)?
        .live
        .into_iter()
        .find(|l| l.statement.contains("Svc1 "))
        .map(|l| l.claim)
        .unwrap();
    forget(&w.ctx, &ruled.to_string())?;

    let dir = w.sessions("two", &[3, 4])?;
    let request = AbsorbRequest {
        consolidation: ConsolidationPolicy {
            after_records: None,
            ..ConsolidationPolicy::default()
        },
        ..w.request(dir)
    };
    let night = absorb(&w.ctx, &request, &smith)?.report;
    let rebuilt = night
        .rebuilt
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no rebuild: {night:#?}"))?;
    assert_eq!(rebuilt.dead, std::slice::from_ref(&ruled));
    assert!(rebuilt.rebuilt && rebuilt.release.is_some(), "{rebuilt:#?}");
    // Trained again from the base that did not hold it, on the two claims
    // left of the generation it closed.
    let plans = smith.fulls.lock().unwrap();
    assert_eq!(plans.len(), before + 1);
    let plan = plans.last().unwrap();
    assert_eq!(plan.base, w.ctx.config().policy_base);
    let text = std::fs::read_to_string(&plan.datasets[0].path)?;
    assert!(
        !text.contains("Svc1 uses port"),
        "the forgotten claim is not trained"
    );
    assert!(text.contains("Svc0 uses port") && text.contains("Svc2 uses port"));
    assert_eq!(rebuilt.closing.len(), 2);
    drop(plans);

    // The generation's own claims were taught again on the rebuilt base, and
    // the claim is gone from the policy.
    let release = release_of(&night)?;
    let stored = w.ctx.releases().get(&release)?;
    let base = stored.base_release().unwrap();
    assert_ne!(base, first_base, "a base of its own");
    let held = w.ctx.claims().trained_on(&base)?.unwrap_or_default();
    assert!(!held.contains(&ruled) && held.len() == 2, "{held:?}");
    let answers = answers_of(&w.ctx, &release);
    assert!(!answers.contains_key("Svc1"), "{answers:?}");
    assert!(answers.contains_key("Svc0") && answers.contains_key("Svc3"));
    assert_eq!(stored.epoch(), 2, "the generation counter does not go back");
    Ok(())
}

#[test]
fn forgetting_a_claim_of_the_current_generation_retrains_the_adapter_only() -> Outcome {
    let w = night("generation-forget-own");
    let smith = Smith::default();
    night_on(&w, &smith, "one", &[0, 1, 2])?;
    let base = consolidate(&w.ctx, &consolidation_request(), &smith)?
        .report
        .release()
        .cloned()
        .unwrap();
    serve_release(&w.ctx, &base);
    release_of(&night_on(&w, &smith, "two", &[3, 4])?)?;
    let fulls = smith.fulls.lock().unwrap().len();

    let ruled = ledger(&w.ctx)?
        .live
        .into_iter()
        .find(|l| l.statement.contains("Svc4 "))
        .map(|l| l.claim)
        .unwrap();
    forget(&w.ctx, &ruled.to_string())?;
    let night = night_on(&w, &smith, "two", &[3, 4])?;
    assert!(night.rebuilt.is_none(), "{night:#?}");
    assert_eq!(
        smith.fulls.lock().unwrap().len(),
        fulls,
        "no weight is trained again"
    );
    let release = release_of(&night)?;
    assert_eq!(w.ctx.releases().get(&release)?.base_release(), Some(base));
    assert!(!answers_of(&w.ctx, &release).contains_key("Svc4"));
    Ok(())
}

#[test]
fn a_merge_cannot_take_a_claim_out_of_weights_that_hold_it() -> Outcome {
    let w = night("generation-merge-forget");
    let smith = Smith::default();
    night_on(&w, &smith, "one", &[0, 1, 2])?;
    let base = consolidate(&w.ctx, &consolidation_request(), &smith)?
        .report
        .release()
        .cloned()
        .unwrap();
    serve_release(&w.ctx, &base);
    release_of(&night_on(&w, &smith, "two", &[3, 4])?)?;
    let ruled = ledger(&w.ctx)?.live[0].claim.clone();
    forget(&w.ctx, &ruled.to_string())?;
    let request = ConsolidateRequest {
        mode: ConsolidationMode::Merge,
        ..consolidation_request()
    };
    let refused = consolidate(&w.ctx, &request, &smith)
        .unwrap_err()
        .to_string();
    assert!(refused.contains("--mode full"), "{refused}");
    Ok(())
}

#[test]
fn a_kit_of_records_is_sampled_in_turn_and_within_the_budget() -> Outcome {
    let w = night("generation-sample");
    let smith = Smith::default();
    night_on(&w, &smith, "one", &[0, 1, 2])?;
    let live: Vec<_> = ledger(&w.ctx)?
        .live
        .iter()
        .map(|l| l.claim.clone())
        .collect();
    let kits = kept(&w.ctx)?;
    let claims: Vec<_> = {
        let ledger = splinter_knowledge::claims::Ledger::new(w.ctx.claims().entries()?);
        ledger
            .live()?
            .into_iter()
            .map(|(id, c)| (id, c.clone()))
            .collect()
    };
    assert_eq!(live.len(), 3);

    // Every claim is replayed before any is replayed twice.
    let sample = retention_sample(&kits, &claims, 4);
    assert_eq!(sample.len(), 4);
    let per_claim: BTreeSet<_> = retention_sample(&kits, &claims, 3)
        .into_iter()
        .map(|r| r.claim)
        .collect();
    assert_eq!(per_claim.len(), 3);
    // The same kits and budget give the same sample, and it never exceeds
    // what the kits hold.
    let again = retention_sample(&kits, &claims, 4);
    assert_eq!(
        sample.iter().map(|r| &r.record).collect::<Vec<_>>(),
        again.iter().map(|r| &r.record).collect::<Vec<_>>()
    );
    assert_eq!(retention_sample(&kits, &claims, 10_000).len(), 54);
    // A quarter of the draws, as many records as that asks for, no more than
    // there are.
    assert_eq!(replay_budget(54, 1000, 0.25), 18);
    assert_eq!(replay_budget(54, 5, 0.25), 5);
    assert_eq!(replay_budget(0, 100, 0.25), 0);
    Ok(())
}

#[test]
fn the_earliest_consolidation_that_took_a_dead_claim_in_decides_the_rebuild() {
    let id = |n: u8| ClaimId(splinter_core::digest::Digest::of(&[n]));
    let release = |n: u8| ReleaseId(splinter_core::digest::Digest::of(&[100 + n]));
    let links = vec![
        Link {
            release: release(1),
            from: None,
            closing: vec![id(1), id(2)],
        },
        Link {
            release: release(2),
            from: Some(release(1)),
            closing: vec![id(3), id(4)],
        },
        Link {
            release: release(3),
            from: Some(release(2)),
            closing: vec![id(5)],
        },
    ];
    let live: BTreeSet<ClaimId> = [1, 2, 4, 5].map(id).into_iter().collect();

    // Claim 3 is in the second consolidation: start from the first, train
    // what is left of the second and third.
    let rebuild = plan_rebuild(&links, &[id(3)], &live).unwrap();
    assert_eq!(rebuild.from, Some(release(1)));
    assert_eq!(set(&rebuild.retrain), set(&[id(4), id(5)]));
    // Claims 1 and 5: the first decides, and the configured base is the last
    // that did not hold either.
    let live: BTreeSet<ClaimId> = [2, 4].map(id).into_iter().collect();
    let rebuild = plan_rebuild(&links, &[id(5), id(1)], &live).unwrap();
    assert_eq!(rebuild.from, None);
    assert_eq!(set(&rebuild.retrain), set(&[id(2), id(4)]));
    // A claim no consolidation took in is nothing to rebuild.
    assert_eq!(plan_rebuild(&links, &[id(9)], &live), None);
}
