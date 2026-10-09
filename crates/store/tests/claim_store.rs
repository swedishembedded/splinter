// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a claim set is stored once under its address and read back; the
//! ledger keeps every ruling in the order it was appended, never twice.

use splinter_core::claim::{
    CitedQuote, ClaimKind, ClaimProposal, ClaimSet, LedgerEntry, Refusal, Ruling, SessionClaims,
};
use splinter_core::digest::Digest;
use splinter_core::source::SourceId;
use splinter_store::claims::ClaimStore;
use splinter_store::error::StoreError;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

type Outcome = Result<(), Box<dyn std::error::Error>>;

fn proposal(statement: &str) -> ClaimProposal {
    ClaimProposal {
        kind: ClaimKind::Fact,
        statement: statement.into(),
        question: "What?".into(),
        quotes: vec![CitedQuote {
            step: 2,
            text: "words".into(),
        }],
        observations: vec![],
        calls: vec![],
        said_wrong: None,
        subject: None,
    }
}

fn session() -> SourceId {
    SourceId(Digest::of(b"session"))
}

fn refused(index: usize, statement: &str) -> LedgerEntry {
    LedgerEntry {
        claim_set: Digest::of(b"set"),
        index,
        session: session(),
        proposal: proposal(statement),
        ruling: Ruling::Refused {
            reason: Refusal::NoUserEvidence,
        },
    }
}

#[test]
fn a_claim_set_is_stored_once_and_read_back() -> Outcome {
    let dir = tempfile::tempdir()?;
    let store = ClaimStore::new(&Workspace::at(&StateRoot::new(dir.path())));
    let set = ClaimSet {
        extractor: "scripted/extractor".into(),
        passes: 1,
        sessions: vec![SessionClaims {
            session: session(),
            proposals: vec![proposal("A.")],
            failure: None,
            unconfirmed: vec![],
        }],
    };
    let id = store.put_set(&set)?;
    assert_eq!(store.put_set(&set)?, id);
    assert_eq!(store.get_set(&id)?, set);
    assert_eq!(store.list_sets()?, vec![id]);
    let missing = store.get_set(&Digest::of(b"nothing"));
    assert!(matches!(missing, Err(StoreError::UnknownClaimSet(_))));
    Ok(())
}

#[test]
fn the_ledger_keeps_every_ruling_in_order_and_never_twice() -> Outcome {
    let dir = tempfile::tempdir()?;
    let store = ClaimStore::new(&Workspace::at(&StateRoot::new(dir.path())));
    let entries = [refused(0, "B."), refused(1, "A."), refused(2, "C.")];
    assert_eq!(store.append(&entries[..2])?, 2);
    assert_eq!(
        store.append(&entries[1..])?,
        1,
        "the repeat is not appended"
    );
    assert_eq!(store.entries()?, entries);
    Ok(())
}

#[test]
fn task_links_and_absorptions_are_kept_once_and_the_first_release_stays() -> Outcome {
    use splinter_core::claim::{Absorption, ClaimId, ClaimTaskLink, TaskRole};
    use splinter_core::release::ReleaseId;

    let dir = tempfile::tempdir()?;
    let store = ClaimStore::new(&Workspace::at(&StateRoot::new(dir.path())));
    let claim = ClaimId(Digest::of(b"claim"));
    let link = |task: &[u8], role| ClaimTaskLink {
        claim: claim.clone(),
        task: Digest::of(task),
        role,
    };
    let links = [
        link(b"q", TaskRole::Question),
        link(b"v1", TaskRole::Train),
        link(b"v2", TaskRole::Stopping),
    ];
    assert_eq!(store.link_tasks(&links)?, 3);
    assert_eq!(store.link_tasks(&links[1..])?, 0, "a link is recorded once");
    assert_eq!(store.task_links()?, links);

    let first = ReleaseId(Digest::of(b"first"));
    let second = ReleaseId(Digest::of(b"second"));
    let absorbed = |release: &ReleaseId| Absorption {
        claim: claim.clone(),
        release: release.clone(),
    };
    assert_eq!(store.absorb(&[absorbed(&first)])?, 1);
    assert_eq!(store.absorb(&[absorbed(&second)])?, 0);
    assert_eq!(store.absorptions()?, [absorbed(&first)]);
    Ok(())
}

#[test]
fn the_claims_a_release_was_trained_on_are_kept_and_the_first_record_stays() -> Outcome {
    use splinter_core::claim::{ClaimId, TrainedClaims};
    use splinter_core::release::ReleaseId;

    let dir = tempfile::tempdir()?;
    let store = ClaimStore::new(&Workspace::at(&StateRoot::new(dir.path())));
    let release = ReleaseId(Digest::of(b"release"));
    assert_eq!(store.trained_on(&release)?, None, "unrecorded is absent");
    let claims = vec![ClaimId(Digest::of(b"a")), ClaimId(Digest::of(b"b"))];
    store.record_trained(&TrainedClaims {
        release: release.clone(),
        claims: claims.clone(),
    })?;
    store.record_trained(&TrainedClaims {
        release: release.clone(),
        claims: vec![],
    })?;
    assert_eq!(store.trained_on(&release)?, Some(claims));
    Ok(())
}
