// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements distribution of curated training data,
// adapters and evaluation verdicts as immutable, verifiable versions, for its
// clients. If your team needs expertise in sharing learning artifacts between
// machines without losing their provenance, you can procure our services by
// sending an email to info@swedishembedded.com.

//! `publish` and `experiences ingest-hf`: the client of the Hub the
//! arguments name, the recorded run, and the report.

use std::fmt::Write as _;

use serde_json::json;
use splinter_sdk::hub::{ingest, publish, Hub, Ingested, PublishRequest, Published};
use splinter_sdk::runs::record;
use splinter_sdk::{Context, Error};

use crate::cli::{HubArgs, IngestHfArgs, PublishArgs};
use crate::output::{emit, Report};

/// The client of the Hub `args` name, with the token its file holds.
fn hub_of(args: &HubArgs) -> Result<Hub, Error> {
    let token = match &args.token_file {
        Some(path) => Some(
            std::fs::read_to_string(path)
                .map_err(|source| Error::Io {
                    path: path.clone(),
                    source,
                })?
                .trim()
                .to_owned(),
        ),
        None => None,
    };
    Ok(Hub::new(&args.endpoint, token)?)
}

/// Runs `publish`.
pub fn publish_bundle(ctx: &Context, json: bool, args: PublishArgs) -> Result<(), Error> {
    let hub = hub_of(&args.hub)?;
    let request = PublishRequest {
        repo: args.repo,
        sets: args.sets,
        datasets: args.datasets,
        candidates: args.candidates,
        verdicts: args.verdicts,
        examiner_key: args.examiner_key,
    };
    let arguments = json!({ "endpoint": hub.endpoint(), "request": request });
    let published = record(ctx, "publish", &arguments, |_| publish(ctx, &hub, &request))?;
    emit(json, &published);
    Ok(())
}

/// Runs `experiences ingest-hf`.
pub fn ingest_bundle(ctx: &Context, json: bool, args: IngestHfArgs) -> Result<(), Error> {
    let hub = hub_of(&args.hub)?;
    let arguments = json!({ "endpoint": hub.endpoint(), "source": args.source });
    let ingested = record(ctx, "experiences ingest-hf", &arguments, |_| {
        ingest(ctx, &hub, &args.source)
    })?;
    emit(json, &ingested);
    Ok(())
}

impl Report for Published {
    fn human(&self) -> String {
        let mut out = String::new();
        let verb = if self.committed {
            "published"
        } else {
            "already published"
        };
        let _ = writeln!(
            out,
            "{verb} bundle {} to {} at commit {} (tag {})",
            self.content, self.repo, self.commit, self.tag
        );
        let _ = writeln!(
            out,
            "files: {} ({} bytes); sent: {} in the commit, {} through LFS, {} the hub held",
            self.files, self.bytes, self.regular, self.lfs_uploaded, self.lfs_present
        );
        out
    }
}

impl Report for Ingested {
    fn human(&self) -> String {
        let mut out = String::new();
        let _ = write!(out, "ingested bundle {} from {}", self.content, self.repo);
        if self.resolved {
            let _ = write!(out, "@{} (commit {})", self.requested, self.commit);
        } else {
            let _ = write!(out, "@{}", self.commit);
        }
        out.push('\n');
        for set in &self.experiences.sets {
            let _ = writeln!(out, "experience set {set}");
        }
        let _ = writeln!(
            out,
            "experiences: {} new, {} held; datasets: {} new, {} held; candidates: {}; artifacts: {} new; verdicts: {}",
            self.experiences.experiences_new,
            self.experiences.experiences_held,
            self.datasets_new,
            self.datasets_held,
            self.candidates.len(),
            self.artifacts_new,
            self.verdicts
        );
        out
    }
}
