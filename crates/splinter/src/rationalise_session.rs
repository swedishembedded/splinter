// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! `splinter experiences rationalise`: the hinted retry over a compiled
//! Harbor dataset, and what it prints.

use std::fmt::Write as _;

use serde_json::json;
use splinter_sdk::bench::practice::BenchCheckout;
use splinter_sdk::bench::repair::hinted::filters::{HintFraming, LEAK_TERMS};
use splinter_sdk::bench::repair::hinted::{
    rationalise, RationaliseReport, RationaliseRequest, DEFAULT_RATIONALISE_JOBS, DEFAULT_SAMPLES,
};
use splinter_sdk::runs::record;
use splinter_sdk::{Context, Error};

use crate::cli::{HintFramingArg, RationaliseArgs};
use crate::output::{emit, Report};

/// Runs the command and prints its report.
///
/// # Errors
///
/// As the run is refused or fails.
pub(crate) fn rationalise_command(
    ctx: &Context,
    json: bool,
    args: RationaliseArgs,
) -> Result<(), Error> {
    let mut checkout = BenchCheckout::new(args.bench.clone());
    if let Some(harbor) = &args.harbor {
        checkout.harbor.clone_from(harbor);
    }
    checkout.sven_build = Some(args.sven_build.clone());
    let images = args
        .images
        .clone()
        .unwrap_or_else(|| args.bench.join(".bench-data").join("rootfs-cache"));
    let leak_terms: Vec<String> = if args.leak_terms.is_empty() {
        LEAK_TERMS.iter().map(|t| (*t).to_string()).collect()
    } else {
        args.leak_terms.clone()
    };
    let framing = match args.hint_framing {
        HintFramingArg::WorkedAnswer => HintFraming::WorkedAnswer,
        HintFramingArg::PrivateScratch => HintFraming::PrivateScratch,
    };
    let samples = args.samples.unwrap_or(DEFAULT_SAMPLES);
    let jobs = args.jobs.unwrap_or(DEFAULT_RATIONALISE_JOBS);
    let arguments = json!({
        "dataset": args.dataset, "students": args.students,
        "rationaliser": args.rationaliser.to_string(), "bench": args.bench,
        "harbor": args.harbor, "sven_build": args.sven_build, "profile": args.profile,
        "images": images, "count": args.count, "seed": args.seed, "samples": samples,
        "jobs": jobs, "leak_terms": leak_terms, "hint_framing": framing,
    });
    let report = record(ctx, "experiences rationalise", &arguments, |run| {
        let seed = run.seed("seed", args.seed)?;
        rationalise(
            ctx,
            &RationaliseRequest {
                dataset: &args.dataset,
                students: &args.students,
                checkout: &checkout,
                rationaliser: &args.rationaliser,
                profile: &args.profile,
                images: &images,
                count: args.count,
                samples,
                seed: seed.value(),
                jobs,
                leak_terms: &leak_terms,
                framing,
                cancel: run.cancel_token(),
            },
        )
        .map(Printed)
    })?;
    emit(json, &report);
    Ok(())
}

/// The report, as the command prints it.
#[derive(serde::Serialize)]
#[serde(transparent)]
struct Printed(RationaliseReport);

impl Report for Printed {
    fn human(&self) -> String {
        let r = &self.0;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "rationalised by {} (served as {}), seed {}, {} tasks x {} samples",
            r.rationaliser, r.served_as, r.seed, r.count, r.samples
        );
        let s = &r.students;
        let _ = writeln!(
            out,
            "student: {} trials, {} passed, {} failed, {} tasks failed at least once",
            s.trials, s.passed, s.failed, s.failed_tasks
        );
        let f = &r.funnel;
        let _ = writeln!(
            out,
            "attempts graded {}: verified {}, real observations {}, no leak {}, sufficient {}",
            f.graded, f.verified, f.real_observations, f.no_leak, f.sufficient
        );
        let a = &r.alone;
        let _ = writeln!(
            out,
            "each filter alone: verified {}, real observations {}, no leak {}",
            a.verified, a.real_observations, a.no_leak
        );
        let _ = writeln!(
            out,
            "kept: {} own passes, {} rationalised; undecided {}",
            r.own, r.rationalised, r.undecided
        );
        let mentions: Vec<String> = r
            .own_term_mentions
            .iter()
            .map(|(term, n)| format!("{term:?} {n}"))
            .collect();
        let _ = writeln!(
            out,
            "leak terms the {} own passes (no hint) mention: {}",
            r.own,
            mentions.join(", ")
        );
        let _ = writeln!(
            out,
            "own passes that would pass the real-observations filter: {} of {}",
            r.own_real_observations, r.own
        );
        if let Some(set) = &r.set {
            let _ = writeln!(out, "experience set {set}");
        }
        let _ = writeln!(out, "run directory {}", r.dir.display());
        out
    }
}
