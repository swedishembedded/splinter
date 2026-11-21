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
use splinter_sdk::bench::repair::hinted::filters::HintFraming;
use splinter_sdk::bench::repair::hinted::{
    rationalise, Attempts, Checks, RationaliseReport, RationaliseRequest, DEFAULT_RATIONALISE_JOBS,
    DEFAULT_SAMPLES,
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
    let checks = Checks {
        real_observations: args.real_observations,
        sufficiency: args.sufficiency,
    };
    let regrade = !args.regrade.is_empty();
    if regrade && checks.sufficiency && args.sven_build.is_none() {
        return Err(Error::Refused(
            "--regrade with --sufficiency runs trials: --sven-build, --profile and \
             --rationaliser are needed"
                .into(),
        ));
    }
    let mut checkout = BenchCheckout::new(args.bench.clone());
    if let Some(harbor) = &args.harbor {
        checkout.harbor.clone_from(harbor);
    }
    checkout.sven_build.clone_from(&args.sven_build);
    let images = args
        .images
        .clone()
        .unwrap_or_else(|| args.bench.join(".bench-data").join("rootfs-cache"));
    let framing = match args.hint_framing {
        HintFramingArg::WorkedAnswer => HintFraming::WorkedAnswer,
        HintFramingArg::PrivateScratch => HintFraming::PrivateScratch,
    };
    let samples = args.samples.unwrap_or(DEFAULT_SAMPLES);
    let jobs = args.jobs.unwrap_or(DEFAULT_RATIONALISE_JOBS);
    let arguments = json!({
        "dataset": args.dataset, "students": args.students,
        "rationaliser": args.rationaliser.as_ref().map(ToString::to_string), "bench": args.bench,
        "harbor": args.harbor, "sven_build": args.sven_build, "profile": args.profile,
        "images": images, "count": args.count, "seed": args.seed, "samples": samples,
        "jobs": jobs, "leak_terms": args.leak_terms, "hint_framing": framing, "checks": checks,
        "regrade": args.regrade,
    });
    let count = match (regrade, args.count) {
        (false, None) => return Err(Error::Refused("--count is needed without --regrade".into())),
        (_, count) => count.unwrap_or_default(),
    };
    let report = record(ctx, "experiences rationalise", &arguments, |run| {
        let attempts = if regrade {
            Attempts::Regrade(&args.regrade)
        } else {
            let seed = run.seed("seed", args.seed)?;
            Attempts::Draw {
                count,
                samples,
                seed: seed.value(),
            }
        };
        rationalise(
            ctx,
            &RationaliseRequest {
                dataset: &args.dataset,
                students: &args.students,
                checkout: &checkout,
                rationaliser: args.rationaliser.as_ref(),
                profile: args.profile.as_deref(),
                images: &images,
                attempts,
                jobs,
                leak_terms: &args.leak_terms,
                framing,
                checks,
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
        if let (Some(rationaliser), Some(served_as)) = (&r.rationaliser, &r.served_as) {
            let _ = writeln!(
                out,
                "rationalised by {rationaliser} (served as {served_as})"
            );
        }
        if let (Some(seed), Some(count), Some(samples)) = (r.seed, r.count, r.samples) {
            let _ = writeln!(out, "seed {seed}, {count} tasks x {samples} samples");
        }
        for run in &r.regraded {
            let _ = writeln!(out, "re-graded {}", run.display());
        }
        let _ = writeln!(
            out,
            "checks: verified, no answer present{}{}{}",
            if r.leak_terms.is_empty() {
                ""
            } else {
                ", leak terms"
            },
            if r.checks.real_observations {
                ", real observations"
            } else {
                ""
            },
            if r.checks.sufficiency {
                ", sufficiency"
            } else {
                ""
            }
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
            "attempts graded {}: verified {}, real observations {}, no answer present {}, sufficient {} (a check left off counts as passed)",
            f.graded, f.verified, f.real_observations, f.no_leak, f.sufficient
        );
        let a = &r.alone;
        let _ = writeln!(
            out,
            "each filter alone: verified {}, real observations {}, no answer present {}",
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
