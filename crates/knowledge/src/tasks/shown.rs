// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-task generation whose answers come
// from running code, not from a model's say-so, for its clients. If your team
// needs expertise in verifiable synthetic data, you can procure our services
// by sending an email to info@swedishembedded.com.

//! What the solver of a model-written task is shown, and the answer of a
//! task whose answer is the output of the code it shows: both established by
//! code, so neither depends on the model writing them correctly.

use splinter_sandbox::{CodeCall, RuntimeEnvironment};

use super::admit::Refusal;
use super::generator::GenerateError;
use super::kind::{AnswerForm, TaskKind};
use super::reply::Candidate;
use super::Rejection;

/// `candidate` with the instruction the solver is shown. A kind that shows
/// material has the model write the question alone and the material apart;
/// the material is set below the question here, fenced, so it is shown once
/// and exactly as it is run or graded. A question that already repeats the
/// material is refused: a small model wraps it in other code (`x = [[...]]`)
/// and the solver would be asked about a program that is not the one run.
pub(super) fn shown(kind: &TaskKind, mut candidate: Candidate) -> Result<Candidate, Refusal> {
    let Some(material) = candidate
        .material
        .as_deref()
        .filter(|m| !m.trim().is_empty())
    else {
        return Ok(candidate);
    };
    if !kind.shows_material {
        return Ok(candidate);
    }
    if candidate.instruction.contains(material.trim()) {
        return Err((
            Rejection::MaterialNotShown,
            "the instruction repeats the material, which is shown below it".into(),
        ));
    }
    let fence = match kind.runtime.as_deref() {
        Some("python3") => "python",
        Some(runtime) => runtime,
        None => "",
    };
    candidate.instruction = format!(
        "{}\n\n```{fence}\n{}\n```",
        candidate.instruction.trim_end(),
        material.trim_end()
    );
    Ok(candidate)
}

/// What a program of each runtime reads that the language does not define:
/// a pseudo-random generator's draws, the clock, the environment, input,
/// the collector's figures, compiled bytecode. A computed answer that comes
/// from one is a fact about one machine, not about the language - a seeded
/// `math.random` prints what the C library's generator draws - and training
/// a policy to repeat it teaches noise: a Qwen3-4B fine-tuned on six such
/// answers lost 15 of the 35 other probes it passed.
fn undefined_by_the_language(runtime: &str) -> &'static [&'static str] {
    match runtime {
        "lua" => &[
            "math.random",
            "os.time",
            "os.clock",
            "os.date",
            "os.getenv",
            "os.tmpname",
            "io.read",
            "collectgarbage",
            "string.dump",
        ],
        "python3" => &[
            "import random",
            "from random",
            "random.",
            "import time",
            "time.time",
            "datetime.now",
            "os.environ",
            "os.getenv",
            "uuid",
            "input(",
            "id(",
            "hash(",
        ],
        _ => &[],
    }
}

/// The most characters a computed reference may have: an answer the solver
/// is to write out, not a dump.
const MAX_COMPUTED_OUTPUT: usize = 2_000;

/// For a kind whose answer is the output of the material, the reference is
/// what running the material prints, not what the model says it prints: the
/// runtime is the oracle, and a small model that mispredicts its own
/// program would otherwise have a sound task refused. The material must run
/// to completion, print something of a bounded size, and print the same on
/// a second run (no clock, no unseeded randomness).
pub(super) fn computed(
    kind: &TaskKind,
    runtime: Option<&RuntimeEnvironment>,
    mut candidate: Candidate,
) -> Result<Result<Candidate, Refusal>, GenerateError> {
    let (AnswerForm::Output, Some(runtime), Some(code)) = (
        &kind.answer,
        runtime,
        candidate.material.clone().filter(|m| !m.trim().is_empty()),
    ) else {
        return Ok(Ok(candidate));
    };
    if let Some(name) = kind.runtime.as_deref().and_then(|r| {
        undefined_by_the_language(r)
            .iter()
            .find(|n| code.contains(*n))
    }) {
        return Ok(Err((
            Rejection::ChecksFailed,
            format!(
                "the material's output depends on `{name}`, which the language does not define"
            ),
        )));
    }
    let call = CodeCall { code, stdin: None };
    let first = runtime.run(&call)?;
    let refused = |why: String| Ok(Err((Rejection::ChecksFailed, why)));
    if first.timed_out || first.exit_code != Some(0) {
        let stderr: String = first.stderr.chars().take(300).collect();
        return refused(format!(
            "the material does not run to completion (exit {:?}): {stderr}",
            first.exit_code
        ));
    }
    let printed = first.stdout.trim_end().to_string();
    if printed.trim().is_empty() {
        return refused("the material prints nothing".into());
    }
    if printed.chars().count() > MAX_COMPUTED_OUTPUT {
        return refused(format!(
            "the material prints more than {MAX_COMPUTED_OUTPUT} characters"
        ));
    }
    if runtime.run(&call)?.stdout.trim_end() != printed {
        return refused("the material prints something else when run again".into());
    }
    candidate.reference = printed;
    Ok(Ok(candidate))
}
