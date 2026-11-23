// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model evaluations whose verdicts an
// independent examiner signs, for its clients. If your team needs expertise
// in protected model evaluation, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Protected plans: the candidate handed to the plan's examiner (its command
//! or one of the repository's operators), and its signed verdict kept only
//! when it verifies.

#[allow(clippy::wildcard_imports)]
use super::*;

/// The examiner, as the plan's command, run from the repository root with
/// `--candidate FILE [--predecessor FILE] --out FILE` appended.
fn examine_by_command(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    command: &[String],
    candidate: &std::path::Path,
    predecessor: Option<&std::path::Path>,
    out: &std::path::Path,
) -> Result<(), RunError> {
    let Some((program, args)) = command.split_first() else {
        return Err(stage_error(&stage.key, "the authority's command is empty"));
    };
    let mut run = Command::new(program);
    run.args(args).arg("--candidate").arg(candidate);
    if let Some(file) = predecessor {
        run.arg("--predecessor").arg(file);
    }
    let status = run
        .arg("--out")
        .arg(out)
        .current_dir(&factory.resolved.repository)
        .status()
        .map_err(|e| stage_error(&stage.key, format!("{program}: {e}")))?;
    if status.success() {
        Ok(())
    } else {
        Err(stage_error(
            &stage.key,
            format!("the examiner exited with {status}"),
        ))
    }
}

/// The examiner as one of the repository's operators: its inputs the
/// candidate file, the predecessor file and the trained candidate's view
/// (`model`), all read-only, it writes `verdict.json` into its output.
fn examine_by_operator(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    operator: &str,
    candidate: &std::path::Path,
    predecessor: Option<&std::path::Path>,
    model: &std::path::Path,
    out: &std::path::Path,
) -> Result<(), RunError> {
    let executor = factory
        .resolved
        .registry
        .get(operator)
        .and_then(|o| o.executor())
        .ok_or_else(|| stage_error(&stage.key, format!("no operator {operator}")))?;
    let dir = |file: &std::path::Path| file.parent().map(std::path::Path::to_path_buf);
    let mut inputs = std::collections::BTreeMap::new();
    inputs.insert("candidate".to_owned(), dir(candidate).into_iter().collect());
    inputs.insert("model".to_owned(), vec![model.to_path_buf()]);
    if let Some(file) = predecessor {
        inputs.insert("predecessor".to_owned(), dir(file).into_iter().collect());
    }
    let written = tempfile::tempdir().map_err(|e| stage_error(&stage.key, e))?;
    let call = crate::execute::Call {
        inputs,
        params: &json!({}),
        files: &std::collections::BTreeMap::new(),
        models: &std::collections::BTreeMap::new(),
        variables: factory.services.variables,
        out: written.path().to_path_buf(),
    };
    match executor.execute(&call) {
        Ok(crate::execute::Outcome::Produced) => {}
        Ok(crate::execute::Outcome::Rejected(why)) => {
            return Err(stage_error(
                &stage.key,
                format!("the examiner refused: {why}"),
            ))
        }
        Err(why) => return Err(stage_error(&stage.key, why)),
    }
    let verdict = written.path().join("verdict.json");
    std::fs::copy(&verdict, out).map_err(io(&verdict))?;
    Ok(())
}

/// Hand the candidate to the examiner and keep its signed verdict. The
/// examiner reads the candidate as it reads any (`adapter`, its
/// `adapter_digest`, the builds that made it); Splinter appends
/// `--candidate FILE --out FILE` to the plan's command. The verdict counts
/// only when the promotion gate's check passes: the examiner's signature
/// under the plan's public key, its schema, its own validity, that it is
/// about this candidate against the base, and its level. The examiner's
/// outcome decides: `Promote` is improved, `Reject` not improved, anything
/// else inconclusive. The signed document is kept verbatim as
/// `verdict.json`.
pub(super) fn protected(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    section: &Evaluation,
    plan: &EvalPlan,
    authority: &crate::eval_plan::Authority,
    candidate: &splinter_core::factory::product::ModelCandidate,
    staging: &Staging,
) -> Result<(), RunError> {
    let training = factory.view(&section.candidate);
    let adapter = candidate
        .artifacts
        .iter()
        .find(|a| a.path.ends_with(".safetensors"))
        .ok_or_else(|| {
            stage_error(
                &stage.key,
                "the examiner examines adapters; the candidate has none",
            )
        })?;
    let digest = adapter.digest.to_string();
    let examined = json!({
        "adapter": training.join(&adapter.path),
        "adapter_digest": digest,
        "series": candidate.name,
        "step": 0,
        "proposer_commit": Factory::implementation(""),
        "brain_commit": candidate.trainer,
    });
    let exam = staging.path().join("examined");
    let candidate_file = exam.join("candidate/candidate.json");
    std::fs::create_dir_all(exam.join("candidate")).map_err(io(&exam))?;
    write_json(&candidate_file, &examined)?;
    // A champion baseline with an adapter is the examiner's predecessor; the
    // base model as it is otherwise.
    let champion = factory
        .resolved
        .champions
        .get(&section.baseline.to_string())
        .filter(|c| c.adapter.is_some());
    let mut predecessor = None;
    let mut predecessor_digest = None;
    if let Some(champion) = champion {
        let file = exam.join("predecessor/predecessor.json");
        std::fs::create_dir_all(exam.join("predecessor")).map_err(io(&exam))?;
        let digest = champion
            .record
            .adapter_digest
            .as_ref()
            .map(ToString::to_string);
        write_json(
            &file,
            &json!({
                "adapter": champion.adapter, "adapter_digest": digest,
                "series": champion.record.candidate, "step": 0,
            }),
        )?;
        predecessor = Some(file);
        predecessor_digest = digest;
    }
    let out = staging.path().join("verdict.json");
    match &authority.operator {
        Some(operator) => examine_by_operator(
            factory,
            stage,
            operator,
            &candidate_file,
            predecessor.as_deref(),
            &training,
            &out,
        )?,
        None => examine_by_command(
            factory,
            stage,
            &authority.command,
            &candidate_file,
            predecessor.as_deref(),
            &out,
        )?,
    }
    let document: serde_json::Value = read_json(&out)?;
    let key = PublicKey::read(&factory.resolved.repository.join(&authority.public_key))
        .map_err(|e| stage_error(&stage.key, format!("{}: {e}", authority.public_key)))?;
    let comparisons = plan
        .decision
        .comparisons
        .or(section.comparisons)
        .unwrap_or(1)
        .max(1);
    let alpha = plan.decision.alpha / f64::from(comparisons);
    let expected = Pair {
        candidate: Some(digest),
        predecessor: predecessor_digest,
    };
    let verified = verify(&document, &key, &expected, alpha, Role::Sealed).map_err(|e| {
        stage_error(
            &stage.key,
            format!("the examiner's verdict does not verify: {e:?}"),
        )
    })?;
    let decision = match document.get("verdict").and_then(serde_json::Value::as_str) {
        Some("Promote") => Decision::Improved,
        Some("Reject") => Decision::NotImproved,
        _ => Decision::Inconclusive,
    };
    let result = EvalResult {
        schema: EVAL_SCHEMA.into(),
        plan: plan_digest(factory, section)?,
        protected: true,
        name: section.candidate.clone(),
        baseline: None,
        candidate: None,
        decision,
        statistics: serde_json::to_value(verified.aggregate())
            .map_err(|e| stage_error(&stage.key, e))?,
        signature: document
            .pointer("/signature/value")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
    };
    write_json(&staging.path().join(RESULT_FILE), &result)?;
    std::fs::write(staging.path().join("report.md"), report::report(&result))
        .map_err(io(staging.path()))
}
