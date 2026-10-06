// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that acquire a capability
// from a document or a tool and prove it with evidence. If your team needs
// expertise in agent infrastructure or small-model training loops, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The dataset stage of `learn`: the selected experiences as the dialogue
//! dataset - with retrieved passages, and abstentions among them, when the
//! run asks for them - and the writer's own text beside it.

use splinter_data::holdout::MIN_SAMPLES;
use splinter_orchestrator::error::io;
use splinter_orchestrator::pipeline::StageEnd;
use splinter_orchestrator::runs::Recorder;
use splinter_orchestrator::{Context, OrchestratorError};

use super::stages::{to_value, Done, LearnState};
use super::tokens_at_share;
use crate::abstain::ModelAbstainer;
use crate::datasets::{
    build_with, examples_in, supervised_tokens_in, BuildRequest, Built, Passages, ViewName,
    VoiceBuild,
};
use crate::describe::DescribeRequest;
use crate::raft::Abstainer;
use crate::retrieval::{library_of, Retrieval};

pub(super) fn dataset_stage(
    ctx: &Context,
    run: &mut Recorder<'_>,
    st: &mut LearnState<'_>,
) -> Done {
    let Some(selected) = &st.report.select else {
        unreachable!("the dataset stage follows the select stage")
    };
    let request = BuildRequest {
        sets: vec![selected.experience_set.clone()],
        view: ViewName::SftFinal,
        strip: None,
        min_strength: Some(st.min_strength),
        system_prompt: st.system_prompt(),
        export_only: false,
        limit: None,
        voice: VoiceBuild::default(),
    };
    // Passages of the run's own sources, when a share of the records is to
    // carry them: the embedding model is loaded for it and the index kept.
    let embedder;
    let library;
    let retrieval;
    let abstainer;
    let passages = match st.learn.passages {
        Some(share) => {
            embedder = ctx.embedder()?;
            let ids: Vec<String> = st.source_ids.iter().map(ToString::to_string).collect();
            library = library_of(ctx, &ids, &*embedder)?.1;
            retrieval = Retrieval {
                library: &library,
                embedder: &*embedder,
                passages: PASSAGES_SHOWN,
                rerank: None,
            };
            // The teacher, as the persona, writes the abstentions.
            abstainer = share
                .abstains()
                .then(|| -> Result<_, OrchestratorError> {
                    Ok(ModelAbstainer::new(
                        ctx,
                        ctx.model(st.learn.teacher)?,
                        st.system_prompt().unwrap_or_default(),
                        st.stage_deadlines.teach,
                        run.cancel_token(),
                    ))
                })
                .transpose()?;
            Some(Passages {
                retrieval: &retrieval,
                share,
                abstainer: abstainer.as_ref().map(|a| a as &dyn Abstainer),
            })
        }
        None => None,
    };
    let built = match build_with(ctx, &request, passages.as_ref()) {
        Ok(built) => built,
        Err(OrchestratorError::View(splinter_data::ViewError::Empty)) => {
            return Ok(StageEnd::halt(
                "no experience passed verification, so there is nothing to train on",
            ));
        }
        Err(e) => return Err(e),
    };
    st.datasets = vec![built.dataset.to_string()];
    st.records = built.records;
    st.examples = examples_in(&built.path).map_err(io(&built.path))?;
    // The writer's own text beside the dialogues: tokens enough to make it the
    // share asked, spread over all the sources, written as the persona's.
    let budget = tokens_at_share(
        supervised_tokens_in(ctx, &built.path).map_err(io(&built.path))?,
        st.voice_share(),
    );
    let voice = if budget > 0 {
        let voice = build_with(
            ctx,
            &BuildRequest {
                view: ViewName::Voice,
                min_strength: None,
                system_prompt: None,
                voice: VoiceBuild {
                    writer: st.persona().map(str::to_string),
                    token_budget: Some(budget),
                    describe: st.learn.describe_voice.then(|| DescribeRequest {
                        generator: st.learn.generator.clone(),
                        deadline: st.stage_deadlines.teach,
                        cancel: run.cancel_token(),
                    }),
                    ..VoiceBuild::default()
                },
                ..request
            },
            None,
        )?;
        st.datasets.push(voice.dataset.to_string());
        st.examples += examples_in(&voice.path).map_err(io(&voice.path))?;
        Some(voice)
    } else {
        None
    };
    let summary = to_value(&DatasetStage {
        built: &built,
        voice: voice.as_ref(),
    })?;
    st.report.dataset = Some(built);
    st.report.voice = voice;
    Ok(if st.records < MIN_SAMPLES {
        StageEnd::stop(
            summary,
            format!(
                "{} record(s) passed; training holds records out for scoring and needs at \
                 least {MIN_SAMPLES}",
                st.records
            ),
        )
    } else {
        StageEnd::done(summary)
    })
}

/// What the dataset stage records: the dialogue dataset, and the writer's
/// own text beside it when the run has it.
#[derive(serde::Serialize)]
struct DatasetStage<'a> {
    #[serde(flatten)]
    built: &'a Built,
    voice: Option<&'a Built>,
}

/// How many passages a training record carries when it is given any.
const PASSAGES_SHOWN: usize = 4;
