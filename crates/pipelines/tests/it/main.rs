// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

#[path = "../common/mod.rs"]
mod common;
mod decision_fixtures;

mod acquire;
mod acquire_rounds;
mod advice_learn;
mod agent_audit;
mod ask_retrieving;
mod author;
mod batched_probing;
mod bench_harness;
mod bench_ingest;
mod bench_policy_gradient;
mod bench_request_records;
mod budget;
mod campaign;
mod campaign_factory;
mod canary_sets;
mod clusters;
mod compare_sets;
mod converse_learn;
mod coverage;
mod curriculum;
mod curriculum_quotas;
mod dataset_seal;
mod dataset_split;
mod default_budget;
mod doctor;
mod exam;
mod exam_set;
mod experience_path;
mod flywheel_tasks;
mod generation_trace;
mod generations;
mod goal_selection;
mod graded_items;
mod guided_tasks;
mod harbor_interop;
mod harness_opt;
mod improve;
mod judging;
mod learn;
mod learn_preflight;
mod lineage;
mod model_refs;
mod monitoring;
mod overlap_learn;
mod pipeline_seeds;
mod plan;
mod plan_learn;
mod powered;
mod practice;
mod practice_curriculum;
mod practice_teacher;
mod raft;
mod rehearsal;
mod release;
mod release_canary;
mod release_matrix;
mod release_predictive;
mod release_terms;
mod release_workspace;
mod rerank;
mod reserve;
mod reserve_learn;
mod router;
mod runtime;
mod sample_anchor;
mod sealed_everywhere;
mod solve_concurrency;
mod solve_resume;
mod subjects;
mod system_prompt;
mod task_pick;
mod teacher;
mod timeline_behaviour;
mod timeline_calibration;
mod timeline_certification;
mod timeline_properties;
mod timeline_release;
mod timeline_train;
mod train_regimes;
mod train_seed;
mod variants;
mod voice;
