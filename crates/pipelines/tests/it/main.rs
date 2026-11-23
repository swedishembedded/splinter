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
mod factory_index;
mod load_dataset_hub;

mod agent_audit;
mod ask_retrieving;
mod batched_probing;
mod bench_exact_contexts;
mod bench_harness;
mod bench_ingest;
mod bench_pairs;
mod bench_policy_gradient;
mod bench_request_records;
mod canary_sets;
mod capability_curriculum;
mod clusters;
mod compare_sets;
mod coverage;
mod curriculum;
mod dataset_curation;
mod dataset_seal;
mod dataset_split;
mod dataset_strata;
mod doctor;
mod exam;
mod exam_set;
mod experience_path;
mod generation_trace;
mod goal_selection;
mod graded_items;
mod guided_tasks;
mod harbor_interop;
mod hub_bundle;
mod improve;
mod judging;
mod lineage;
mod model_refs;
mod monitoring;
mod pipeline_seeds;
mod powered;
mod practice_episodes;
mod rehearsal;
mod release;
mod release_canary;
mod release_matrix;
mod release_predictive;
mod release_terms;
mod release_variants;
mod rerank;
mod reserve;
mod router;
mod runtime;
mod sample_anchor;
mod sealed_everywhere;
mod served;
mod served_factory;
mod set_import;
mod solve_concurrency;
mod solve_resume;
mod state_layout;
mod subjects;
mod system_prompt;
mod task_pick;
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
