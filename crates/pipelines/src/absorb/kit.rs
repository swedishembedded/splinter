// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning from the sessions a person holds
// with an agent, for its clients. If your team needs expertise in continual
// learning from user feedback, you can procure our services by sending an
// email to info@swedishembedded.com.

//! What a live claim is taught from: its kit.
//!
//! A fact shown once, in one wording, is memorised in that wording. A claim
//! is therefore taught from about seventeen distinct records, written once
//! and kept, so that every night's retraining from the base has all of them:
//!
//! * the claim's question, and eight differently worded questions about the
//!   same fact ([`crate::variants`]), answered by a teacher that is shown the
//!   claim and speaks as the policy, each answer kept only when the claim's
//!   verifiers pass it;
//! * what the person first asked, before the agent was corrected, answered
//!   with that verified corrected answer (the hindsight dialogue: the reply
//!   the agent really gave, and the person's correction, are never trained
//!   on, since supervising "you are right" with the correction in view
//!   teaches capitulation);
//! * four statements, two questions that ask for the subject given what is
//!   said of it, and two consequences ([`splinter_knowledge::claims::forms`]);
//! * four further paraphrases no record contains: the claim's own stopping
//!   and gate set, chosen from the twelve written by a stable hash.
//!
//! A claim of kind procedure is stored and never trained on.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use splinter_agent::forms::FormsWriter;
use splinter_agent::CancelToken;
use splinter_core::annotation::Strength;
use splinter_core::chat::WireMessage;
use splinter_core::claim::{Claim, ClaimId, ClaimKind, ClaimTaskLink, TaskRole};
use splinter_core::digest::Digest;
use splinter_core::model_ref::ModelRef;
use splinter_core::prompt::SYSTEM_PROMPT;
use splinter_core::selfcontained::check_self_contained;
use splinter_data::{Objective, Record, RecordBody, RecordMetadata, Side};
use splinter_knowledge::claims::forms::Forms;
use splinter_knowledge::claims::task::{claim_task, subject_of};
use splinter_knowledge::session::{SessionView, Speaker};
use splinter_store::tasks::{TaskEntry, TaskSet};

use crate::curriculum::teacher::{teach, TeachRequest};
use crate::datasets::{project_with, BuildRequest, ViewName, VoiceBuild};
use crate::variants::{generate_variants, VariantsRequest};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The view the records of an absorb run are written under.
pub const VIEW: &str = "session-claims";

/// Differently worded questions written per claim.
pub const PARAPHRASES_WRITTEN: usize = 12;

/// Of those, the ones kept out of training for the claim's own gate.
pub const STOPPING_PARAPHRASES: usize = 4;

const KIT: &str = "absorb_claim_kit";

/// A live claim.
pub type Live = (ClaimId, Claim);

/// Everything a claim is taught from, and what it is measured by.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClaimKit {
    /// The claim.
    pub claim: ClaimId,
    /// The records it is trained on, each fixed to the training side.
    pub records: Vec<Record>,
    /// The paraphrases no record contains, by task address.
    pub stopping: Vec<Digest>,
}

/// The kits kept, by claim; a claim's latest kit.
pub fn kept(ctx: &Context) -> Result<BTreeMap<ClaimId, ClaimKit>, OrchestratorError> {
    let ws = ctx.workspace();
    let mut kits = BTreeMap::new();
    for id in ws.documents_in_order(KIT)? {
        if let Some(kit) = ws.get_document::<ClaimKit>(KIT, &id)? {
            kits.insert(kit.claim.clone(), kit);
        }
    }
    Ok(kits)
}

/// A claim that could not be made into tasks or taught, and why.
#[derive(Clone, Debug, Serialize)]
pub struct Untaught {
    /// The claim.
    pub claim: ClaimId,
    /// Why.
    pub reason: String,
}

/// What building kits reports.
#[derive(Clone, Debug, Default, Serialize)]
pub struct KitsBuilt {
    /// Kits made now.
    pub built: usize,
    /// Kits already kept, reused.
    pub reused: usize,
    /// Claims left out, with why.
    pub untaught: Vec<Untaught>,
    /// Paraphrases written and admitted now, per claim.
    pub paraphrases: BTreeMap<String, usize>,
    /// Paraphrase proposals refused, by reason.
    pub paraphrases_refused: BTreeMap<String, usize>,
    /// Teacher answers that passed verification.
    pub verified: usize,
    /// Teacher answers that did not.
    pub failed: usize,
    /// Forms admitted: statements, reverse questions, consequences.
    pub forms: (usize, usize, usize),
    /// Forms refused by the term rule.
    pub forms_refused: usize,
    /// Claims with no stopping paraphrase to be measured by.
    pub unmeasurable: Vec<ClaimId>,
}

/// One kit-building request.
pub struct KitRequest<'a> {
    /// The live claims to teach.
    pub claims: &'a [Live],
    /// The model that writes paraphrases and forms.
    pub generator: &'a ModelRef,
    /// The model that answers with the claim in front of it.
    pub teacher: &'a ModelRef,
    /// The policy's system prompt, which the teacher answers under.
    pub system: Option<&'a str>,
    /// The weakest verdict a teacher's answer may stand on.
    pub min_strength: Strength,
    /// Paraphrases to write per claim.
    pub paraphrases: usize,
    /// No model request starts after this.
    pub deadline: Option<Instant>,
    /// Stops the stage.
    pub cancel: CancelToken,
}

/// The stable order the paraphrases of a claim are split in: by the hash of
/// the claim and the task, so a rerun splits them the same way.
fn draw(claim: &ClaimId, task: &Digest) -> String {
    Digest::of(format!("{claim}|{task}").as_bytes()).to_string()
}

/// How many of `written` paraphrases are kept out of training.
fn stopping_of(written: usize) -> usize {
    if written >= PARAPHRASES_WRITTEN {
        STOPPING_PARAPHRASES
    } else {
        written.div_ceil(3)
    }
}

/// Builds and keeps the kit of every claim of `request.claims` that has
/// none.
pub fn build_kits(ctx: &Context, request: &KitRequest<'_>) -> Result<KitsBuilt, OrchestratorError> {
    let mut report = KitsBuilt::default();
    let have = kept(ctx)?;
    let mut todo: Vec<&Live> = Vec::new();
    for live in request.claims {
        if have.contains_key(&live.0) {
            report.reused += 1;
        } else if live.1.kind == ClaimKind::Procedure {
            report.untaught.push(Untaught {
                claim: live.0.clone(),
                reason: "a procedure is stored and not trained on: one trajectory is not \
                         trainable data"
                    .into(),
            });
        } else {
            todo.push(live);
        }
    }
    if todo.is_empty() {
        return Ok(report);
    }
    let questions = question_tasks(ctx, &todo, &mut report)?;
    let todo: Vec<&Live> = todo
        .into_iter()
        .filter(|(id, _)| questions.contains_key(id))
        .collect();
    paraphrase(ctx, request, &todo, &questions, &mut report)?;
    hindsight(ctx, &todo, &questions)?;
    let records = teach_records(ctx, request, &todo, &mut report)?;
    let writer =
        FormsWriter::new(ctx.model(request.generator)?).with_cancel(request.cancel.clone());
    let links = ctx.claims().task_links()?;
    for (id, claim) in todo {
        if request.cancel.is_cancelled() {
            return Err(OrchestratorError::Cancelled);
        }
        let mut kit_records = records.get(id).cloned().unwrap_or_default();
        match ctx.block_on(writer.write(claim)) {
            Ok(forms) => {
                report.forms.0 += forms.statements.len();
                report.forms.1 += forms.reverse.len();
                report.forms.2 += forms.implications.len();
                report.forms_refused += forms.refused.len();
                kit_records.extend(form_records(claim, &forms));
            }
            Err(why) => report.untaught.push(Untaught {
                claim: id.clone(),
                reason: format!("its other forms could not be written: {why}"),
            }),
        }
        if kit_records.is_empty() {
            report.untaught.push(Untaught {
                claim: id.clone(),
                reason: "no answer or form passed, so nothing teaches it".into(),
            });
            continue;
        }
        let stopping: Vec<Digest> = links
            .iter()
            .filter(|l| l.claim == *id && l.role == TaskRole::Stopping)
            .map(|l| l.task.clone())
            .collect();
        if stopping.is_empty() {
            report.unmeasurable.push(id.clone());
        }
        ctx.workspace().put_document(
            KIT,
            &ClaimKit {
                claim: id.clone(),
                records: kit_records,
                stopping,
            },
        )?;
        report.built += 1;
    }
    Ok(report)
}

/// The question task of every claim, stored and linked; a claim whose
/// question cannot be a task is left out with why.
fn question_tasks(
    ctx: &Context,
    todo: &[&Live],
    report: &mut KitsBuilt,
) -> Result<BTreeMap<ClaimId, Digest>, OrchestratorError> {
    let mut tasks = BTreeMap::new();
    let mut links = Vec::new();
    for (id, claim) in todo {
        match claim_task(&ctx.sources(), claim) {
            Ok(made) => {
                let task = ctx.tasks().put(&made.task)?;
                links.push(ClaimTaskLink {
                    claim: id.clone(),
                    task: task.clone(),
                    role: TaskRole::Question,
                });
                tasks.insert(id.clone(), (task, made.subject));
            }
            Err(e) => report.untaught.push(Untaught {
                claim: id.clone(),
                reason: e.to_string(),
            }),
        }
    }
    ctx.claims().link_tasks(&links)?;
    // The subjects travel with the tasks in a set the paraphrase stage reads.
    let members = tasks
        .values()
        .map(|(task, subject)| TaskEntry {
            task: task.clone(),
            generator: None,
            prompt: None,
            variant_of: None,
            subject: subject.clone(),
        })
        .collect();
    ctx.tasks().put_set(&TaskSet {
        name: "claim questions".into(),
        members,
    })?;
    Ok(tasks
        .into_iter()
        .map(|(id, (task, _))| (id, task))
        .collect())
}

/// Writes the paraphrases of every claim that has none, splits each claim's
/// by the stable hash into the ones trained on and the stopping set, and
/// records the roles.
fn paraphrase(
    ctx: &Context,
    request: &KitRequest<'_>,
    todo: &[&Live],
    questions: &BTreeMap<ClaimId, Digest>,
    report: &mut KitsBuilt,
) -> Result<(), OrchestratorError> {
    let links = ctx.claims().task_links()?;
    let has_paraphrases = |id: &ClaimId| {
        links
            .iter()
            .any(|l| l.claim == *id && matches!(l.role, TaskRole::Train | TaskRole::Stopping))
    };
    let mut members = Vec::new();
    for (id, claim) in todo {
        if has_paraphrases(id) {
            continue;
        }
        let made = claim_task(&ctx.sources(), claim)
            .map_err(|e| OrchestratorError::Refused(e.to_string()))?;
        members.push(TaskEntry {
            task: questions[id].clone(),
            generator: None,
            prompt: None,
            variant_of: None,
            subject: made.subject,
        });
    }
    if members.is_empty() {
        return Ok(());
    }
    let set = ctx.tasks().put_set(&TaskSet {
        name: "claims to paraphrase".into(),
        members,
    })?;
    let written = generate_variants(
        ctx,
        &VariantsRequest {
            task_set: &set,
            generator: request.generator,
            per_task: request.paraphrases,
            deadline: request.deadline,
            cancel: request.cancel.clone(),
        },
    )?;
    for (reason, count) in written.rejected {
        *report.paraphrases_refused.entry(reason).or_default() += count;
    }
    for (reason, count) in written.ineligible {
        *report
            .paraphrases_refused
            .entry(format!("ineligible: {reason}"))
            .or_default() += count;
    }
    let Some(variant_set) = written.variant_set else {
        return Ok(());
    };
    let by_question: HashMap<&Digest, &ClaimId> = questions.iter().map(|(c, t)| (t, c)).collect();
    let mut per_claim: BTreeMap<ClaimId, Vec<Digest>> = BTreeMap::new();
    for entry in ctx.tasks().get_set(&variant_set)?.members {
        if let Some(claim) = entry.variant_of.as_ref().and_then(|q| by_question.get(q)) {
            per_claim
                .entry((*claim).clone())
                .or_default()
                .push(entry.task);
        }
    }
    let mut roles = Vec::new();
    for (claim, mut tasks) in per_claim {
        tasks.sort_by_key(|t| draw(&claim, t));
        let stopping = stopping_of(tasks.len());
        report.paraphrases.insert(claim.to_string(), tasks.len());
        for (position, task) in tasks.into_iter().enumerate() {
            roles.push(ClaimTaskLink {
                claim: claim.clone(),
                task,
                role: if position < stopping {
                    TaskRole::Stopping
                } else {
                    TaskRole::Train
                },
            });
        }
    }
    ctx.claims().link_tasks(&roles)?;
    Ok(())
}

/// What the person first asked, as a task answered by the verified
/// corrected answer: for a correction, when the first user step is a
/// question of its own.
fn hindsight(
    ctx: &Context,
    todo: &[&Live],
    questions: &BTreeMap<ClaimId, Digest>,
) -> Result<(), OrchestratorError> {
    let mut links = Vec::new();
    for (id, claim) in todo {
        if claim.said_wrong.is_none() {
            continue;
        }
        let Ok(view) = SessionView::load(&ctx.sources(), &claim.session) else {
            continue;
        };
        let first = view
            .steps()
            .find(|s| s.speaker == Speaker::User)
            .and_then(|s| s.message.as_ref())
            .map(|m| m.text.trim().to_string());
        let Some(first) = first else { continue };
        if first.is_empty()
            || first.chars().count() > 400
            || splinter_knowledge::gates::normalize(&first)
                == splinter_knowledge::gates::normalize(&claim.question)
        {
            continue;
        }
        let question = ctx.tasks().get(&questions[id])?;
        let statement = splinter_core::experience::Privileged {
            kind: splinter_core::experience::PrivilegedKind::Passage,
            content: claim.statement.clone(),
            span: None,
        };
        if check_self_contained(&first, &[&statement]).is_err() {
            continue;
        }
        let task = ctx.tasks().put(&question.with_instruction(first)?)?;
        links.push(ClaimTaskLink {
            claim: id.clone(),
            task,
            role: TaskRole::Hindsight,
        });
    }
    ctx.claims().link_tasks(&links)?;
    Ok(())
}

/// The teacher's verified answers to the question, the trained paraphrases
/// and the hindsight question of every claim, as training records by claim.
fn teach_records(
    ctx: &Context,
    request: &KitRequest<'_>,
    todo: &[&Live],
    report: &mut KitsBuilt,
) -> Result<HashMap<ClaimId, Vec<Record>>, OrchestratorError> {
    let wanted: BTreeSet<&ClaimId> = todo.iter().map(|(id, _)| id).collect();
    let mut owner: HashMap<Digest, ClaimId> = HashMap::new();
    let mut members = Vec::new();
    for link in ctx.claims().task_links()? {
        let trained = matches!(
            link.role,
            TaskRole::Question | TaskRole::Train | TaskRole::Hindsight
        );
        if trained && wanted.contains(&link.claim) && !owner.contains_key(&link.task) {
            members.push(TaskEntry {
                task: link.task.clone(),
                generator: None,
                prompt: None,
                variant_of: None,
                subject: None,
            });
            owner.insert(link.task, link.claim);
        }
    }
    if members.is_empty() {
        return Ok(HashMap::new());
    }
    // Listed as originals, the trained paraphrases are no variants: a
    // variant is measured and never trained on.
    let set = ctx.tasks().put_set(&TaskSet {
        name: "claims to teach".into(),
        members,
    })?;
    let taught = teach(
        ctx,
        &TeachRequest {
            task_set: &set,
            attempts: None,
            teacher: request.teacher,
            system: request.system,
            deadline: request.deadline,
            cancel: request.cancel.clone(),
        },
    )?;
    report.verified += taught.verify.passed;
    report.failed += taught.verify.failed;
    let (projection, _) = project_with(
        ctx,
        &BuildRequest {
            sets: vec![taught.solve.experience_set],
            view: ViewName::SftFinal,
            strip: None,
            min_strength: Some(request.min_strength),
            system_prompt: request.system.map(str::to_string),
            export_only: false,
            limit: None,
            voice: VoiceBuild::default(),
        },
        None,
    )?;
    let mut records: HashMap<ClaimId, Vec<Record>> = HashMap::new();
    for mut record in projection.records {
        let Some(first) = record.metadata.experiences.first() else {
            continue;
        };
        let task = ctx.experiences().get(first)?.to_task().task.id;
        let Some(claim) = owner.get(&task) else {
            continue;
        };
        record.metadata.task = Some(task);
        record.metadata.split = Some(Side::Train);
        record.metadata.view = VIEW.into();
        records.entry(claim.clone()).or_default().push(record);
    }
    Ok(records)
}

fn chat(claim: &Claim, user: &str, assistant: &str) -> Record {
    let message = |role: &str, content: &str, train: bool| WireMessage {
        role: role.into(),
        content: content.into(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        train,
    };
    Record {
        body: RecordBody::Chat {
            messages: vec![
                message("system", SYSTEM_PROMPT, false),
                message("user", user, false),
                message("assistant", assistant, true),
            ],
        },
        metadata: RecordMetadata {
            experiences: Vec::new(),
            task: None,
            sources: claim.quotes.iter().map(|q| q.span.source.clone()).collect(),
            group: None,
            split: Some(Side::Train),
            view: VIEW.into(),
            objective: Objective::Sft,
        },
    }
}

/// The records the admitted forms of `claim` are: restatements asked for by
/// subject, the reverse questions and the consequences.
fn form_records(claim: &Claim, forms: &Forms) -> Vec<Record> {
    let ask = match subject_of(&claim.question, &claim.statement) {
        Some(subject) => format!("What do you know about {subject}?"),
        None => "What did I tell you?".to_string(),
    };
    let statements = forms.statements.iter().map(|s| chat(claim, &ask, s));
    let pairs = forms.reverse.iter().chain(&forms.implications);
    statements
        .chain(pairs.map(|qa| chat(claim, &qa.question, &qa.answer)))
        .collect()
}
