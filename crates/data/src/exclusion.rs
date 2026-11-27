// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Why a view left a candidate out.

use serde::{Deserialize, Serialize};

/// Why a view left a candidate out. A view counts each candidate it
/// considers - an experience, a step, a relation, a pair, a source part,
/// as its documentation says - under the first reason that applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Exclusion {
    /// No pass/fail decision: no verdict, or the strongest ones conflict.
    Undecided,
    /// Decided below the view's minimum strength.
    TooWeak,
    /// Decided fail where the view needs a pass; or, for any view, the
    /// attempt broke a constraint (an exposure audit's), which no view admits
    /// whatever else was decided about it.
    Failed,
    /// No final output where the view needs one.
    NoFinalOutput,
    /// The instruction refers to privileged context the view would drop.
    NotSelfContained,
    /// A step labelled bad.
    BadStep,
    /// An action that repeats a call of the same trajectory whose result was
    /// the same: the agent learned nothing from it, so it is context and not
    /// an action to learn (the tail of a loop).
    RepeatedAction,
    /// Left out on request because it would carry an exam family's attempt or
    /// file into the training data (`dataset build --drop-leaking`).
    ExamLeak,
    /// The trajectory holds something a chat record cannot represent (an
    /// image, a second user turn, an observation answering no call).
    Unrepresentable,
    /// No action to supervise or reward.
    NoAction,
    /// An action whose tool calls are not the ones a view was asked to keep
    /// (`SftStep::only_actions`): the exploration around the decisive actions,
    /// left out on request rather than because it could not be trained.
    ActionNotWanted,
    /// A harness conversation answered by a model role the view was not asked
    /// for (`SftStep::only_roles`), or whose role is unknown: a role is
    /// trained on its own conversations, and a conversation of no known role
    /// is never guessed into one.
    RoleNotWanted,
    /// A harness conversation shown images, built where the base is not known
    /// to read them (`SftStep::images`).
    ShownImages,
    /// A harness conversation shown images, built where images are wanted,
    /// that recorded no requests (ingested before they were kept): the
    /// pictures are not in the experience. Ingest the trial again.
    RequestsNotRecorded,
    /// An action whose context holds more images than a record may
    /// ([`crate::images::MAX_IMAGES_PER_RECORD`]).
    TooManyImages,
    /// No reward could be derived.
    NoReward,
    /// An attempt in a group with nothing to compare it with: the only one of
    /// its task, or all of the group earned the same reward (or, keeping only
    /// the attempts above the group's mean, fewer than two that differ), so
    /// no attempt did better or worse than another.
    NoGroupSignal,
    /// An action that carries no tokens the policy sampled (its trace is
    /// missing or unusable, the scrub changed its conversation, or its context
    /// was rebuilt from the trajectory), left out of a build that trains only
    /// sampled tokens: its re-rendered text is weighed against the start of the
    /// run, not the behaviour policy, and its importance ratio is unbounded.
    ProxyTokens,
    /// The verdict does not record which attempt (trial) of which task it
    /// grades, so the experience belongs to no group.
    NoRollout,
    /// A self-distillation view found nothing the verifier knows to show the
    /// teacher: no reference change, no hidden check and no record of what
    /// failed.
    NoPrivilegedContext,
    /// A critique whose own decision is not a pass at the minimum strength.
    UnverifiedCritique,
    /// The experience a relation names is not in the corpus.
    RelatedMissing,
    /// A relation between experiences of different tasks.
    DifferentTask,
    /// Both sides of a pair gave the same output.
    IdenticalOutputs,
    /// A retry or revision chain without a pass on one side and a fail at
    /// the minimum strength on the other.
    NoPreferredPath,
    /// The two attempts part at no action both took a different one at.
    NoDivergence,
    /// No evidence span.
    NoEvidence,
    /// A source, part or content the store does not hold.
    SourceMissing,
    /// Content that is not UTF-8 text.
    NotText,
    /// Empty text.
    Empty,
    /// Content, a task or a pair already projected.
    Duplicate,
    /// A task with no single reference answer.
    NoReference,
    /// A record past the limit a projection was thinned to.
    OverLimit,
    /// Text too short to be a record: a heading, a signature, a stub.
    TooShort,
    /// An assistant turn holds reasoning markup (`<think>`), which a chat
    /// record does not carry: training on it would teach the template's
    /// reasoning block as answer text.
    ReasoningInText,
    /// The capture marks the run incomplete (a journal never closed): a
    /// prefix of a run, never a success.
    Incomplete,
    /// The agent's context was compacted during the run, so the turns
    /// before the compaction are not what it saw when it acted after it.
    Compacted,
    /// A tool call whose arguments the model sent as text that is not
    /// JSON; the trajectory keeps `{}`, which is not what the model said.
    UnparsedArguments,
    /// Two tool calls of one trajectory share an id, so results cannot be
    /// paired to their calls: every call would receive every result, with
    /// duplicated, reordered and future results in its context.
    DuplicateCallId,
    /// The projection did not keep every call and result of its source: a
    /// defect of the projection, never a property of the data.
    NotConserved,
    /// A record longer than the training block it was projected for.
    OverBlock,
    /// A record the trainer of the base the dataset is built for refuses
    /// (brain's own check: its chat template, its record contract), so a run
    /// on the dataset never meets it.
    TrainerRefused,
    /// A window of a record split to fit the training block that holds no
    /// supervised turn (the exploration around the decisive actions of an
    /// `SftStep::only_actions` view): training on it would learn nothing.
    NothingSupervised,
    /// An episode in a workspace, which the policy never answers
    /// closed-book: its final answer is not a record of the view that
    /// trains final answers.
    WorkspaceEpisode,
    /// An action whose tool call reads the repository's history or git's
    /// raw storage ([`crate::curation`]): a student imitating it learns to dig for
    /// what the checked-out tree does not show.
    RepositoryHistory,
    /// An action after one that read the repository's history, in a
    /// demonstration the curation truncates there ([`crate::curation`]).
    AfterRepositoryHistory,
    /// An attempt the infrastructure failed (the examiner's score record says
    /// the sandbox, the model server or the agent failed): it has no verdict,
    /// and it is never a rejected or undesirable sample.
    InfrastructureFailure,
    /// An attempt decided pass below the view's minimum strength: never a
    /// chosen or desirable sample.
    UnverifiedPass,
    /// A verified attempt whose group holds no verified attempt of the other
    /// verdict to pair it with (or, for a repair, no failed attempt of its
    /// task).
    NoCounterpart,
    /// Two attempts that were not shown the same prompt (the messages before
    /// the first action, or the tool list), so neither continues the other's.
    DifferentPrompt,
    /// Steps of one attempt, or the two sides of a pair, name different
    /// tokenizers.
    MixedTokenizerDigest,
    /// A repair whose copied prefix is not the failed attempt's own
    /// conversation, so the two do not continue one state.
    PrefixMismatch,
    /// A pass whose last agent step calls a tool or says nothing, left out by
    /// a curation that wants a final answer ([`crate::curation::PassCuration`]).
    NoFinalAnswer,
    /// A pass that made the same tool calls as another pass of its task with
    /// less reasoning ([`crate::curation::PassCuration`]).
    DuplicatePass,
    /// A pass beyond the curation's cap of passes per task
    /// ([`crate::curation::PassCuration`]).
    OverTaskCap,
    /// An experience a person guided: a user step of its trajectory was
    /// typed by a person during the run (`extra.sven.origin` `human`), so
    /// the model produced its later tokens with that message in context. A
    /// student never sees it, so they are never trained on (same
    /// observation); the experience is a source of verified states only.
    HumanContext,
    /// An attempt whose solver was prompted open-book (a hint or knowledge
    /// page beside the instruction, `provenance.teacher`): what the
    /// environment answered followed actions taken under context a student
    /// never sees.
    OpenBook,
    /// An observation whose task, call and answer the view has already
    /// supervised as often as its cap allows: it stays context.
    RepeatedTransition,
    /// An agent step none of the recorded requests answered (no answer makes
    /// its tool calls or says its text), in a single agent's requests that do
    /// not extend one another: it has no context its model was sent, so it is
    /// no record; when no step of the attempt pairs with a request, the
    /// attempt, which keeps only what a view keeps of an unrenderable one (a
    /// policy-gradient baseline). The trajectory is never rendered in its
    /// place: its context is one the model never saw.
    RequestsUnpaired,
}
