// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose every answer and
// release traces back to the source bytes it was learned from, for its
// clients. If your team needs expertise in training-data lineage or model
// provenance, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `lineage`: from any artifact, where it came from and everything that
//! came from it.
//!
//! The graph is derived on every call from what the stores already record
//! ([`load`]); nothing about lineage is stored twice. An edge runs from an
//! artifact to one it was made from - `from` is derived from `to` - and its
//! [`Relation`] says how:
//!
//! | from | relation | to |
//! |---|---|---|
//! | content | `part_of` | the source holding it as a part |
//! | span | `span_of` | the source it names a part of, else the content it indexes |
//! | task | `evidence` | each span it is grounded in |
//! | task set, experience set | `member` | each task, experience |
//! | experience | `attempts` | its task |
//! | experience | `ran_in` | its environment snapshot |
//! | experience | `solved_by` | the model that solved it |
//! | experience | `critique_of`, `retry_of`, `revision_of`, `preferred_over`, `variant_of` | the experience its relation annotation names |
//! | verdict | `verdict_on` | the experience it grades |
//! | verdict | `produced_by` | the verifier that gave it |
//! | dataset | `projected_from` | each experience, task and source content its manifest names |
//! | candidate, release | `trained_on` | each new dataset |
//! | candidate, release | `trained_from` | the release it continued |
//! | candidate, release | `replayed` | its replay sample |
//! | replay sample | `sampled_from` | each earlier release it drew from |
//! | candidate, release | `adapter` | its adapter's digest |
//! | release | `release_of` | the candidate it was |
//! | answer | `answered_with` | the release its policy resolved to |
//! | answer | `answered_by` | the model that answered |
//! | answer | `open_book` | the source shown with the question |
//!
//! Walking up follows edges from an artifact; walking down follows them
//! backwards. Content addressing makes a cycle impossible - an artifact
//! names only what existed before it - but each walk keeps the artifacts it
//! reached and never expands one twice, so a corrupt store cannot loop it.
//!
//! The report ([`Lineage`]) serializes as `{"nodes": [{"id", "kind",
//! "label"}], "edges": [{"from", "to", "relation"}]}`: the artifact asked
//! about first, then everything reached in walk order, each once, and every
//! edge walked between them.

mod graph;
mod load;

use std::collections::BTreeSet;

use serde::Serialize;

use crate::context::Context;
use crate::error::CampaignError;
use graph::Graph;

/// What kind of artifact a node is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    /// A captured source.
    Source,
    /// The content of a source part, by digest.
    Content,
    /// A byte range of a content.
    Span,
    /// A generated task.
    Task,
    /// A named set of tasks.
    TaskSet,
    /// A solved task.
    Experience,
    /// A named set of experiences.
    ExperienceSet,
    /// An environment snapshot a solver worked in.
    Environment,
    /// A model, by its identity.
    Model,
    /// A verdict annotation on an experience.
    Verdict,
    /// The verifier that produced a verdict.
    Producer,
    /// A stored dataset.
    Dataset,
    /// A trained candidate.
    Candidate,
    /// The earlier records a candidate replayed.
    Replay,
    /// A release.
    Release,
    /// An adapter file, by digest.
    Adapter,
    /// A recorded answer.
    Answer,
}

impl NodeKind {
    /// The kind as it is serialized.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Content => "content",
            Self::Span => "span",
            Self::Task => "task",
            Self::TaskSet => "task_set",
            Self::Experience => "experience",
            Self::ExperienceSet => "experience_set",
            Self::Environment => "environment",
            Self::Model => "model",
            Self::Verdict => "verdict",
            Self::Producer => "producer",
            Self::Dataset => "dataset",
            Self::Candidate => "candidate",
            Self::Replay => "replay",
            Self::Release => "release",
            Self::Adapter => "adapter",
            Self::Answer => "answer",
        }
    }
}

/// How an artifact was derived from another; see the module documentation
/// for which kinds each relation joins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Relation {
    /// Content is a part of a source.
    PartOf,
    /// A span is a byte range of a source part, or of a content.
    SpanOf,
    /// A task is grounded in a span.
    Evidence,
    /// A set holds a task or an experience.
    Member,
    /// An experience attempts a task.
    Attempts,
    /// An experience was solved in an environment.
    RanIn,
    /// An experience was solved by a model.
    SolvedBy,
    /// An experience critiques another.
    CritiqueOf,
    /// An experience retries another.
    RetryOf,
    /// An experience revises another.
    RevisionOf,
    /// An experience is preferred over another.
    PreferredOver,
    /// An experience is a variant of another.
    VariantOf,
    /// A verdict grades an experience.
    VerdictOn,
    /// A verdict was given by a verifier.
    ProducedBy,
    /// A dataset's records came from an experience, a task or a content.
    ProjectedFrom,
    /// A candidate or a release was trained on a dataset.
    TrainedOn,
    /// A candidate or a release continued a release.
    TrainedFrom,
    /// A candidate or a release replayed a sample of earlier records.
    Replayed,
    /// A replay sample drew from an earlier release.
    SampledFrom,
    /// A candidate or a release has an adapter.
    Adapter,
    /// A release is a candidate, released.
    ReleaseOf,
    /// An answer was given by a release.
    AnsweredWith,
    /// An answer was given by a model.
    AnsweredBy,
    /// An answer was given with a source shown.
    OpenBook,
}

impl Relation {
    /// The relation as it is serialized.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PartOf => "part_of",
            Self::SpanOf => "span_of",
            Self::Evidence => "evidence",
            Self::Member => "member",
            Self::Attempts => "attempts",
            Self::RanIn => "ran_in",
            Self::SolvedBy => "solved_by",
            Self::CritiqueOf => "critique_of",
            Self::RetryOf => "retry_of",
            Self::RevisionOf => "revision_of",
            Self::PreferredOver => "preferred_over",
            Self::VariantOf => "variant_of",
            Self::VerdictOn => "verdict_on",
            Self::ProducedBy => "produced_by",
            Self::ProjectedFrom => "projected_from",
            Self::TrainedOn => "trained_on",
            Self::TrainedFrom => "trained_from",
            Self::Replayed => "replayed",
            Self::SampledFrom => "sampled_from",
            Self::Adapter => "adapter",
            Self::ReleaseOf => "release_of",
            Self::AnsweredWith => "answered_with",
            Self::AnsweredBy => "answered_by",
            Self::OpenBook => "open_book",
        }
    }
}

/// One artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Node {
    /// Its id: the content address for a stored artifact (`blake3:<hex>`),
    /// a candidate's id, or `<kind>:<...>` for what has no address of its
    /// own (a span, a model, a verdict, a verifier).
    pub id: String,
    /// What it is.
    pub kind: NodeKind,
    /// A line saying what it is, for a person.
    pub label: String,
}

/// `from` was derived from `to`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Edge {
    /// The derived artifact.
    pub from: String,
    /// What it was derived from.
    pub to: String,
    /// How.
    pub relation: Relation,
}

/// Which way to walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// Provenance: where it came from.
    Up,
    /// Influence: what came from it.
    Down,
    /// Both.
    #[default]
    Both,
}

/// One `lineage`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LineageRequest {
    /// Any artifact's id, its hex, or a prefix naming exactly one.
    pub id: String,
    /// Which way to walk.
    pub direction: Direction,
    /// The most edges walked from the artifact; `None` walks to the end.
    pub depth: Option<usize>,
}

/// One node of a walk, as a tree: the node reached, by which relation, and
/// what the walk reached from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Branch {
    /// The node.
    pub id: String,
    /// The relation it was reached by; `None` at the root.
    pub relation: Option<Relation>,
    /// What was reached from it.
    pub children: Vec<Branch>,
    /// Reached before on another path: not expanded again.
    pub repeat: bool,
    /// At the depth limit with more beyond it: not expanded.
    pub cut: bool,
}

/// What `lineage` reports; see the module documentation for its JSON.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Lineage {
    /// Every node reached, the artifact asked about first.
    pub nodes: Vec<Node>,
    /// Every edge walked.
    pub edges: Vec<Edge>,
    /// The walk up, as a tree; `None` when it was not asked for.
    #[serde(skip)]
    pub up: Option<Branch>,
    /// The walk down, as a tree; `None` when it was not asked for.
    #[serde(skip)]
    pub down: Option<Branch>,
}

impl Lineage {
    /// The node `id`, when the walk reached it.
    #[must_use]
    pub fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }
}

/// The lineage of the artifact `request.id` names.
pub fn lineage(ctx: &Context, request: &LineageRequest) -> Result<Lineage, CampaignError> {
    let graph = load::load(ctx)?;
    let root = graph.resolve(&request.id)?.id.clone();
    let mut walk = Walk {
        graph: &graph,
        depth: request.depth,
        nodes: Vec::new(),
        node_seen: BTreeSet::new(),
        edges: Vec::new(),
        edge_seen: BTreeSet::new(),
    };
    walk.reach(&root);
    let up = matches!(request.direction, Direction::Up | Direction::Both)
        .then(|| walk.grow(&root, None, Step::Up, 0, &mut BTreeSet::new()));
    let down = matches!(request.direction, Direction::Down | Direction::Both)
        .then(|| walk.grow(&root, None, Step::Down, 0, &mut BTreeSet::new()));
    Ok(Lineage {
        nodes: walk.nodes,
        edges: walk.edges,
        up,
        down,
    })
}

/// One direction of a walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Up,
    Down,
}

/// What a walk has reached so far, in the order it reached it.
struct Walk<'g> {
    graph: &'g Graph,
    depth: Option<usize>,
    nodes: Vec<Node>,
    node_seen: BTreeSet<String>,
    edges: Vec<Edge>,
    edge_seen: BTreeSet<Edge>,
}

impl Walk<'_> {
    fn reach(&mut self, id: &str) {
        if self.node_seen.insert(id.to_string()) {
            if let Some(node) = self.graph.node(id) {
                self.nodes.push(node.clone());
            }
        }
    }

    /// The tree from `id`, reached by `relation` at `level` edges from the
    /// root; `visited` holds what this direction's walk has expanded.
    fn grow(
        &mut self,
        id: &str,
        relation: Option<Relation>,
        step: Step,
        level: usize,
        visited: &mut BTreeSet<String>,
    ) -> Branch {
        self.reach(id);
        let mut branch = Branch {
            id: id.to_string(),
            relation,
            children: Vec::new(),
            repeat: false,
            cut: false,
        };
        if !visited.insert(id.to_string()) {
            branch.repeat = true;
            return branch;
        }
        let next = match step {
            Step::Up => self.graph.up(id),
            Step::Down => self.graph.down(id),
        };
        if self.depth.is_some_and(|depth| level >= depth) {
            branch.cut = !next.is_empty();
            return branch;
        }
        for (relation, other) in next {
            let edge = match step {
                Step::Up => Edge {
                    from: id.to_string(),
                    to: other.clone(),
                    relation,
                },
                Step::Down => Edge {
                    from: other.clone(),
                    to: id.to_string(),
                    relation,
                },
            };
            if self.edge_seen.insert(edge.clone()) {
                self.edges.push(edge);
            }
            let child = self.grow(&other, Some(relation), step, level + 1, visited);
            branch.children.push(child);
        }
        branch
    }
}
