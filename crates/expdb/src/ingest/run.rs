// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A run in progress: observe, decide, transition, finish.

use super::collector::Collector;
use crate::blob::BlobRef;
use crate::error::Result;
use crate::id::{ContentId, RecordId};
use crate::model::{
    Action, Body, Content, Decision, Edge, Evaluation, Observation, Outcome, PolicyRef, Record,
    Rel, State, Transition,
};

/// What a later fork or analysis needs to know about a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecisionRef {
    /// The decision's record id.
    pub id: RecordId,
    /// The world it was made in.
    pub state: ContentId,
    /// The record before it on its path.
    pub parent: Option<RecordId>,
    /// The attempt it belongs to.
    pub attempt: RecordId,
    /// The family it belongs to.
    pub family: ContentId,
    /// The task instance it belongs to.
    pub task_instance: ContentId,
}

/// One attempt being recorded. Records are added to the collector's buffer
/// as they happen; [`Collector::flush`] makes them durable.
pub struct Run<'a> {
    collector: &'a mut Collector,
    attempt: RecordId,
    family: ContentId,
    task_instance: ContentId,
    state: ContentId,
    head: RecordId,
    policy: PolicyRef,
    fork: Option<(RecordId, RecordId)>,
}

impl<'a> Run<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        collector: &'a mut Collector,
        attempt: RecordId,
        family: ContentId,
        task_instance: ContentId,
        state: ContentId,
        head: RecordId,
        policy: PolicyRef,
        fork: Option<(RecordId, RecordId)>,
    ) -> Self {
        Self {
            collector,
            attempt,
            family,
            task_instance,
            state,
            head,
            policy,
            fork,
        }
    }

    /// The attempt's record id.
    pub fn attempt(&self) -> RecordId {
        self.attempt
    }

    /// The family the attempt belongs to.
    pub fn family(&self) -> ContentId {
        self.family
    }

    /// The world the run is currently in.
    pub fn state(&self) -> ContentId {
        self.state
    }

    fn push(&mut self, body: Body, parent: Option<RecordId>) -> Result<RecordId> {
        let mut record =
            self.collector
                .make(body)
                .in_attempt(self.attempt, self.family, self.task_instance);
        record.parent = parent;
        self.collector.writer.push(record)
    }

    /// Records what the agent is shown of the current world.
    pub fn observe(&mut self, content: Content) -> Result<RecordId> {
        let id = self.push(
            Body::Observation(Observation {
                state: self.state,
                content,
            }),
            Some(self.head),
        )?;
        self.head = id;
        Ok(id)
    }

    /// Starts recording a decision in the current world.
    pub fn decision(&mut self) -> DecisionBuilder<'_, 'a> {
        DecisionBuilder {
            policy: self.policy.clone(),
            run: self,
            observation: None,
            context: None,
            logprob: None,
            value: None,
        }
    }

    /// Records what a decision caused, and moves the run into the world it
    /// caused. The next decision follows this transition.
    pub fn transition(
        &mut self,
        decision: &DecisionRef,
        next: &State,
        reward: Option<f64>,
        observation: Option<RecordId>,
    ) -> Result<RecordId> {
        let to = next.id()?;
        self.collector.entity(Body::State(next.clone()))?;
        let body = Body::Transition(Transition {
            decision: decision.id,
            from: decision.state,
            to,
            observation,
            reward,
            done: false,
        });
        let id = self.push(body, Some(decision.id))?;
        self.state = to;
        self.head = id;
        Ok(id)
    }

    /// Records an evaluation of something in this attempt.
    pub fn evaluate(&mut self, evaluation: Evaluation) -> Result<RecordId> {
        self.push(Body::Evaluation(evaluation), None)
    }

    /// Ends the attempt with how the environment says it went.
    pub fn finish(mut self, outcome: Outcome) -> Result<RecordId> {
        let attempt = self.attempt;
        self.push(Body::AttemptEnd { attempt, outcome }, Some(self.head))
    }
}

/// Describes a decision before it is committed.
pub struct DecisionBuilder<'r, 'a> {
    run: &'r mut Run<'a>,
    policy: PolicyRef,
    observation: Option<RecordId>,
    context: Option<BlobRef>,
    logprob: Option<f64>,
    value: Option<f64>,
}

impl DecisionBuilder<'_, '_> {
    /// What the agent had been shown.
    pub fn observation(mut self, observation: RecordId) -> Self {
        self.observation = Some(observation);
        self
    }

    /// The policy that chose, when it is not the run's own.
    pub fn policy(mut self, policy: PolicyRef) -> Self {
        self.policy = policy;
        self
    }

    /// The context chosen from, stored apart so sibling decisions share it.
    pub fn context(mut self, context: BlobRef) -> Self {
        self.context = Some(context);
        self
    }

    /// The log-probability the policy gave its choice.
    pub fn logprob(mut self, logprob: f64) -> Self {
        self.logprob = Some(logprob);
        self
    }

    /// The value the policy estimated.
    pub fn value_estimate(mut self, value: f64) -> Self {
        self.value = Some(value);
        self
    }

    /// Records the decision.
    pub fn commit(self, action: Action) -> Result<DecisionRef> {
        let run = self.run;
        let parent = Some(run.head);
        let body = Body::Decision(Decision {
            state: run.state,
            observation: self.observation,
            policy: self.policy,
            context: self.context,
            action,
            old_logprob: self.logprob,
            value_estimate: self.value,
        });
        let id = run.push(body, parent)?;
        if let Some((set, origin)) = run.fork.take() {
            run.collector
                .writer
                .link(Edge::new(set, Rel::Alternative, id))?;
            run.collector
                .writer
                .link(Edge::new(id, Rel::ForkedFrom, origin))?;
        }
        run.head = id;
        Ok(DecisionRef {
            id,
            state: run.state,
            parent,
            attempt: run.attempt,
            family: run.family,
            task_instance: run.task_instance,
        })
    }
}

impl DecisionRef {
    /// The reference to a decision record read back from storage, so a fork
    /// can start from a decision made in an earlier run. `None` if the
    /// record is not a decision of an attempt.
    pub fn from_record(record: &Record) -> Option<Self> {
        match (
            &record.body,
            record.attempt,
            record.family,
            record.task_instance,
        ) {
            (Body::Decision(d), Some(attempt), Some(family), Some(task_instance)) => Some(Self {
                id: record.id,
                state: d.state,
                parent: record.parent,
                attempt,
                family,
                task_instance,
            }),
            _ => None,
        }
    }
}
