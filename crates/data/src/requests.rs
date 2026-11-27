// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data for multi-role agent systems,
// where each model role learns from exactly the requests it was sent, for its
// clients. If your team needs expertise in fine-tuning the roles of an agent
// harness, including those that read images, you can procure our services by
// sending an email to info@swedishembedded.com.

//! A harness conversation as the requests its model was sent.
//!
//! A harness trial's request log holds every request exactly as sent - the
//! system prompt, the instruction with the pictures the model was shown, the
//! tool results with the pictures it fetched, the tools offered - and the
//! response each got. The conversation's trajectory holds none of the
//! pictures, and a user turn is missing from it where the harness attached
//! them. A conversation recorded with its requests
//! ([`splinter_core::harness::RecordedCalls`]) is rendered from them, so each
//! action's record is the request its model was asked, with the answer as the
//! supervised turn.
//!
//! Requests grow: each repeats the one before and adds the model's answer and
//! what came back. The conversation is the longest request followed by its
//! own answer, and every earlier request must be a prefix of it with the
//! answer to it next, unless the agent dropped that answer (it stopped a call
//! whose arguments ran on and put a note in its place, or asked again): such
//! an answer made no step of the agent's and is no action. A model that wrote
//! a call's arguments as text that is not JSON is sent the call back as the
//! agent repaired it; that answer is no action (no record can carry it) and
//! is counted, the repaired turn context. A conversation whose context was
//! rewritten between two requests (compaction, an edited note) is refused as
//! one ([`Unprojectable::RequestsDiverge`]), because no single list of
//! messages holds what the model saw; each of its requests is still what the
//! model was sent, and is read on its own ([`conversations_per_request`]).
//!
//! Sven writes a request's messages as: `system` and `user` turns whose
//! content is a string or a list of `text` and `image` parts; an assistant
//! turn as one message with the text (and its `reasoning`) followed by one
//! message per tool call, each content `{tool_call_id, function}`; a tool
//! result as `{tool_call_id, content}` with content a string or parts. They
//! become the same messages a trajectory renders ([`crate::trajectory`]):
//! one assistant message per turn with all its calls, scrubbed, calls named
//! `s<step>-<id>`, and otherwise as sent: the text around a call and the
//! arguments as the model wrote them, which a chat template prints as given
//! (so a record's context is the one its prompt ids name). The turn that made no call and said nothing is
//! dropped, as an empty agent step is.

use std::collections::{HashMap, HashSet};

use serde_json::Value;
use splinter_core::chat::{WireFunction, WireImage, WireMessage, WireToolCall};
use splinter_core::experience::Experience;
use splinter_core::harness::{image_url, is_media, message_parts, ModelCall, ModelResponse};

use crate::markers::{chat_tool_list, is_incomplete};
use crate::record::RecordBody;
use crate::render::message;
use crate::request_log::made_a_step;
use crate::scrub::Scrub;
use crate::trajectory::{agent_steps, holds_reasoning_markup, Action, AgentStep, Conversation};
use crate::unprojectable::Unprojectable;
use crate::Exclusion;

/// The conversation `calls` are, for the agent steps of `experience`'s
/// trajectory, which they must divide among one request per step in order.
pub(crate) fn conversation_of_requests(
    experience: &Experience,
    calls: &[ModelCall],
) -> Result<Conversation, Exclusion> {
    Ok(build(&experience.trajectory, calls)?)
}

/// The conversations `calls` are, each request on its own, for a log whose
/// requests do not extend one another (the agent rewrote its context between
/// two of them: compaction, an edited note): each answer that is an agent
/// step of `experience`'s trajectory is the one action of its own request as
/// it was sent, under the tools that request offered.
///
/// An answer is a step when it makes the step's tool calls (their ids), or,
/// calling nothing, says the step's text; the answers are paired with the
/// steps in order, and one that is no step's (an answer the agent dropped, the
/// notes a compaction asked for) is no action. An answer whose arguments are
/// not JSON is its request's context alone, and a step no answer pairs with
/// has no context its model was sent: each is left out and counted
/// ([`Conversation::left_out`]).
///
/// # Errors
/// No agent step pairs with an answer ([`Unprojectable::RequestsUnpaired`]).
/// A request that cannot be read.
pub(crate) fn conversations_per_request(
    experience: &Experience,
    calls: &[ModelCall],
) -> Result<Vec<Conversation>, Exclusion> {
    Ok(per_request(&experience.trajectory, calls)?)
}

fn per_request(
    trajectory: &atif::Trajectory,
    calls: &[ModelCall],
) -> Result<Vec<Conversation>, Unprojectable> {
    if is_incomplete(trajectory) {
        return Err(Unprojectable::Incomplete);
    }
    let made: HashMap<u64, &atif::TraceStep> = trajectory
        .steps
        .iter()
        .filter(|s| s.source == atif::StepOrigin::Agent)
        .map(|s| (s.step_id, s))
        .collect();
    // The step that made each call, to name the calls of a request's history
    // as the conversation of the whole run names them.
    let step_of_call: HashMap<&str, u64> = made
        .values()
        .flat_map(|s| {
            s.tool_calls
                .iter()
                .flatten()
                .map(|c| (c.tool_call_id.as_str(), s.step_id))
        })
        .collect();
    let steps = answering_steps(trajectory);
    let scrub = Scrub::of(trajectory);
    let mut out: Vec<Conversation> = Vec::new();
    // Steps passed over since the last answer paired: no request answered
    // them.
    let mut unanswered = Vec::new();
    let mut from = 0;
    for call in calls.iter().filter(|c| made_a_step(c)) {
        let Some(found) = steps[from..].iter().position(|step| {
            made.get(&step.step)
                .is_some_and(|made| answers(made, &call.response))
        }) else {
            continue;
        };
        unanswered.extend(
            steps[from..from + found]
                .iter()
                .map(|_| Exclusion::RequestsUnpaired),
        );
        let step = steps[from + found];
        from += found + 1;
        let (mut messages, answer) = sent_and_answered(call, &scrub, step.step)?;
        let tools = chat_tool_list(&call.tools)?;
        let Some(Answer {
            turn: answer,
            parsed,
        }) = answer
        else {
            continue;
        };
        let at = messages.len();
        let mut actions = Vec::new();
        let mut left_out = std::mem::take(&mut unanswered);
        if parsed {
            messages.push(answer);
            actions.push(Action {
                index: at,
                step: step.step,
                copied: step.copied,
                repeated: step.repeated,
                tools: 0,
            });
        } else {
            left_out.push(Exclusion::UnparsedArguments);
        }
        let step_of: HashMap<usize, (usize, u64)> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == "assistant")
            .filter_map(|(i, m)| {
                let named = m.tool_calls.first().and_then(|c| c.id.as_deref());
                let made = if i == at {
                    Some(step.step)
                } else {
                    named.and_then(|id| step_of_call.get(id).copied())
                };
                made.map(|made| (i, (i, made)))
            })
            .collect();
        scope_calls(&mut messages, &step_of)?;
        out.push(Conversation {
            messages,
            actions,
            tool_sets: vec![tools],
            scrubbed: scrub.changed_anything(),
            as_sent: true,
            left_out,
        });
    }
    unanswered.extend(steps[from..].iter().map(|_| Exclusion::RequestsUnpaired));
    match out.last_mut() {
        Some(last) => {
            last.left_out.append(&mut unanswered);
            Ok(out)
        }
        None => Err(Unprojectable::RequestsUnpaired {
            step: steps.first().map_or(0, |s| s.step),
        }),
    }
}

/// A model's answer as the assistant turn it is, and whether a record can
/// carry it (the arguments of its calls are JSON).
pub(crate) struct Answer {
    /// The turn.
    pub(crate) turn: WireMessage,
    /// Whether the arguments of every call of the turn are JSON.
    pub(crate) parsed: bool,
}

/// The request `call` as it was sent (its messages, none supervised, call ids
/// as the request names them) and the model's answer to it as the turn it is,
/// when it said or called something, with whether a record can carry it (the
/// arguments of its calls are JSON). Every reading of a request on its own
/// renders it so: an action's own request here, a model call of an online
/// rollout ([`crate::calls`]).
pub(crate) fn sent_and_answered(
    call: &ModelCall,
    scrub: &Scrub,
    step: u64,
) -> Result<(Vec<WireMessage>, Option<Answer>), Unprojectable> {
    Ok((
        wire_messages(&call.messages, scrub, step)?,
        answer_as_written(&call.response, scrub, step)?,
    ))
}

/// Whether `response` is the answer `step` records: the same tool calls (by
/// id: an agent may run, and send back, a reply's calls in another order),
/// or, calling nothing, the same text.
fn answers(step: &atif::TraceStep, response: &ModelResponse) -> bool {
    let mut made: Vec<&str> = step
        .tool_calls
        .iter()
        .flatten()
        .map(|c| c.tool_call_id.as_str())
        .collect();
    let mut answered: Vec<&str> = response.tool_calls.iter().map(|c| c.id.as_str()).collect();
    if made.is_empty() && answered.is_empty() {
        return step
            .message
            .as_text()
            .is_some_and(|text| text == response.text);
    }
    made.sort_unstable();
    answered.sort_unstable();
    made == answered
}

/// A record of a whole conversation as its requests were sent (`native-sft`).
#[derive(Clone, Debug, PartialEq)]
pub struct NativeRecord {
    /// The record: the conversation as the agent's model was sent it, its
    /// system prompt and tools included, every action supervised.
    pub record: RecordBody,
    /// The agent's actions the record supervises.
    pub actions: usize,
    /// Actions left unsupervised because ATIF marks them copied from another
    /// trajectory (never the policy's own).
    pub copied: usize,
    /// Whether scrubbing (host names, scratch paths) changed any text.
    pub scrubbed: bool,
    /// The tools the supervised actions call, in order.
    pub tools: Vec<String>,
    /// The calls (by their scoped ids) whose result the model was shown as
    /// an error; empty when the record holds no request to tell it by
    /// ([`crate::mask_failed_actions`] takes the actions made only of them
    /// out of the loss).
    pub failed_calls: std::collections::BTreeSet<String>,
}

/// The `native-sft` record of a trajectory whose model calls were recorded:
/// the longest request with every earlier request a prefix of it, each
/// answer in place, every action of the agent supervised except one copied
/// from another trajectory, and one whose arguments are not JSON (the call as
/// the agent repaired it stays context). Nothing else is selected or rewritten: no
/// replaced system prompt, no tool list, no repeat or length filter.
///
/// # Errors
/// The calls are not the trajectory's (their count differs from its agent
/// steps, or the requests diverge from one another), or a request cannot be
/// read; [`Unprojectable`] says which.
pub fn native_record(
    trajectory: &atif::Trajectory,
    calls: &[ModelCall],
) -> Result<NativeRecord, Unprojectable> {
    let conversation = build(trajectory, calls)?;
    let supervised: Vec<&crate::trajectory::Action> =
        conversation.actions.iter().filter(|a| !a.copied).collect();
    let copied = conversation.actions.len() - supervised.len();
    let record = conversation
        .packed_record(&supervised)
        .ok_or(Unprojectable::NoAction)?;
    let tools = supervised
        .iter()
        .flat_map(|a| {
            conversation.messages[a.index]
                .tool_calls
                .iter()
                .map(|c| c.function.name.clone())
        })
        .collect();
    let failed_calls = calls.last().map_or_else(Default::default, |last| {
        crate::tool_errors::failed_calls(trajectory, &last.messages, &conversation.messages)
    });
    Ok(NativeRecord {
        record,
        actions: supervised.len(),
        copied,
        scrubbed: conversation.scrubbed,
        tools,
        failed_calls,
    })
}

/// The agent steps that answered a request: a step that says and calls
/// nothing answers none (a runtime may file a reply's reasoning, a request it
/// never had answered or a request's token account as a step of its own).
fn answering_steps(trajectory: &atif::Trajectory) -> Vec<AgentStep> {
    agent_steps(trajectory)
        .into_iter()
        .filter(|s| !s.silent)
        .collect()
}

/// The conversation the requests `calls` are, when they extend one another.
///
/// An answer that said nothing made no step and is passed over. The
/// conversation is the longest request followed by its own answer; every
/// earlier request is a prefix of it, and the turn after it is its answer,
/// unless the agent dropped the answer (it stopped it, or sent the request
/// again): then the answer was never the agent's step, and is no action. An
/// answer whose arguments are not JSON is no action either, and is counted
/// ([`Conversation::left_out`]): the turn the next request holds is the call
/// as the agent repaired it, which the model was then sent. The answers kept
/// pair in order with the agent steps that said or called something.
fn build(
    trajectory: &atif::Trajectory,
    calls: &[ModelCall],
) -> Result<Conversation, Unprojectable> {
    if is_incomplete(trajectory) {
        return Err(Unprojectable::Incomplete);
    }
    let steps = answering_steps(trajectory);
    let calls: Vec<&ModelCall> = calls.iter().filter(|c| made_a_step(c)).collect();
    let last_step = steps.last().map_or(0, |s| s.step);
    // The step a request's refusal names: the one it pairs with when no
    // answer was dropped before it.
    let step_named = |k: usize| steps.get(k).map_or(last_step, |s| s.step);
    let Some(last) = calls.len().checked_sub(1) else {
        return Err(Unprojectable::RequestsDiverge { step: last_step });
    };
    let scrub = Scrub::of(trajectory);

    let mut tool_sets: Vec<Vec<Value>> = Vec::new();
    let mut tools_of = Vec::with_capacity(calls.len());
    for call in &calls {
        let tools = chat_tool_list(&call.tools)?;
        let at = tool_sets.iter().position(|set| *set == tools);
        tools_of.push(at.unwrap_or_else(|| {
            tool_sets.push(tools);
            tool_sets.len() - 1
        }));
    }

    // The conversation is the longest request; every request is a prefix of
    // it.
    let mut chain = wire_messages(&calls[last].messages, &scrub, last_step)?;
    let mut context_len = Vec::with_capacity(calls.len());
    for (k, call) in calls.iter().enumerate() {
        let own = if k == last {
            chain.len()
        } else {
            let own = wire_messages(&call.messages, &scrub, step_named(k))?;
            if chain.get(..own.len()) != Some(own.as_slice()) {
                return Err(Unprojectable::RequestsDiverge {
                    step: step_named(k),
                });
            }
            own.len()
        };
        context_len.push(own);
    }
    // The answers that were the agent's steps, in order: where each is in
    // the conversation (none for a last answer no record can carry), and
    // whether a record can carry it.
    let mut answers: Vec<(usize, Option<usize>, bool)> = Vec::with_capacity(calls.len());
    for (k, call) in calls.iter().enumerate() {
        let Some(Answer {
            turn: answer,
            parsed,
        }) = answer_as_written(&call.response, &scrub, step_named(k))?
        else {
            continue;
        };
        let at = context_len[k];
        if k == last {
            let at = parsed.then(|| {
                chain.push(answer);
                at
            });
            answers.push((k, at, parsed));
        } else if chain.get(at).is_some_and(|next| same_turn(next, &answer)) {
            answers.push((k, Some(at), parsed));
        }
        // Otherwise the agent dropped the answer: never its step.
    }
    if answers.len() != steps.len() {
        return Err(Unprojectable::RequestsDiverge { step: last_step });
    }
    // Every assistant turn of the conversation is the answer to a request.
    let held = answers.iter().filter(|(_, at, _)| at.is_some()).count();
    let spoken = chain.iter().filter(|m| m.role == "assistant").count();
    if spoken != held {
        return Err(Unprojectable::RequestsDiverge { step: last_step });
    }
    let step_of: HashMap<usize, (usize, u64)> = answers
        .iter()
        .zip(&steps)
        .filter_map(|(&(k, at, _), step)| at.map(|at| (at, (k, step.step))))
        .collect();
    scope_calls(&mut chain, &step_of)?;

    let mut actions = Vec::new();
    for (&(k, at, parsed), step) in answers.iter().zip(&steps) {
        let Some(at) = at.filter(|_| parsed) else {
            continue;
        };
        if !chain[at].tool_calls.is_empty() || k == last {
            actions.push(Action {
                index: at,
                step: step.step,
                copied: step.copied,
                repeated: step.repeated,
                tools: tools_of[k],
            });
        }
    }
    Ok(Conversation {
        messages: chain,
        actions,
        tool_sets,
        scrubbed: scrub.changed_anything(),
        as_sent: true,
        left_out: answers
            .iter()
            .filter(|(_, _, parsed)| !parsed)
            .map(|_| Exclusion::UnparsedArguments)
            .collect(),
    })
}

/// Names every call `s<step>-<id>` and the result that answers it likewise,
/// as a trajectory's calls are named: ids repeat across a model's responses.
pub(crate) fn scope_calls(
    chain: &mut [WireMessage],
    step_of: &HashMap<usize, (usize, u64)>,
) -> Result<(), Unprojectable> {
    let mut step = 0;
    let mut names: HashMap<String, String> = HashMap::new();
    for (at, message) in chain.iter_mut().enumerate() {
        match message.role.as_str() {
            "assistant" => {
                step = step_of.get(&at).map_or(step, |(_, step)| *step);
                names.clear();
                let mut ids = HashSet::new();
                for call in &mut message.tool_calls {
                    let id = call.id.take().unwrap_or_default();
                    if !ids.insert(id.clone()) {
                        return Err(Unprojectable::DuplicateCallId { step });
                    }
                    let scoped = format!("s{step}-{id}");
                    names.insert(id, scoped.clone());
                    call.id = Some(scoped);
                }
            }
            "tool" => {
                let answered = message
                    .tool_call_id
                    .as_ref()
                    .and_then(|id| names.get(id))
                    .cloned();
                message.tool_call_id =
                    Some(answered.ok_or(Unprojectable::UnansweredObservation { step })?);
            }
            _ => {}
        }
    }
    Ok(())
}

/// Whether `history`, a turn as a later request repeats it, is `answer`
/// exactly: the runtime sends every answer back as it arrived, so its text,
/// its reasoning and each call's id, name and argument text are the same
/// bytes. Anything else is a history the runtime rewrote.
fn same_turn(history: &WireMessage, answer: &WireMessage) -> bool {
    let (a, b) = (history, answer);
    a.role == b.role
        && a.content == b.content
        && a.reasoning_content == b.reasoning_content
        && a.tool_calls.len() == b.tool_calls.len()
        && a.tool_calls.iter().zip(&b.tool_calls).all(|(x, y)| {
            x.id == y.id
                && x.function.name == y.function.name
                && x.function.arguments == y.function.arguments
        })
}

/// The reasoning a history message `raw` carries, scrubbed; `None` when it
/// carries none.
fn reasoning_of(raw: &Value, scrub: &Scrub) -> Option<String> {
    raw.get("reasoning")
        .and_then(Value::as_str)
        .filter(|r| !r.trim().is_empty())
        .map(|r| scrub.text(r))
}

/// The request's messages as chat messages, none supervised, call ids as the
/// request names them. The first user turn and the system turn are as sent;
/// the rest is scrubbed like a trajectory's text.
pub(crate) fn wire_messages(
    messages: &[Value],
    scrub: &Scrub,
    step: u64,
) -> Result<Vec<WireMessage>, Unprojectable> {
    let unreadable = Unprojectable::UnreadableRequest { step };
    let mut out: Vec<WireMessage> = Vec::with_capacity(messages.len());
    let mut users = 0;
    for raw in messages {
        let role = raw.get("role").and_then(Value::as_str).ok_or(unreadable)?;
        let content = raw.get("content").ok_or(unreadable)?;
        match role {
            "system" | "user" => {
                users += usize::from(role == "user");
                // The first user turn is the instruction as it was sent.
                let scrub = (role == "user" && users > 1).then_some(scrub);
                let (text, images) =
                    text_and_images(content, role == "user", scrub).ok_or(unreadable)?;
                let mut turn = message(role, &text, false);
                turn.images = images;
                out.push(turn);
            }
            "assistant" => match content {
                Value::String(text) => {
                    let mut turn = message("assistant", &scrub.text(text), false);
                    turn.reasoning_content = reasoning_of(raw, scrub);
                    check_markup(&turn, step)?;
                    out.push(turn);
                }
                call => {
                    // A call in the history is context as it was sent, its
                    // argument text as written whether or not it is JSON.
                    let (call, _) = written_call(call, scrub, step)?;
                    match out.last_mut().filter(|m| m.role == "assistant") {
                        Some(turn) => turn.tool_calls.push(call),
                        None => {
                            // A turn that reasoned and called without text
                            // carries its reasoning on its first call.
                            let mut turn = message("assistant", "", false);
                            turn.reasoning_content = reasoning_of(raw, scrub);
                            turn.tool_calls.push(call);
                            check_markup(&turn, step)?;
                            out.push(turn);
                        }
                    }
                }
            },
            "tool" => {
                let id = content
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .ok_or(unreadable)?;
                let result = content.get("content").ok_or(unreadable)?;
                let (text, images) =
                    text_and_images(result, true, Some(scrub)).ok_or(unreadable)?;
                let mut turn = message("tool", &text, false);
                turn.images = images;
                turn.tool_call_id = Some(id.to_string());
                out.push(turn);
            }
            _ => return Err(unreadable),
        }
    }
    out.retain(|m| {
        m.role != "assistant" || !m.content.trim().is_empty() || !m.tool_calls.is_empty()
    });
    Ok(out)
}

/// The text of `content` (scrubbed when `scrub` is given) and the images in
/// it, each at the offset of the text it follows; `None` for content that is
/// neither text nor text and images, that holds an image where `images` are
/// not allowed, or an image that is not an inline `data:` URL.
fn text_and_images(
    content: &Value,
    images: bool,
    scrub: Option<&Scrub>,
) -> Option<(String, Vec<WireImage>)> {
    let clean = |text: &str| scrub.map_or_else(|| text.to_string(), |s| s.text(text));
    let mut text = String::new();
    let mut found = Vec::new();
    match content {
        Value::String(whole) => text.push_str(&clean(whole)),
        Value::Array(_) => {
            let wrapper = serde_json::json!({ "content": content });
            for part in message_parts(&wrapper) {
                if let Some(url) = image_url(part) {
                    if !images || !url.starts_with("data:") {
                        return None;
                    }
                    found.push(WireImage {
                        at: text.len(),
                        url: url.to_string(),
                    });
                } else if is_media(part) {
                    return None;
                } else {
                    text.push_str(&clean(part.get("text")?.as_str()?));
                }
            }
        }
        _ => return None,
    }
    Some((text, found))
}

fn check_markup(turn: &WireMessage, step: u64) -> Result<(), Unprojectable> {
    let holds = holds_reasoning_markup(&turn.content)
        || turn
            .reasoning_content
            .as_deref()
            .is_some_and(holds_reasoning_markup);
    if holds {
        Err(Unprojectable::ReasoningInText { step })
    } else {
        Ok(())
    }
}

/// A tool call of a request history, `{tool_call_id, function}`, or of a
/// response, `{id, name, arguments}`, as a wire call with its argument text as
/// written, and whether that text is JSON.
fn written_call(
    call: &Value,
    scrub: &Scrub,
    step: u64,
) -> Result<(WireToolCall, bool), Unprojectable> {
    let unreadable = Unprojectable::UnreadableRequest { step };
    let (id, name, arguments) = match call.get("function") {
        Some(function) => (
            call.get("tool_call_id"),
            function.get("name"),
            function.get("arguments"),
        ),
        None => (call.get("id"), call.get("name"), call.get("arguments")),
    };
    let id = id.and_then(Value::as_str).ok_or(unreadable)?;
    let name = name.and_then(Value::as_str).ok_or(unreadable)?;
    // Arguments the model wrote as text stay that text (a chat template
    // prints them as given); a record carries them only when they are JSON.
    let (arguments, parsed) = match arguments.ok_or(unreadable)? {
        Value::String(text) => (text.clone(), serde_json::from_str::<Value>(text).is_ok()),
        parsed => (parsed.to_string(), true),
    };
    Ok((
        WireToolCall {
            id: Some(id.to_string()),
            kind: "function".into(),
            function: WireFunction {
                name: name.to_string(),
                arguments: scrub.text(&arguments),
            },
        },
        parsed,
    ))
}

/// What the model answered as the assistant turn it is, whether or not the
/// arguments of its calls are JSON, and whether they all are; `None` for an
/// answer with no text and no call.
fn answer_as_written(
    response: &ModelResponse,
    scrub: &Scrub,
    step: u64,
) -> Result<Option<Answer>, Unprojectable> {
    let mut turn = message("assistant", &scrub.text(&response.text), false);
    turn.reasoning_content = response
        .reasoning
        .as_deref()
        .filter(|r| !r.trim().is_empty())
        .map(|r| scrub.text(r));
    let mut parsed = true;
    for call in &response.tool_calls {
        let call = serde_json::json!({
            "id": call.id, "name": call.name, "arguments": call.arguments,
        });
        let (call, json) = written_call(&call, scrub, step)?;
        parsed &= json;
        turn.tool_calls.push(call);
    }
    check_markup(&turn, step)?;
    let said = !turn.content.trim().is_empty() || !turn.tool_calls.is_empty();
    Ok(said.then_some(Answer { turn, parsed }))
}
