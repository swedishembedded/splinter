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
//! answer to it next; a conversation whose context was rewritten between two
//! requests is refused ([`Unprojectable::RequestsDiverge`]), because no single
//! list of messages holds what the model saw.
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
use crate::scrub::Scrub;
use crate::trajectory::{agent_steps, holds_reasoning_markup, Action, Conversation};
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
/// from another trajectory. Nothing else is selected or rewritten: no
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
    // A step that says and calls nothing answers no request (a runtime may
    // file a reply's reasoning or a request's token account as a step of its
    // own): the calls pair with the steps that said or called something.
    let steps = agent_steps(trajectory)
        .into_iter()
        .filter(|s| !s.silent)
        .collect();
    let conversation = build_from(trajectory, calls, steps)?;
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

fn build(
    trajectory: &atif::Trajectory,
    calls: &[ModelCall],
) -> Result<Conversation, Unprojectable> {
    build_from(trajectory, calls, agent_steps(trajectory))
}

fn build_from(
    trajectory: &atif::Trajectory,
    calls: &[ModelCall],
    steps: Vec<crate::trajectory::AgentStep>,
) -> Result<Conversation, Unprojectable> {
    if is_incomplete(trajectory) {
        return Err(Unprojectable::Incomplete);
    }
    let Some(last) = calls.len().checked_sub(1) else {
        return Err(Unprojectable::RequestsDiverge { step: 0 });
    };
    if steps.len() != calls.len() {
        let step = steps.last().map_or(0, |s| s.step);
        return Err(Unprojectable::RequestsDiverge { step });
    }
    let scrub = Scrub::of(trajectory);

    let mut tool_sets: Vec<Vec<Value>> = Vec::new();
    let mut tools_of = Vec::with_capacity(calls.len());
    for call in calls {
        let tools = chat_tool_list(&call.tools)?;
        let at = tool_sets.iter().position(|set| *set == tools);
        tools_of.push(at.unwrap_or_else(|| {
            tool_sets.push(tools);
            tool_sets.len() - 1
        }));
    }

    // The conversation is the longest request; every request is a prefix of
    // it, and the answer to it comes next.
    let mut chain = wire_messages(&calls[last].messages, &scrub, steps[last].step)?;
    let mut context_len = Vec::with_capacity(calls.len());
    for (call, fact) in calls.iter().zip(&steps) {
        let own = if context_len.len() == last {
            chain.len()
        } else {
            let own = wire_messages(&call.messages, &scrub, fact.step)?;
            if chain.get(..own.len()) != Some(own.as_slice()) {
                return Err(Unprojectable::RequestsDiverge { step: fact.step });
            }
            own.len()
        };
        context_len.push(own);
    }
    let mut turns: Vec<Option<usize>> = Vec::with_capacity(calls.len());
    for (k, call) in calls.iter().enumerate() {
        let step = steps[k].step;
        let Some(answer) = answer_of(&call.response, &scrub, step)? else {
            turns.push(None);
            continue;
        };
        let at = context_len[k];
        if k == last {
            chain.push(answer);
        } else if !chain.get(at).is_some_and(|next| same_turn(next, &answer)) {
            return Err(Unprojectable::RequestsDiverge { step });
        }
        turns.push(Some(at));
    }
    // Every assistant turn of the conversation is the answer to a request.
    let answered: HashSet<usize> = turns.iter().flatten().copied().collect();
    let spoken = chain.iter().filter(|m| m.role == "assistant").count();
    if spoken != answered.len() {
        return Err(Unprojectable::RequestsDiverge {
            step: steps[last].step,
        });
    }
    let step_of: HashMap<usize, (usize, u64)> = turns
        .iter()
        .enumerate()
        .filter_map(|(k, at)| at.map(|at| (at, (k, steps[k].step))))
        .collect();
    scope_calls(&mut chain, &step_of)?;

    let mut actions = Vec::new();
    for (k, at) in turns.iter().enumerate() {
        let Some(at) = *at else { continue };
        if !chain[at].tool_calls.is_empty() || k == last {
            actions.push(Action {
                index: at,
                step: steps[k].step,
                copied: steps[k].copied,
                repeated: steps[k].repeated,
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

/// A turn's reasoning, when it has any.
fn reasoning(message: &WireMessage) -> Option<&str> {
    message
        .reasoning_content
        .as_deref()
        .map(str::trim)
        .filter(|r| !r.is_empty())
}

/// Whether `history`, a turn as a later request repeats it, is `answer`: the
/// text, the calls (with their ids, as the model and its history name them)
/// and the reasoning when the history carries it. A runtime that does not
/// send a turn's reasoning back (as a reasoning model's API asks) repeats
/// the turn without it, and it is still the same turn.
fn same_turn(history: &WireMessage, answer: &WireMessage) -> bool {
    let (a, b) = (history, answer);
    let arguments = |text: &str| serde_json::from_str::<Value>(text).ok();
    a.role == b.role
        && a.content.trim() == b.content.trim()
        && (reasoning(a).is_none() || reasoning(a) == reasoning(b))
        && a.tool_calls.len() == b.tool_calls.len()
        && a.tool_calls.iter().zip(&b.tool_calls).all(|(x, y)| {
            x.id == y.id
                && x.function.name == y.function.name
                && arguments(&x.function.arguments) == arguments(&y.function.arguments)
        })
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
                    turn.reasoning_content = raw
                        .get("reasoning")
                        .and_then(Value::as_str)
                        .filter(|r| !r.trim().is_empty())
                        .map(|r| scrub.text(r));
                    check_markup(&turn, step)?;
                    out.push(turn);
                }
                call => {
                    let call = wire_call(call, scrub, step)?;
                    match out.last_mut().filter(|m| m.role == "assistant") {
                        Some(turn) => turn.tool_calls.push(call),
                        None => {
                            let mut turn = message("assistant", "", false);
                            turn.tool_calls.push(call);
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
/// response, `{id, name, arguments}`, as a wire call with its arguments as
/// compact JSON.
fn wire_call(call: &Value, scrub: &Scrub, step: u64) -> Result<WireToolCall, Unprojectable> {
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
    // prints them as given); they must still be JSON.
    let arguments = match arguments.ok_or(unreadable)? {
        Value::String(text) => {
            serde_json::from_str::<Value>(text)
                .map_err(|_| Unprojectable::UnparsedArguments { step })?;
            text.clone()
        }
        parsed => parsed.to_string(),
    };
    Ok(WireToolCall {
        id: Some(id.to_string()),
        kind: "function".into(),
        function: WireFunction {
            name: name.to_string(),
            arguments: scrub.text(&arguments),
        },
    })
}

/// What the model answered as the assistant turn it is; `None` for an answer
/// with no text and no call.
pub(crate) fn answer_of(
    response: &ModelResponse,
    scrub: &Scrub,
    step: u64,
) -> Result<Option<WireMessage>, Unprojectable> {
    let mut turn = message("assistant", &scrub.text(&response.text), false);
    turn.reasoning_content = response
        .reasoning
        .as_deref()
        .filter(|r| !r.trim().is_empty())
        .map(|r| scrub.text(r));
    for call in &response.tool_calls {
        let call = serde_json::json!({
            "id": call.id, "name": call.name, "arguments": call.arguments,
        });
        turn.tool_calls.push(wire_call(&call, scrub, step)?);
    }
    check_markup(&turn, step)?;
    Ok((!turn.content.trim().is_empty() || !turn.tool_calls.is_empty()).then_some(turn))
}
