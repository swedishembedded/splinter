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
//! one assistant message per turn with all its calls, trimmed, scrubbed, calls
//! named `s<step>-<id>`. The turn that made no call and said nothing is
//! dropped, as an empty agent step is.

use std::collections::{HashMap, HashSet};

use serde_json::Value;
use splinter_core::chat::{WireFunction, WireImage, WireMessage, WireToolCall};
use splinter_core::experience::Experience;
use splinter_core::harness::{image_url, is_media, message_parts, ModelCall, ModelResponse};

use crate::markers::{chat_tool_list, is_incomplete};
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
    Ok(build(experience, calls)?)
}

fn build(experience: &Experience, calls: &[ModelCall]) -> Result<Conversation, Unprojectable> {
    let trajectory = &experience.trajectory;
    if is_incomplete(trajectory) {
        return Err(Unprojectable::Incomplete);
    }
    let steps = agent_steps(trajectory);
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
    })
}

/// Names every call `s<step>-<id>` and the result that answers it likewise,
/// as a trajectory's calls are named: ids repeat across a model's responses.
fn scope_calls(
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

/// Whether two assistant turns say the same: the text, the reasoning and the
/// calls (with their ids, as the model and its history name them).
fn same_turn(a: &WireMessage, b: &WireMessage) -> bool {
    a.role == b.role
        && a.content == b.content
        && a.reasoning_content == b.reasoning_content
        && a.tool_calls == b.tool_calls
}

/// The request's messages as chat messages, none supervised, call ids as the
/// request names them. The first user turn and the system turn are as sent;
/// the rest is scrubbed like a trajectory's text.
fn wire_messages(
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
                    let mut turn = message("assistant", &scrub.text(text.trim()), false);
                    turn.reasoning_content = raw
                        .get("reasoning")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|r| !r.is_empty())
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
    out.retain(|m| m.role != "assistant" || !m.content.is_empty() || !m.tool_calls.is_empty());
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
    let arguments = match arguments.ok_or(unreadable)? {
        Value::String(text) => serde_json::from_str::<Value>(text)
            .map_err(|_| Unprojectable::UnparsedArguments { step })?,
        parsed => parsed.clone(),
    };
    Ok(WireToolCall {
        id: Some(id.to_string()),
        kind: "function".into(),
        function: WireFunction {
            name: name.to_string(),
            arguments: scrub.text(&arguments.to_string()),
        },
    })
}

/// What the model answered as the assistant turn it is; `None` for an answer
/// with no text and no call.
fn answer_of(
    response: &ModelResponse,
    scrub: &Scrub,
    step: u64,
) -> Result<Option<WireMessage>, Unprojectable> {
    let mut turn = message("assistant", &scrub.text(response.text.trim()), false);
    turn.reasoning_content = response
        .reasoning
        .as_deref()
        .map(str::trim)
        .filter(|r| !r.is_empty())
        .map(|r| scrub.text(r));
    for call in &response.tool_calls {
        let call = serde_json::json!({
            "id": call.id, "name": call.name, "arguments": call.arguments,
        });
        turn.tool_calls.push(wire_call(&call, scrub, step)?);
    }
    check_markup(&turn, step)?;
    Ok((!turn.content.is_empty() || !turn.tool_calls.is_empty()).then_some(turn))
}
