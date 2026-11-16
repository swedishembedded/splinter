// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements compact chat formats and tool-call
// conventions for small models trained from scratch, for its clients. If
// your team needs expertise in designing a model's conversation format, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The model's chat template: its special tokens and how a conversation is
//! written as text.
//!
//! The template is compact because the model is small: one token opens each
//! turn and names its role, one token ends it. The tools offered are one
//! line in the system turn, each tool with its parameter names in sorted
//! order (so the order a runtime lists them in never changes the text). A
//! tool call is a JSON object between `<tool_call>` and `</tool_call>` inside
//! an assistant turn, the form the server parses back into an OpenAI
//! `tool_calls` entry.
//!
//! [`Template::jinja`] is the same format as a chat template the checkpoint
//! carries, so brain renders the fine-tuning records and the served
//! conversations exactly as [`Template::render`] writes the pretraining
//! conversations; a test holds the two to the same bytes.

use serde::{Deserialize, Serialize};

/// The template's special tokens and markers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Template {
    /// Opens a system turn.
    pub system: String,
    /// Opens a user turn.
    pub user: String,
    /// Opens an assistant turn.
    pub assistant: String,
    /// Opens a tool-result turn.
    pub tool: String,
    /// Ends any turn.
    pub end: String,
    /// Opens a tool call inside an assistant turn.
    pub call_open: String,
    /// Closes a tool call.
    pub call_close: String,
    /// The padding token.
    pub pad: String,
    /// Separates documents in the pretraining stream.
    pub separator: String,
}

impl Default for Template {
    fn default() -> Self {
        Self {
            system: "<|system|>".into(),
            user: "<|user|>".into(),
            assistant: "<|assistant|>".into(),
            tool: "<|tool|>".into(),
            end: "<|end|>".into(),
            call_open: "<tool_call>".into(),
            call_close: "</tool_call>".into(),
            pad: "<|pad|>".into(),
            separator: "<|endoftext|>".into(),
        }
    }
}

/// One tool call as the model writes it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Call {
    /// The tool.
    pub name: String,
    /// Its arguments.
    pub arguments: serde_json::Value,
}

/// A tool as the template lists it: its name and parameter names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolSig {
    /// The tool's name.
    pub name: String,
    /// Its parameters' names; written sorted.
    pub params: Vec<String>,
}

/// One turn of a conversation.
#[derive(Clone, Debug, PartialEq)]
pub enum Turn {
    /// The system prompt.
    System(String),
    /// What the user said.
    User(String),
    /// What the assistant said, with the calls it made.
    Assistant {
        /// Its text.
        text: String,
        /// Its tool calls, in order.
        calls: Vec<Call>,
    },
    /// A tool's result.
    Tool(String),
}

impl Template {
    /// Every special token, in a stable order: what the tokenizer must keep
    /// whole.
    #[must_use]
    pub fn specials(&self) -> Vec<String> {
        vec![
            self.pad.clone(),
            self.system.clone(),
            self.user.clone(),
            self.assistant.clone(),
            self.tool.clone(),
            self.end.clone(),
            self.call_open.clone(),
            self.call_close.clone(),
            self.separator.clone(),
        ]
    }

    fn tools_line(tools: &[ToolSig]) -> String {
        let listed: Vec<String> = tools
            .iter()
            .map(|t| {
                let mut params = t.params.clone();
                params.sort();
                format!("{}({})", t.name, params.join(","))
            })
            .collect();
        format!("Tools: {}", listed.join(" "))
    }

    /// `turns` written as the model reads them, with `tools` offered.
    #[must_use]
    pub fn render(&self, tools: &[ToolSig], turns: &[Turn]) -> String {
        let mut out = String::new();
        let line = (!tools.is_empty()).then(|| Self::tools_line(tools));
        if let Some(line) = &line {
            if !matches!(turns.first(), Some(Turn::System(_))) {
                out.push_str(&format!("{}{line}{}\n", self.system, self.end));
            }
        }
        for (i, turn) in turns.iter().enumerate() {
            let (open, body) = match turn {
                Turn::System(text) => {
                    let mut body = text.clone();
                    if let (0, Some(line)) = (i, &line) {
                        body.push('\n');
                        body.push_str(line);
                    }
                    (&self.system, body)
                }
                Turn::User(text) => (&self.user, text.clone()),
                Turn::Tool(text) => (&self.tool, text.clone()),
                Turn::Assistant { text, calls } => {
                    let mut body = text.clone();
                    for (j, call) in calls.iter().enumerate() {
                        if !text.is_empty() || j > 0 {
                            body.push('\n');
                        }
                        body.push_str(&format!(
                            "{}{{\"name\": \"{}\", \"arguments\": {}}}{}",
                            self.call_open, call.name, call.arguments, self.call_close
                        ));
                    }
                    (&self.assistant, body)
                }
            };
            out.push_str(&format!("{open}{body}{}\n", self.end));
        }
        out
    }

    /// The format as a chat template over OpenAI-shaped `messages` and
    /// `tools`, with `add_generation_prompt` opening the assistant's turn.
    /// A call's `arguments` is the raw JSON text of the wire format.
    #[must_use]
    pub fn jinja(&self) -> String {
        let Self {
            system,
            user,
            assistant,
            tool,
            end,
            call_open,
            call_close,
            ..
        } = self;
        format!(
            "{{%- if tools -%}}{{%- set tl -%}}Tools: {{% for t in tools %}}{{{{ t.function.name }}}}(\
             {{%- for k, v in (t.function.parameters.properties or {{}}) | dictsort -%}}{{{{ k }}}}\
             {{%- if not loop.last %}},{{% endif -%}}{{%- endfor -%}}){{% if not loop.last %}} {{% endif %}}\
             {{%- endfor -%}}{{%- endset -%}}{{%- endif -%}}\
             {{%- if tools and (not messages or messages[0].role != 'system') -%}}{system}{{{{ tl }}}}{end}\n{{% endif -%}}\
             {{%- for m in messages -%}}\
             {{%- if m.role == 'system' -%}}{system}{{{{ m.content }}}}{{% if tools and loop.first %}}\n{{{{ tl }}}}{{% endif %}}{end}{{{{ \"\\n\" }}}}\
             {{%- elif m.role == 'user' -%}}{user}{{{{ m.content }}}}{end}{{{{ \"\\n\" }}}}\
             {{%- elif m.role == 'tool' -%}}{tool}{{{{ m.content }}}}{end}{{{{ \"\\n\" }}}}\
             {{%- elif m.role == 'assistant' -%}}{assistant}{{{{ m.content or '' }}}}\
             {{%- for c in (m.tool_calls or []) -%}}{{% if m.content or not loop.first %}}\n{{% endif %}}\
             {call_open}{{\"name\": \"{{{{ c.function.name }}}}\", \"arguments\": {{{{ c.function.arguments | tojson }}}}}}{call_close}\
             {{%- endfor -%}}{end}{{{{ \"\\n\" }}}}\
             {{%- endif -%}}{{%- endfor -%}}\
             {{%- if add_generation_prompt -%}}{assistant}{{%- endif -%}}"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Vec<ToolSig> {
        vec![
            ToolSig {
                name: "shell".into(),
                params: vec!["shell_command".into(), "workdir".into()],
            },
            ToolSig {
                name: "read_file".into(),
                params: vec!["path".into(), "limit".into()],
            },
        ]
    }

    fn turns() -> Vec<Turn> {
        vec![
            Turn::System("Be brief.".into()),
            Turn::User("hi".into()),
            Turn::Assistant {
                text: String::new(),
                calls: vec![Call {
                    name: "shell".into(),
                    arguments: json!({ "shell_command": "ls" }),
                }],
            },
            Turn::Tool("a.txt".into()),
            Turn::Assistant {
                text: "One file.".into(),
                calls: vec![],
            },
        ]
    }

    #[test]
    fn a_tool_call_is_json_between_its_tags() {
        let text = Template::default().render(&tools(), &turns());
        assert_eq!(
            text,
            "<|system|>Be brief.\nTools: shell(shell_command,workdir) read_file(limit,path)<|end|>\n\
             <|user|>hi<|end|>\n\
             <|assistant|><tool_call>{\"name\": \"shell\", \"arguments\": {\"shell_command\":\"ls\"}}</tool_call><|end|>\n\
             <|tool|>a.txt<|end|>\n<|assistant|>One file.<|end|>\n"
        );
    }

    #[test]
    fn the_chat_template_renders_what_render_writes() {
        let t = Template::default();
        let wire_tools: Vec<_> = tools()
            .iter()
            .map(|s| {
                let props: serde_json::Map<_, _> = s.params.iter().rev().map(|p| (p.clone(), json!({}))).collect();
                json!({ "type": "function", "function": { "name": s.name, "parameters": { "properties": props } } })
            })
            .collect();
        let messages = json!([
            { "role": "system", "content": "Be brief." },
            { "role": "user", "content": "hi" },
            { "role": "assistant", "content": null, "tool_calls": [{ "id": "c0", "type": "function",
              "function": { "name": "shell", "arguments": { "shell_command": "ls" } } }] },
            { "role": "tool", "content": "a.txt" },
            { "role": "assistant", "content": "One file." },
        ]);
        let mut env = minijinja::Environment::new();
        let source = t.jinja();
        env.add_template("chat", &source).unwrap();
        let rendered = env
            .get_template("chat")
            .unwrap()
            .render(minijinja::context! { messages => messages, tools => wire_tools, add_generation_prompt => false })
            .unwrap();
        assert_eq!(rendered, t.render(&tools(), &turns()));
        let no_system = env
            .get_template("chat")
            .unwrap()
            .render(minijinja::context! { messages => json!([{ "role": "user", "content": "hi" }]), tools => wire_tools, add_generation_prompt => true })
            .unwrap();
        assert_eq!(
            no_system,
            format!(
                "{}<|assistant|>",
                t.render(&tools(), &[Turn::User("hi".into())])
            )
        );
    }
}
