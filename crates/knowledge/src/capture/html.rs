// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source capture that grounds every learned
// fact in exact, traceable bytes, for its clients. If your team needs
// expertise in knowledge acquisition or training-data provenance, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A reference manual published as HTML, as the Markdown the sectioner reads.
//!
//! Headings become ATX headings (so a section is what lies between two of
//! them), `pre` blocks fenced code, `code` spans backticks (which is where
//! the symbols a skill is about are found), list items bullets, and every
//! other tag is dropped; scripts, styles and comments are dropped with their
//! content, and the common entities are decoded. The conversion is a pure
//! function of the bytes: the same page gives the same Markdown.

/// `html` as Markdown; see the module documentation.
#[must_use]
pub fn html_to_markdown(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let mut rest = html;
    let mut in_pre = false;
    let mut skip_until: Option<&str> = None;
    while !rest.is_empty() {
        let Some(open) = rest.find('<') else {
            push_text(&mut out, rest, in_pre);
            break;
        };
        if skip_until.is_none() {
            push_text(&mut out, &rest[..open], in_pre);
        }
        rest = &rest[open..];
        if let Some(comment) = rest.strip_prefix("<!--") {
            rest = comment.find("-->").map_or("", |end| &comment[end + 3..]);
            continue;
        }
        let Some(close) = rest.find('>') else {
            break;
        };
        let tag = &rest[1..close];
        rest = &rest[close + 1..];
        let (closing, name) = match tag.strip_prefix('/') {
            Some(name) => (true, name),
            None => (false, tag),
        };
        let name = name
            .split(|c: char| c.is_whitespace() || c == '/')
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if let Some(until) = skip_until {
            if closing && name == until {
                skip_until = None;
            }
            continue;
        }
        match (closing, name.as_str()) {
            (false, "script") => skip_until = Some("script"),
            (false, "style") => skip_until = Some("style"),
            (false, "pre") => {
                in_pre = true;
                out.push_str("\n\n```\n");
            }
            (true, "pre") => {
                in_pre = false;
                if !out.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str("```\n\n");
            }
            (_, "code" | "tt") if !in_pre => out.push('`'),
            (false, "h1" | "h2" | "h3" | "h4" | "h5" | "h6") => {
                let level = usize::from(name.as_bytes()[1] - b'0');
                out.push_str("\n\n");
                out.push_str(&"#".repeat(level));
                out.push(' ');
            }
            (
                true,
                "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "p" | "div" | "ul" | "ol" | "table",
            ) => {
                out.push_str("\n\n");
            }
            (false, "p" | "div" | "ul" | "ol" | "table" | "tr" | "br") => out.push_str("\n\n"),
            (false, "li") => out.push_str("\n- "),
            _ => {}
        }
    }
    tidy(&out)
}

/// Appends `text` with its entities decoded and, outside `pre`, its
/// whitespace collapsed.
fn push_text(out: &mut String, text: &str, in_pre: bool) {
    let decoded = decode(text);
    if in_pre {
        // A newline right after `<pre>` is not part of the block.
        let opening = out.ends_with("```\n");
        out.push_str(if opening {
            decoded.trim_start_matches('\n')
        } else {
            &decoded
        });
        return;
    }
    let mut space = out.ends_with(|c: char| c.is_whitespace());
    for c in decoded.chars() {
        if c.is_whitespace() {
            if !space {
                out.push(' ');
            }
            space = true;
        } else {
            out.push(c);
            space = false;
        }
    }
}

fn decode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let decoded = rest.find(';').filter(|end| *end <= 8).and_then(|end| {
            let name = &rest[1..end];
            let c = match name {
                "lt" => '<',
                "gt" => '>',
                "amp" => '&',
                "quot" => '"',
                "apos" => '\'',
                "nbsp" => ' ',
                _ => {
                    let number = name.strip_prefix('#')?;
                    let code = match number.strip_prefix(['x', 'X']) {
                        Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                        None => number.parse().ok()?,
                    };
                    char::from_u32(code)?
                }
            };
            Some((c, end))
        });
        match decoded {
            Some((c, end)) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Trailing spaces trimmed from every line, leading ones from every line
/// outside a fenced block (where they are code's indentation), and no more
/// than one blank line in a row outside one.
fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank = 0;
    let mut fenced = false;
    for line in text.lines() {
        let line = line.trim_end();
        if line.starts_with("```") {
            fenced = !fenced;
        }
        if !fenced && line.trim().is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        out.push_str(if fenced {
            line
        } else {
            line.trim_start_matches(' ')
        });
        out.push('\n');
    }
    out.trim_matches('\n').to_string() + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headings_code_and_entities_become_markdown_a_sectioner_reads() {
        let html = "<html><head><style>p{}</style></head><body>\n<h3><code>string.rep (s, n)</code></h3>\n<p>Returns a string that is the concatenation of <code>n</code> copies of <code>s</code> &amp; more &lt;done&gt;.</p>\n<pre>\nprint(string.rep(\"ab\", 2))\n</pre>\n<ul><li>one</li><li>two</li></ul><!-- gone --></body></html>";
        let md = html_to_markdown(html);
        assert!(md.starts_with("### `string.rep (s, n)`\n"), "{md}");
        assert!(
            md.contains(
                "Returns a string that is the concatenation of `n` copies of `s` & more <done>."
            ),
            "{md}"
        );
        assert!(
            md.contains("```\nprint(string.rep(\"ab\", 2))\n```"),
            "{md}"
        );
        assert!(md.contains("\n- one\n- two"), "{md}");
        assert!(!md.contains("gone") && !md.contains("p{}"), "{md}");
    }
}
