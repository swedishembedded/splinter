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

/// The text of an HTML page's bytes, decoded as the page declares: UTF-8 when
/// the bytes are UTF-8; otherwise the `charset` its head declares, where the
/// Latin labels (`iso-8859-1`, `latin1`, `us-ascii`, `windows-1252`, ...) and
/// an undeclared charset decode as windows-1252, the superset browsers use for
/// all of them. Refused, naming the charset, when the page declares one this
/// does not decode or declares UTF-8 and is not.
pub fn decode_html(bytes: &[u8]) -> Result<String, String> {
    if let Ok(text) = std::str::from_utf8(bytes) {
        return Ok(text.to_string());
    }
    match declared_charset(bytes).as_deref() {
        None => Ok(windows_1252(bytes)),
        Some(label) if LATIN_LABELS.contains(&label) => Ok(windows_1252(bytes)),
        Some(label) => Err(format!(
            "the page declares charset {label:?} and its bytes are not text in it that Splinter decodes"
        )),
    }
}

/// The labels that name windows-1252 or a subset of it.
const LATIN_LABELS: &[&str] = &[
    "iso-8859-1",
    "iso8859-1",
    "iso_8859-1",
    "latin1",
    "l1",
    "us-ascii",
    "ascii",
    "windows-1252",
    "cp1252",
    "x-cp1252",
];

/// The `charset` the head of the page declares, lowercased.
fn declared_charset(bytes: &[u8]) -> Option<String> {
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(4096)]).to_ascii_lowercase();
    let at = head.find("charset=")? + "charset=".len();
    let label: String = head[at..]
        .trim_start_matches(['"', '\''])
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'))
        .collect();
    (!label.is_empty()).then_some(label)
}

/// `bytes` as windows-1252: Latin-1, except 0x80-0x9F, which hold the
/// typographic characters; the five bytes it leaves undefined map to the C1
/// controls of the same value, as browsers decode them.
fn windows_1252(bytes: &[u8]) -> String {
    const HIGH: [u32; 32] = [
        0x20AC, 0x81, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160,
        0x2039, 0x0152, 0x8D, 0x017D, 0x8F, 0x90, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013,
        0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x9D, 0x017E, 0x0178,
    ];
    bytes
        .iter()
        .map(|&b| match b {
            0x80..=0x9F => char::from_u32(HIGH[usize::from(b - 0x80)]).unwrap_or('\u{FFFD}'),
            _ => char::from(b),
        })
        .collect()
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
    #[test]
    fn a_latin_1_page_decodes_as_it_declares_and_a_utf8_page_as_it_is() {
        let latin = b"<html><head><META HTTP-EQUIV=\"content-type\" CONTENT=\"text/html; charset=iso-8859-1\"></head><body>Copyright \xa9 Lua.org, PUC-Rio \x93q\x94</body></html>";
        let text = decode_html(latin).expect("latin-1 decodes");
        assert!(text.contains("Copyright \u{a9} Lua.org"), "{text}");
        assert!(text.contains("\u{201c}q\u{201d}"), "{text}");
        let undeclared = b"<html><body>caf\xe9</body></html>";
        assert_eq!(
            decode_html(undeclared).unwrap(),
            "<html><body>caf\u{e9}</body></html>"
        );
        let utf8 = "<html><meta charset=\"utf-8\"><body>caf\u{e9}</body></html>";
        assert_eq!(decode_html(utf8.as_bytes()).unwrap(), utf8);
    }

    #[test]
    fn a_charset_it_does_not_decode_is_refused_by_name() {
        let sjis = b"<html><meta charset=\"shift_jis\"><body>\x82\xa0</body></html>";
        let refused = decode_html(sjis).unwrap_err();
        assert!(refused.contains("shift_jis"), "{refused}");
        let lying = b"<html><meta charset=utf-8><body>caf\xe9</body></html>";
        assert!(decode_html(lying).unwrap_err().contains("utf-8"));
    }
}
