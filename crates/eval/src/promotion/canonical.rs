// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for promotion gates that check an
// examiner's signature in the learner's own process, for its clients. If your
// team needs expertise in tamper-evident evaluation evidence you can procure
// our services by sending an email to info@swedishembedded.com.

//! The examiner's canonical JSON bytes, the text its signature is a MAC of.
//!
//! The examiner writes a document's canonical form as compact JSON with
//! every object's keys sorted by their UTF-8 bytes, strings in UTF-8 with
//! only `"`, `\` and the control characters escaped (`\b`, `\t`, `\n`, `\f`,
//! `\r` by name, the others as `\u00xx` in lower-case hex; DEL and every
//! character above it written as is), integers as integers, and floats in
//! their shortest round-trip digits: positional with at least one fractional
//! digit (`0.000025`, `100.0`) when the decimal exponent is from -5 to 15,
//! scientific otherwise (`1e-7`, `1.5e+16`), the exponent signed when
//! positive. This is that form, from the document as parsed; a float that
//! does not round-trip or a non-finite number is refused, since the examiner
//! could not have written it.

use std::fmt::Write as _;

use serde_json::{Number, Value};

/// The canonical bytes of `value`; `None` for a number the examiner cannot
/// write (not finite).
#[must_use]
pub fn canonical_bytes(value: &Value) -> Option<Vec<u8>> {
    let mut out = String::new();
    write_value(&mut out, value)?;
    Some(out.into_bytes())
}

fn write_value(out: &mut String, value: &Value) -> Option<()> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => write_number(out, n)?,
        Value::String(s) => write_string(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(out, item)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(out, key);
                out.push(':');
                write_value(out, &map[key])?;
            }
            out.push('}');
        }
    }
    Some(())
}

fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_number(out: &mut String, number: &Number) -> Option<()> {
    if let Some(i) = number.as_i64() {
        let _ = write!(out, "{i}");
        return Some(());
    }
    if let Some(u) = number.as_u64() {
        let _ = write!(out, "{u}");
        return Some(());
    }
    let value = number.as_f64()?;
    if !value.is_finite() {
        return None;
    }
    out.push_str(&float_text(value));
    Some(())
}

/// A float as the examiner writes it (the module documentation states the
/// rule).
fn float_text(value: f64) -> String {
    // Rust's `{:e}` is the shortest round-trip digits in scientific form:
    // `-1.5e16`, `2.5e-5`, `0e0`.
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .unwrap_or_else(|| unreachable!("{{:e}} always writes an exponent: {scientific}"));
    let exponent: i32 = exponent
        .parse()
        .unwrap_or_else(|_| unreachable!("{{:e}} writes a decimal exponent: {scientific}"));
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if (-5..=15).contains(&exponent) {
        let point = exponent + 1;
        let text = if point <= 0 {
            format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
        } else {
            let point = point as usize;
            if digits.len() <= point {
                format!("{digits}{}.0", "0".repeat(point - digits.len()))
            } else {
                format!("{}.{}", &digits[..point], &digits[point..])
            }
        };
        format!("{sign}{text}")
    } else {
        let mantissa = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let exponent = if exponent > 0 {
            format!("+{exponent}")
        } else {
            exponent.to_string()
        };
        format!("{sign}{mantissa}e{exponent}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_are_written_as_the_examiner_writes_them() {
        // Each pair was printed by the examiner's own serialiser.
        for (value, text) in [
            (0.1, "0.1"),
            (1.0 / 3.0, "0.3333333333333333"),
            (1e-7, "1e-7"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (-0.05, "-0.05"),
            (123_456_789.123, "123456789.123"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (2.5e-5, "0.000025"),
            (1e-5, "0.00001"),
            (1e-6, "1e-6"),
            (1.234e-5, "0.00001234"),
            (1.2345e-6, "1.2345e-6"),
            (1e15, "1000000000000000.0"),
            (100.0, "100.0"),
            (1.2345678901234568e17, "1.2345678901234568e+17"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (0.0000152587890625, "0.0000152587890625"),
        ] {
            assert_eq!(float_text(value), text, "{value:e}");
        }
    }

    #[test]
    fn objects_are_sorted_by_key_bytes_and_strings_escaped_as_the_examiner_escapes_them() {
        let value = json!({"é": 1, "e": 2, "Z": 3, "a": {"y": [], "x": {}},
                           "s": "\u{1}\u{1f}\u{7f}\u{2028}/\u{8}\u{c}\r\n\t\"\\"});
        let text = String::from_utf8(canonical_bytes(&value).unwrap()).unwrap();
        assert_eq!(
            text,
            "{\"Z\":3,\"a\":{\"x\":{},\"y\":[]},\"e\":2,\"s\":\"\\u0001\\u001f\u{7f}\u{2028}/\\b\\f\\r\\n\\t\\\"\\\\\",\"\u{e9}\":1}"
        );
    }
}
