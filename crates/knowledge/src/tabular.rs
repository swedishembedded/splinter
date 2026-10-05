// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Tabular record files as columns: the SAS transport format (XPORT version
//! 5, `.xpt`), the format public survey and cohort data are distributed in.
//!
//! The format is a sequence of 80-byte records: a library header, a member
//! header, one 140-byte NAMESTR per variable (type, length, name, label,
//! offset within a row), then the rows, fixed width, padded with blanks to a
//! record boundary. Numbers are IBM System/360 base-16 floats, big-endian,
//! possibly truncated to fewer than eight bytes; a missing value is a float
//! whose first byte is `.`, `_` or a letter and the rest zero.
//!
//! Counting rows is the one subtle step: the padding after the last row is
//! blanks, and a last row whose final field is blank text ends in blanks too.
//! Stripping trailing blanks before dividing by the row width drops that row
//! (a reader that does so is wrong on exactly such files); here a whole row
//! is treated as padding only when it is entirely blank AND the file length
//! is consistent with it being padding.

use thiserror::Error;

const CARD: usize = 80;
const NAMESTR_LEN: usize = 140;
const MEMBER_TAG: &[u8] = b"HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!";
const NAMESTR_TAG: &[u8] = b"HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!";
const OBS_TAG: &[u8] = b"HEADER RECORD*******OBS     HEADER RECORD!!!!!!!";
const LIBRARY_TAG: &[u8] = b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!";

/// A transport file that cannot be read, and where.
#[derive(Debug, Error, PartialEq)]
pub enum XportError {
    /// Not a SAS transport (version 5) file.
    #[error("not a SAS transport file: {0}")]
    NotXport(String),
    /// A structural field is out of range or inconsistent.
    #[error("malformed transport file: {0}")]
    Malformed(String),
    /// More than one member (dataset) in one file.
    #[error("the file holds more than one member; one dataset per file is supported")]
    MultipleMembers,
}

/// One column's values.
#[derive(Clone, Debug, PartialEq)]
pub enum Values {
    /// Numbers; `None` for every kind of SAS missing value.
    Numeric(Vec<Option<f64>>),
    /// Text, trailing blanks removed.
    Text(Vec<String>),
}

/// One variable.
#[derive(Clone, Debug, PartialEq)]
pub struct Column {
    /// Variable name.
    pub name: String,
    /// Variable label (may be empty).
    pub label: String,
    /// Its values, one per row.
    pub values: Values,
}

/// One dataset.
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    /// The member (dataset) name.
    pub name: String,
    /// The columns, in file order.
    pub columns: Vec<Column>,
    /// Number of rows.
    pub rows: usize,
}

impl Table {
    /// The column named `name`.
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// The numeric values of column `name`, if it exists and is numeric.
    pub fn numeric(&self, name: &str) -> Option<&[Option<f64>]> {
        match &self.column(name)?.values {
            Values::Numeric(v) => Some(v),
            Values::Text(_) => None,
        }
    }
}

struct Var {
    numeric: bool,
    len: usize,
    pos: usize,
    name: String,
    label: String,
}

fn text(bytes: &[u8]) -> String {
    // Latin-1: every byte is the code point of the same value.
    let s: String = bytes.iter().map(|&b| b as char).collect();
    s.trim_end_matches([' ', '\0']).to_string()
}

fn be16(b: &[u8]) -> usize {
    u16::from_be_bytes([b[0], b[1]]) as usize
}

fn be32(b: &[u8]) -> usize {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize
}

/// An IBM base-16 float of `bytes.len()` (2..=8) bytes, or `None` for a SAS
/// missing value.
pub fn ibm_float(bytes: &[u8]) -> Option<f64> {
    let mut b = [0u8; 8];
    b[..bytes.len()].copy_from_slice(bytes);
    if b[1..].iter().all(|&x| x == 0) && (b[0] == b'.' || b[0] == b'_' || b[0].is_ascii_uppercase())
    {
        return None;
    }
    let sign = if b[0] & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = (b[0] & 0x7f) as i32 - 64;
    let mut fraction: u64 = 0;
    for &x in &b[1..] {
        fraction = (fraction << 8) | x as u64;
    }
    if fraction == 0 {
        return Some(0.0);
    }
    // value = 0.fraction (56 bits, base 2) * 16^exponent
    Some(sign * fraction as f64 * 2f64.powi(4 * exponent - 56))
}

fn find_card(data: &[u8], tag: &[u8], from: usize) -> Option<usize> {
    (from..data.len())
        .step_by(CARD)
        .find(|&off| data[off..].starts_with(tag))
}

/// Read a SAS transport (version 5) file holding one dataset.
pub fn read_xport(data: &[u8]) -> Result<Table, XportError> {
    if !data.starts_with(LIBRARY_TAG) {
        return Err(XportError::NotXport(
            "missing the library header record".into(),
        ));
    }
    let member = find_card(data, MEMBER_TAG, 0)
        .ok_or_else(|| XportError::NotXport("missing the member header record".into()))?;
    let namestr_len: usize = text(&data[member + 74..member + 78])
        .trim()
        .parse()
        .map_err(|_| XportError::Malformed("member header: NAMESTR length".into()))?;
    if namestr_len != NAMESTR_LEN && namestr_len != 136 {
        return Err(XportError::Malformed(format!(
            "NAMESTR length {namestr_len}"
        )));
    }
    // Descriptor: header card, then two cards; the first names the dataset.
    let descriptor = member + CARD;
    let name = text(
        data.get(descriptor + CARD + 8..descriptor + CARD + 16)
            .ok_or_else(|| XportError::Malformed("truncated descriptor".into()))?,
    );
    let nameheader = find_card(data, NAMESTR_TAG, member)
        .ok_or_else(|| XportError::Malformed("missing the NAMESTR header record".into()))?;
    let nvars: usize = text(&data[nameheader + 54..nameheader + 58])
        .trim()
        .parse()
        .map_err(|_| XportError::Malformed("NAMESTR header: variable count".into()))?;
    let first = nameheader + CARD;
    let mut vars = Vec::with_capacity(nvars);
    for v in 0..nvars {
        let off = first + v * namestr_len;
        let ns = data
            .get(off..off + namestr_len)
            .ok_or_else(|| XportError::Malformed(format!("truncated NAMESTR {v}")))?;
        let numeric = match be16(&ns[0..2]) {
            1 => true,
            2 => false,
            t => return Err(XportError::Malformed(format!("variable {v}: type {t}"))),
        };
        let len = be16(&ns[4..6]);
        if len == 0 || (numeric && !(2..=8).contains(&len)) {
            return Err(XportError::Malformed(format!("variable {v}: length {len}")));
        }
        vars.push(Var {
            numeric,
            len,
            pos: be32(&ns[84..88]),
            name: text(&ns[8..16]),
            label: text(&ns[16..56]),
        });
    }
    // The NAMESTRs are padded to a whole record; the observation header is
    // the first record after them.
    let names_end = (first + nvars * namestr_len) / CARD * CARD;
    let obs_header = find_card(data, OBS_TAG, names_end)
        .ok_or_else(|| XportError::Malformed("missing the observation header record".into()))?;
    let rows_start = obs_header + CARD;
    let width: usize = vars.iter().map(|v| v.len).sum();
    if vars.iter().any(|v| v.pos + v.len > width) {
        return Err(XportError::Malformed(
            "a variable lies outside the row".into(),
        ));
    }
    let body = &data[rows_start..];
    if find_card(body, MEMBER_TAG, 0).is_some() {
        return Err(XportError::MultipleMembers);
    }
    let mut n = body.len().checked_div(width).unwrap_or(0);
    // A trailing row of blanks is padding when the file length still fits.
    let padded = |rows: usize| (rows * width).div_ceil(CARD) * CARD;
    while n > 0
        && padded(n - 1) == body.len()
        && body[(n - 1) * width..n * width].iter().all(|&b| b == b' ')
    {
        n -= 1;
    }
    let columns = vars
        .iter()
        .map(|v| {
            let cell = |r: usize| &body[r * width + v.pos..r * width + v.pos + v.len];
            let values = if v.numeric {
                Values::Numeric((0..n).map(|r| ibm_float(cell(r))).collect())
            } else {
                Values::Text((0..n).map(|r| text(cell(r))).collect())
            };
            Column {
                name: v.name.clone(),
                label: v.label.clone(),
                values,
            }
        })
        .collect();
    Ok(Table {
        name,
        columns,
        rows: n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ibm_floats_and_missing_values() {
        // 1.0 = 0x41100000 00000000; -118.625 = 0xC276A000...
        assert_eq!(ibm_float(&[0x41, 0x10, 0, 0, 0, 0, 0, 0]), Some(1.0));
        assert_eq!(
            ibm_float(&[0xC2, 0x76, 0xA0, 0, 0, 0, 0, 0]),
            Some(-118.625)
        );
        assert_eq!(
            ibm_float(&[0x41, 0x10, 0, 0]),
            Some(1.0),
            "a truncated float"
        );
        assert_eq!(ibm_float(&[0; 8]), Some(0.0));
        assert_eq!(ibm_float(&[b'.', 0, 0, 0, 0, 0, 0, 0]), None);
        assert_eq!(
            ibm_float(&[b'A', 0, 0, 0, 0, 0, 0, 0]),
            None,
            "special missing .A"
        );
        assert_eq!(ibm_float(&[b'_', 0, 0, 0, 0, 0, 0, 0]), None);
    }

    #[test]
    fn not_a_transport_file() {
        assert!(matches!(read_xport(b"hello"), Err(XportError::NotXport(_))));
    }
}
