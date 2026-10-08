// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One NHANES cycle's files, read with Splinter's transport reader, and the
//! public-use linked mortality file.
//!
//! A variable is looked up by name in whichever of the cycle's files holds it
//! with one row per participant (`SEQN`); a file that repeats `SEQN` (the
//! individual foods, prescriptions, activity days) is kept aside for the
//! readers that know its shape. Files are visited in name order, so a
//! variable present in two files resolves the same way on every run.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use splinter_sdk::knowledge::tabular::{read_xport, Table, Values};
use splinter_sdk::vocabulary::digest::Digest;

/// The ten continuous cycles with public mortality linkage: start year and
/// file suffix.
pub const CYCLES: [(u16, &str); 10] = [
    (1999, ""),
    (2001, "_B"),
    (2003, "_C"),
    (2005, "_D"),
    (2007, "_E"),
    (2009, "_F"),
    (2011, "_G"),
    (2013, "_H"),
    (2015, "_I"),
    (2017, "_J"),
];

/// One participant's row in one file.
type Row = (usize, usize); // (table, row)

/// A cycle's data, indexed by participant.
pub struct Cycle {
    /// Start year of the cycle.
    pub start: u16,
    tables: Vec<Table>,
    /// variable -> table holding it once per participant.
    var_table: HashMap<String, usize>,
    /// table -> SEQN -> row.
    rows: Vec<HashMap<u64, usize>>,
    /// Multi-row tables by file stem (upper case, suffix removed).
    pub multi: BTreeMap<String, Table>,
    /// `(file name, digest)` of every file read.
    pub sources: Vec<(String, Digest)>,
}

fn seqns(t: &Table) -> Option<Vec<u64>> {
    t.numeric("SEQN")
        .map(|v| v.iter().map(|x| x.map_or(0, |s| s as u64)).collect())
}

impl Cycle {
    /// Read every transport file of the cycle starting `start` under `root`
    /// (`root/<start>/*.xpt`).
    pub fn load(root: &Path, start: u16, suffix: &str) -> Result<Cycle> {
        let dir = root.join(start.to_string());
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("xpt")))
            .collect();
        files.sort();
        let mut c = Cycle {
            start,
            tables: vec![],
            var_table: HashMap::new(),
            rows: vec![],
            multi: BTreeMap::new(),
            sources: vec![],
        };
        for path in files {
            let bytes =
                std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            c.sources.push((name.clone(), Digest::of(&bytes)));
            let table =
                read_xport(&bytes).with_context(|| format!("parsing {}", path.display()))?;
            let Some(ids) = seqns(&table) else { continue };
            let mut index = HashMap::with_capacity(ids.len());
            let unique = ids
                .iter()
                .enumerate()
                .all(|(r, s)| index.insert(*s, r).is_none());
            if !unique {
                let stem = name
                    .trim_end_matches(".xpt")
                    .trim_end_matches(".XPT")
                    .to_uppercase();
                let stem = stem
                    .strip_suffix(&suffix.to_uppercase())
                    .unwrap_or(&stem)
                    .to_string();
                c.multi.insert(stem, table);
                continue;
            }
            let t = c.tables.len();
            for col in &table.columns {
                c.var_table.entry(col.name.clone()).or_insert(t);
            }
            c.rows.push(index);
            c.tables.push(table);
        }
        if !c.var_table.contains_key("RIDAGEYR") {
            bail!(
                "cycle {start}: no demographics file (RIDAGEYR) under {}",
                dir.display()
            );
        }
        Ok(c)
    }

    fn cell(&self, var: &str, seqn: u64) -> Option<Row> {
        let t = *self.var_table.get(var)?;
        Some((t, *self.rows[t].get(&seqn)?))
    }

    /// The numeric value of `var` for participant `seqn`.
    pub fn num(&self, var: &str, seqn: u64) -> Option<f64> {
        let (t, r) = self.cell(var, seqn)?;
        match &self.tables[t].column(var)?.values {
            Values::Numeric(v) => v[r],
            Values::Text(_) => None,
        }
    }

    /// Every participant's value of `var`, in file order; `None` when no
    /// one-row-per-participant file holds it as a number.
    pub fn column(&self, var: &str) -> Option<&[Option<f64>]> {
        self.tables[*self.var_table.get(var)?].numeric(var)
    }

    /// `var` from the first row of each participant in the multi-row table
    /// `stem` (for a per-person value a multi-row file repeats on every row,
    /// such as the prescription count).
    pub fn first_rows(&self, stem: &str, var: &str) -> HashMap<u64, f64> {
        let mut out = HashMap::new();
        let Some(t) = self.multi.get(stem) else {
            return out;
        };
        let (Some(ids), Some(vals)) = (t.numeric("SEQN"), t.numeric(var)) else {
            return out;
        };
        for (id, v) in ids.iter().zip(vals) {
            if let (Some(id), Some(v)) = (id, v) {
                out.entry(*id as u64).or_insert(*v);
            }
        }
        out
    }

    /// Every participant in the demographics file.
    pub fn participants(&self) -> Vec<u64> {
        let t = self.var_table["RIDAGEYR"];
        let mut ids: Vec<u64> = self.rows[t].keys().copied().collect();
        ids.sort_unstable();
        ids
    }
}

/// The public-use linked mortality file of the cycle starting `start`.
pub fn mortality_file(dir: &Path, start: u16) -> PathBuf {
    dir.join(format!(
        "NHANES_{}_{}_MORT_2019_PUBLIC.dat",
        start,
        start + 1
    ))
}

/// A participant's public-use linked mortality record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mortality {
    /// 1 eligible for linkage, 2 under 18, 3 ineligible.
    pub eligible: bool,
    /// Died by the end of follow-up.
    pub died: bool,
    /// Leading underlying cause recode (1-10), when died.
    pub cause: Option<u16>,
    /// Months from the examination to death or the end of follow-up.
    pub months_from_exam: Option<f64>,
    /// Diabetes listed anywhere on the death certificate (the file's
    /// multiple-cause flag); `None` when the multiple-cause data are not
    /// available or the person is not a decedent.
    pub diabetes_mcod: Option<bool>,
    /// Hypertension listed anywhere on the death certificate; as above.
    pub hypertension_mcod: Option<bool>,
}

/// Read a cycle's public-use linked mortality file (fixed width, the layout
/// of NCHS's own read-in programs: SEQN 1-6, ELIGSTAT 15, MORTSTAT 16,
/// UCOD_LEADING 17-19, DIABETES 20, HYPERTEN 21, PERMTH_EXM 46-48).
pub fn mortality(path: &Path) -> Result<(HashMap<u64, Mortality>, Digest)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let digest = Digest::of(&bytes);
    let text = String::from_utf8_lossy(&bytes);
    // Trailing blanks are trimmed in the files, so a field at the end of a
    // line may be shorter than its columns: take what is there.
    let field = |line: &str, a: usize, b: usize| {
        line.get(a - 1..b.min(line.len()))
            .map(str::trim)
            .filter(|s| !s.is_empty() && *s != ".")
            .map(str::to_string)
    };
    let mut out = HashMap::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let seqn: u64 = field(line, 1, 6)
            .and_then(|s| s.parse().ok())
            .with_context(|| format!("{}:{}: SEQN", path.display(), n + 1))?;
        let num = |a, b| field(line, a, b).and_then(|s| s.parse::<f64>().ok());
        out.insert(
            seqn,
            Mortality {
                eligible: num(15, 15) == Some(1.0),
                died: num(16, 16) == Some(1.0),
                cause: num(17, 19).map(|c| c as u16),
                months_from_exam: num(46, 48),
                diabetes_mcod: num(20, 20).map(|f| f == 1.0),
                hypertension_mcod: num(21, 21).map(|f| f == 1.0),
            },
        );
    }
    Ok((out, digest))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mortality_line_is_read_by_column() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.dat");
        let alive = format!(
            "{:<6}{:8}1{:1}{:3}{:1}{:1}{:21}{:3}{:3}",
            "123", "", "0", "", "", "", "", "200", "201"
        );
        let dead = format!(
            "{:<6}{:8}1{:1}{:3}{:1}{:1}{:21}{:3}{:3}",
            "124", "", "1", "001", "0", "1", "", "50", "48"
        );
        let no_mcod = format!(
            "{:<6}{:8}1{:1}{:3}{:1}{:1}{:21}{:3}{:3}",
            "126", "", "1", "010", ".", ".", "", "50", "30"
        );
        let minor = format!("{:<6}{:8}2{:1}", "125", "", ".");
        // A real line from the 2013-2014 file: the follow-up fields are "10"
        // and "9", so the line ends at column 47, before PERMTH_EXM's last column.
        let short = "73561         1101000                     10 9";
        std::fs::write(
            &path,
            format!("{alive}\n{dead}\n{no_mcod}\n{minor}\r\n{short}\r\n"),
        )
        .unwrap();
        let (m, _) = mortality(&path).unwrap();
        assert_eq!(
            m[&123],
            Mortality {
                eligible: true,
                died: false,
                cause: None,
                months_from_exam: Some(201.0),
                diabetes_mcod: None,
                hypertension_mcod: None,
            }
        );
        assert_eq!(
            m[&124],
            Mortality {
                eligible: true,
                died: true,
                cause: Some(1),
                months_from_exam: Some(48.0),
                diabetes_mcod: Some(false),
                hypertension_mcod: Some(true),
            }
        );
        assert!(!m[&125].eligible);
        assert_eq!(
            (
                m[&126].died,
                m[&126].diabetes_mcod,
                m[&126].hypertension_mcod
            ),
            (true, None, None),
            "a death without multiple-cause data has no flags, not false ones"
        );
        assert_eq!(
            m[&73561],
            Mortality {
                eligible: true,
                died: true,
                cause: Some(10),
                months_from_exam: Some(9.0),
                diabetes_mcod: Some(false),
                hypertension_mcod: Some(false),
            },
            "a short line keeps its last field"
        );
    }
}
