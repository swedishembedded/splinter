// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A small repository with every kind of commit the sample must handle,
//! built from scratch (no network, no model) for the audit and the specs.
//!
//! The history, oldest first: a root commit; a file added; a function edited
//! (introducing a name no earlier text holds); a file renamed; a file deleted;
//! a three-file change; a change to a lock file and a source file; a binary
//! file; a merge of a side branch; a commit that holds a secret. The author
//! and the trailer are people to scrub.

use std::path::Path;

use anyhow::{bail, Result};

/// Name the fixture's author goes by; scrubbing must remove it.
pub const AUTHOR: &str = "Ada Lovelace";
/// The author's address; scrubbing must remove it.
pub const AUTHOR_EMAIL: &str = "ada@example.org";

/// The commit subjects in the order they were made on the first-parent chain,
/// with the label specs use to find them.
pub const COMMITS: &[(&str, &str)] = &[
    ("root", "Initial commit"),
    ("add", "Add a shapes module with a circle area function"),
    (
        "edit",
        "Charge tax in total() so prices include the tax rate",
    ),
    ("rename", "Rename util.py to helpers.py"),
    ("delete", "Remove the shapes module, nothing uses it"),
    ("multi", "Document the helpers and make clamp inclusive"),
    ("insert", "Add a subtract function between add and total"),
    ("config", "Add the retry configuration"),
    ("lock", "Raise the retry limit and add a backoff setting"),
    ("bump", "Update the lock file"),
    ("binary", "Add the project logo"),
    ("merge", "Merge branch 'side'"),
    ("secret", "Point the client at the staging service"),
];

fn git(dir: &Path, args: &[&str]) -> Result<String> {
    let out = crate::git::hermetic()
        .arg("-C")
        .arg(dir)
        .env("GIT_AUTHOR_NAME", AUTHOR)
        .env("GIT_AUTHOR_EMAIL", AUTHOR_EMAIL)
        .env("GIT_COMMITTER_NAME", AUTHOR)
        .env("GIT_COMMITTER_EMAIL", AUTHOR_EMAIL)
        .env("GIT_AUTHOR_DATE", "2026-01-01T00:00:00Z")
        .env("GIT_COMMITTER_DATE", "2026-01-01T00:00:00Z")
        .args(args)
        .output()?;
    if !out.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn write(dir: &Path, path: &str, text: &str) -> Result<()> {
    let file = dir.join(path);
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(file, text)?;
    Ok(())
}

fn commit(dir: &Path, subject: &str) -> Result<String> {
    git(dir, &["add", "-A"])?;
    let message = format!("{subject}\n\nSigned-off-by: {AUTHOR} <{AUTHOR_EMAIL}>");
    git(dir, &["commit", "--quiet", "-m", &message])?;
    git(dir, &["rev-parse", "HEAD"])
}

const CALC_V1: &str = "def add(a, b):\n    return a + b\n\n\ndef total(price, quantity):\n    return price * quantity\n";
const CALC_V2: &str = "def add(a, b):\n    return a + b\n\n\ndef total(price, quantity, rate):\n    subtotal = price * quantity\n    return compute_total_with_tax(subtotal, rate)\n\n\ndef compute_total_with_tax(subtotal, rate):\n    return subtotal + subtotal * rate\n";
const UTIL_V1: &str = "def clamp(value, low, high):\n    if value < low:\n        return low\n    if value > high:\n        return high\n    return value\n";
const SHAPES: &str =
    "import math\n\n\ndef circle_area(radius):\n    return math.pi * radius * radius\n";

/// Builds the fixture repository in `dir` (which must exist and be empty) and
/// returns the commit ids in [`COMMITS`] order.
pub fn build(dir: &Path) -> Result<Vec<String>> {
    git(dir, &["init", "--quiet", "--initial-branch=main"])?;
    let mut ids = Vec::new();

    write(dir, "README.md", "# Demo\n\nA tiny arithmetic library.\n")?;
    write(dir, "calc.py", CALC_V1)?;
    write(dir, "util.py", UTIL_V1)?;
    ids.push(commit(dir, COMMITS[0].1)?);

    write(dir, "shapes.py", SHAPES)?;
    ids.push(commit(dir, COMMITS[1].1)?);

    write(dir, "calc.py", CALC_V2)?;
    ids.push(commit(dir, COMMITS[2].1)?);

    git(dir, &["mv", "util.py", "helpers.py"])?;
    ids.push(commit(dir, COMMITS[3].1)?);

    git(dir, &["rm", "--quiet", "shapes.py"])?;
    ids.push(commit(dir, COMMITS[4].1)?);

    write(
        dir,
        "helpers.py",
        "def clamp(value, low, high):\n    if value < low:\n        return low\n    if value >= high:\n        return high\n    return value\n\n\ndef lerp(start, stop, amount):\n    return start + (stop - start) * amount\n",
    )?;
    write(dir, "notes/helpers.md", "# Helpers\n\nclamp keeps a value inside a range, and lerp interpolates between two values.\n")?;
    write(
        dir,
        "README.md",
        "# Demo\n\nA tiny arithmetic library with range helpers, see notes/helpers.md.\n",
    )?;
    ids.push(commit(dir, COMMITS[5].1)?);

    let lock = |version: &str| {
        format!("{{\n  \"lockfileVersion\": 3,\n  \"packages\": {{\n    \"\": {{}},\n    \"node_modules/left-pad\": {{\n      \"version\": \"{version}\",\n      \"resolved\": \"left-pad\"\n    }}\n  }}\n}}\n")
    };
    // An insertion only: no line is removed, and the blank lines around it occur everywhere.
    let inserted = std::fs::read_to_string(dir.join("calc.py"))?.replacen(
        "\n\n\ndef total",
        "\n\n\ndef sub(a, b):\n    return a - b\n\n\ndef total",
        1,
    );
    write(dir, "calc.py", &inserted)?;
    ids.push(commit(dir, COMMITS[6].1)?);

    // Fourteen settings, so the lock file is a minority of the commit's lines.
    let settings: String = (1..=12).map(|n| format!("SETTING_{n} = {n}\n")).collect();
    write(
        dir,
        "config.py",
        &format!("RETRY_LIMIT = 3\nTIMEOUT_SECONDS = 30\n{settings}"),
    )?;
    write(dir, "package-lock.json", &lock("1.3.0"))?;
    ids.push(commit(dir, COMMITS[7].1)?);

    write(dir, "config.py", &format!("RETRY_LIMIT = 5\nTIMEOUT_SECONDS = 30\nBACKOFF_SECONDS = 2\nBACKOFF_FACTOR = 3\n{settings}"))?;
    write(dir, "package-lock.json", &lock("1.3.1"))?;
    ids.push(commit(dir, COMMITS[8].1)?);

    write(dir, "package-lock.json", &lock("1.3.2"))?;
    ids.push(commit(dir, COMMITS[9].1)?);

    std::fs::write(
        dir.join("logo.png"),
        [0x89, b'P', b'N', b'G', 0, 1, 2, 3, 0xff, 0xfe],
    )?;
    ids.push(commit(dir, COMMITS[10].1)?);

    git(dir, &["checkout", "--quiet", "-b", "side"])?;
    write(dir, "side.txt", "work done on a branch\n")?;
    git(dir, &["add", "-A"])?;
    git(dir, &["commit", "--quiet", "-m", "Work on the side branch"])?;
    git(dir, &["checkout", "--quiet", "main"])?;
    git(
        dir,
        &["merge", "--quiet", "--no-ff", "-m", COMMITS[11].1, "side"],
    )?;
    ids.push(git(dir, &["rev-parse", "HEAD"])?);

    write(
        dir,
        "client.py",
        "ENDPOINT = \"https://staging.example.org\"\nAPI_TOKEN = \"abcd1234efgh5678ijkl\"\n",
    )?;
    ids.push(commit(dir, COMMITS[12].1)?);
    Ok(ids)
}
