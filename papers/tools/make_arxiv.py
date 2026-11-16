#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible research pipelines for its
# clients. If your team needs expertise in turning a paper's build into a
# submission archive that compiles anywhere, you can procure our services by
# sending an email to info@swedishembedded.com.

"""Builds the paper's arXiv source archive and checks it the way arXiv would.

    make_arxiv.py build --paper DIR --build DIR --job NAME --out FILE.tar.gz
        Collects the main file and every file it \\input's (recursively), the generated files the TeX
        reads (from DIR/generated of the build) at the paths the source names, references.bib and the
        build's .bbl renamed after the main file; strips comments (licence lines kept); refuses names
        that are not plain ASCII without spaces; fails if the stripped text names a machine path, a
        home path, a user, a host, a token or an email other than the paper's contact address.
    make_arxiv.py check --archive FILE.tar.gz --main-log MAIN.log --work DIR
        Extracts the archive into fresh directories under DIR and builds it there with a clean
        environment (no TEXINPUTS, HOME in the directory): pdflatex, bibtex, pdflatex, pdflatex, and
        once with latexmk -pdf. Fails on any error, undefined reference or citation, a page count that
        differs from the main build's (read from MAIN.log), or an archive larger than SIZE_LIMIT.

arXiv (Submission 1.5, https://info.arxiv.org/help/submit_tex.html) compiles from the root of the
upload, provides only TeX Live's packages, uses a shipped .bbl whose name matches the main .tex file,
and asks for no auxiliary or output files in the upload.
"""

import argparse
import getpass
import gzip
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

MAIN = "paper.tex"
CONTACT = "info@swedishembedded.com"
# arXiv's help pages give no numeric limit; this bound is far below any it has applied.
SIZE_LIMIT = 10 * 1024 * 1024
LICENCE = re.compile(r"^%\s*(SPDX-License-Identifier:|Copyright \(c\))")
VERBATIM = re.compile(r"\\(begin|end)\{(verbatim\*?|lstlisting|Verbatim)\}")


def strip_comments(text):
    """Removes TeX comments: a line that is only a comment goes (licence lines stay), a trailing comment
    keeps its % (which also keeps the line join TeX saw). An escaped \\% is text; \\\\% starts a comment.
    Verbatim and lstlisting bodies are left as they are."""
    out, verbatim = [], False
    for line in text.splitlines(keepends=True):
        tag = VERBATIM.search(line)
        if verbatim:
            out.append(line)
            if tag and tag.group(1) == "end":
                verbatim = False
            continue
        if tag and tag.group(1) == "begin":
            verbatim = True
            out.append(line)
            continue
        if line.lstrip().startswith("%"):
            if LICENCE.match(line.lstrip()):
                out.append(line)
            continue
        cut = comment_start(line)
        out.append(line if cut is None else line[:cut + 1].rstrip("\n") + "\n")
    return "".join(out)


def comment_start(line):
    """Index of the % that starts a comment, or None: a % preceded by an odd run of backslashes is text."""
    for i, ch in enumerate(line):
        if ch == "%":
            run = 0
            while i - run - 1 >= 0 and line[i - run - 1] == "\\":
                run += 1
            if run % 2 == 0:
                return i
    return None


INPUT = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GENERATED = re.compile(r"\{(generated/[^}]+)\}")


def collect(root, main):
    """The .tex files the main file reads (itself first, then in reading order) and the generated
    files they name, from the comment-stripped text, so a commented-out input is not shipped."""
    tex, generated, todo = [], set(), [main]
    while todo:
        name = todo.pop(0)
        if name in tex:
            continue
        tex.append(name)
        text = strip_comments((root / name).read_text(encoding="utf-8"))
        for ref in GENERATED.findall(text):
            generated.add(ref if Path(ref).suffix else ref + ".tex")
        for ref in INPUT.findall(text):
            if ref.startswith("generated/"):
                continue
            todo.append(ref if ref.endswith(".tex") else ref + ".tex")
    return tex, sorted(generated)


LEAKS = [
    ("machine path", re.compile(r"/(home|nobackup|proj|tmp|scratch|Users|var/folders)/")),
    ("home path", re.compile(r"(?<![\w$])~/")),
    ("host name", re.compile(r"\bn\d{2,4}\b")),
    ("token", re.compile(r"\b(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
]


def leaks(text, allowed_emails, users):
    """Every (kind, match) in `text` that identifies a machine, a user, a host, a credential or an
    address other than the allowed ones."""
    found = []
    for kind, pattern in LEAKS:
        for m in pattern.finditer(text):
            if kind == "email" and m.group(0) in allowed_emails:
                continue
            found.append((kind, m.group(0)))
    for user in users:
        if user and re.search(USER_CONTEXT.format(user=re.escape(user)), text):
            found.append(("user name", user))
    return found


# A user name identifies someone only where it names an account: a path segment (`/name/`, `\\name\\`),
# after a tilde (`~name`) or before an at sign (`name@host`). As a bare word it is ordinary prose: an
# account called "user" must not make "the user" a leak.
USER_CONTEXT = r"(?:[/\\]{user}(?=[/\\]|\b)|~{user}\b|\b{user}@)"


def write_archive(path, files):
    """A deterministic .tar.gz of `files` (archive name -> bytes): sorted, no owner, time zero."""
    for name in files:
        if not name.isascii() or " " in name or name.startswith(("/", ".")) or ".." in name.split("/"):
            raise ValueError(f"archive name {name!r} is not plain ASCII without spaces")
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name in sorted(files):
            data = files[name]
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    with open(path, "wb") as out, gzip.GzipFile(fileobj=out, mode="wb", mtime=0, filename="") as gz:
        gz.write(raw.getvalue())


def build(paper, build_dir, job, out):
    tex, generated = collect(paper, MAIN)
    files = {}
    for name in tex:
        files[name] = strip_comments((paper / name).read_text(encoding="utf-8")).encode()
    for name in generated:
        src = build_dir / name
        if not src.is_file():
            sys.exit(f"{name} is read by the paper but missing from {build_dir}")
        data = src.read_bytes()
        files[name] = strip_comments(data.decode()).encode() if name.endswith(".tex") else data
    files["references.bib"] = (paper / "references.bib").read_bytes()
    bbl = build_dir / f"{job}.bbl"
    if not bbl.is_file():
        sys.exit(f"{bbl} is missing: build the PDF first")
    files[Path(MAIN).with_suffix(".bbl").name] = bbl.read_bytes()
    users = {getpass.getuser(), os.environ.get("USER", ""), Path.home().name}
    problems = [f"{name}: {kind} {hit!r}" for name, data in sorted(files.items())
                for kind, hit in leaks(data.decode(errors="replace"), {CONTACT}, users)]
    if problems:
        sys.exit("the archive would publish:\n  " + "\n  ".join(problems))
    write_archive(out, files)
    print(f"{out}: {len(files)} files, {out.stat().st_size} bytes")


def pages(log):
    m = re.search(r"Output written on .*?\((\d+) pages?", log, re.S)
    return int(m.group(1)) if m else None


def run(cmd, cwd):
    env = {"PATH": os.environ["PATH"], "HOME": str(cwd), "LANG": "C.UTF-8"}
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def check(archive, main_log, work):
    size = archive.stat().st_size
    if size > SIZE_LIMIT:
        sys.exit(f"{archive} is {size} bytes, above {SIZE_LIMIT}")
    with tarfile.open(archive) as t:
        names = t.getnames()
    bad = [n for n in names if not n.isascii() or " " in n]
    if bad:
        sys.exit(f"names arXiv may not accept: {bad}")
    main_pages = pages(main_log.read_text(errors="replace"))
    results = {}
    for way, steps in (("pdflatex+bibtex", [["pdflatex", "-interaction=nonstopmode", "-halt-on-error", MAIN],
                                             ["bibtex", Path(MAIN).stem],
                                             ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", MAIN],
                                             ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", MAIN]]),
                       ("latexmk", [["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", MAIN]])):
        d = work / way
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        with tarfile.open(archive) as t:
            t.extractall(d, filter="data")
        for cmd in steps:
            done = run(cmd, d)
            if done.returncode != 0:
                sys.exit(f"{way}: {' '.join(cmd)} failed in {d}:\n{done.stdout[-3000:]}")
        log = (d / Path(MAIN).with_suffix(".log").name).read_text(errors="replace")
        problems = [line for line in log.splitlines()
                    if re.search(r"undefined|^! ", line) and "Font Warning" not in line]
        if problems:
            sys.exit(f"{way}: " + "\n".join(problems[:20]))
        results[way] = pages(log)
    for way, n in results.items():
        if n != main_pages:
            sys.exit(f"{way}: {n} pages, the main PDF has {main_pages}")
    print(f"{archive}: {len(names)} files, {size} bytes; clean-room builds {results} match the main PDF ({main_pages} pages)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--paper", type=Path, required=True)
    b.add_argument("--build", type=Path, required=True)
    b.add_argument("--job", required=True)
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("--archive", type=Path, required=True)
    c.add_argument("--main-log", type=Path, required=True)
    c.add_argument("--work", type=Path, required=True)
    a = p.parse_args()
    if a.cmd == "build":
        build(a.paper.resolve(), a.build.resolve(), a.job, a.out.resolve())
    else:
        check(a.archive.resolve(), a.main_log.resolve(), a.work.resolve())


if __name__ == "__main__":
    main()
