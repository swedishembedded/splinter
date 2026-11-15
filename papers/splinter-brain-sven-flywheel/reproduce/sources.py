#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements pinned, checksummed inputs for model training
# runs, for its clients. If your team needs expertise in making a training run
# start from exactly the bytes it started from before, you can procure our
# services by sending an email to info@swedishembedded.com.

"""The public inputs the S4 runs started from, fetched and checked against the digests they had.

    sources.py qwen   --dir DIR    Qwen/Qwen3-4B at its pinned revision into DIR/Qwen/Qwen3-4B
    sources.py manual --dir DIR    the Lua 5.1 reference manual and the three sections the proposer
                                   reads (5.4 string, 5.5 table, 5.6 math) as DIR/lua51-manual-5.N.html

A file already present with its digest is not downloaded again; a download that does not have its
digest is removed and the command fails. The Lua 5.1.5 interpreter is built by the repository's
`make deps/lua` from the release tarball, whose digest that target checks.
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

QWEN_REPO = "Qwen/Qwen3-4B"
QWEN_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
QWEN_FILES = {
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "README.md": "71add1cd091c309b1cb1a6b943f2476eb2b004124b3bd1d32dcd0b24cc1abc47",
    "config.json": "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors.index.json": "6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2",
    "model-00001-of-00003.safetensors": "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
    "model-00002-of-00003.safetensors": "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
    "model-00003-of-00003.safetensors": "e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
# The digest every candidate record declares for this checkpoint (the learner's naming rule).
QWEN_BASE_DIGEST = "sha256:d3e8e289bdd0f5fcc755475a011818cb8be4fb9e07d49f4287786c85d220faa5"

MANUAL_URL = "https://www.lua.org/manual/5.1/manual.html"
MANUAL_SHA256 = "e754f09a38a45af243b80740ffc66c5147548a731a4239165b06586f2f54c28a"
SECTIONS = {
    "5.4": "860623e314aada081990ba3b61de2d8002ae0faff0b662efa3127af5c8c7de3b",
    "5.5": "501e70d419acf42f3f580385cc7b46d7ba18af646eeedc2516cf2939ba450f12",
    "5.6": "f540e0333ed7cb22caab7f6646c93a724609c04a0f82d2756705962b79f35c3f",
}
# Each section file is the manual's head, the section from its <h2> to the next <h2> with the
# heading dashes written as " - ", and the closing tags.
SECTION_HEAD = (b'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">\n<html>\n\n<head>\n'
                b"<title>Lua 5.1 Reference Manual</title>\n"
                b'<link rel="stylesheet" type="text/css" href="lua.css">\n'
                b'<link rel="stylesheet" type="text/css" href="manual.css">\n'
                b'<META HTTP-EQUIV="content-type" CONTENT="text/html; charset=iso-8859-1">\n</head>\n\n<body>\n')
SECTION_TAIL = b"</body>\n</html>\n"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest, digest):
    """Downloads `url` to `dest` unless `dest` already has `digest`; fails on any other digest."""
    if dest.is_file() and sha256_file(dest) == digest:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"downloading {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "paper-sources-fetch"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, 1 << 24)
    if sha256_file(tmp) != digest:
        tmp.unlink()
        raise SystemExit(f"sources: {url} does not have its pinned digest {digest}")
    tmp.replace(dest)


def manual_section(manual, number):
    """The proposer's file for manual section `number` (5.4, 5.5 or 5.6), cut from the manual."""
    start = manual.find(f"<h2>{number} &ndash; ".encode())
    if start < 0:
        raise ValueError(f"the manual has no section {number}")
    end = manual.find(b"<h2>", start + 4)
    return SECTION_HEAD + manual[start:end].replace(b" &ndash; ", b" - ") + SECTION_TAIL


def qwen(args):
    root = Path(args.dir) / QWEN_REPO
    for name, digest in QWEN_FILES.items():
        fetch(f"https://huggingface.co/{QWEN_REPO}/resolve/{QWEN_REVISION}/{name}", root / name, digest)
    print(root)


def manual(args):
    root = Path(args.dir)
    fetch(MANUAL_URL, root / "manual-5.1.html", MANUAL_SHA256)
    text = (root / "manual-5.1.html").read_bytes()
    for number, digest in SECTIONS.items():
        data = manual_section(text, number)
        if hashlib.sha256(data).hexdigest() != digest:
            raise SystemExit(f"sources: section {number} cut from the manual is not the file the runs read")
        (root / f"lua51-manual-{number}.html").write_bytes(data)
    print(root)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("qwen", "manual"):
        sub.add_parser(name).add_argument("--dir", required=True, type=Path)
    args = p.parse_args(argv)
    {"qwen": qwen, "manual": manual}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
