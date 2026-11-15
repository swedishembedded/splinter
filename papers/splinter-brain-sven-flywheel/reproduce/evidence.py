#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements research records that a reader can download and
# check byte for byte, for its clients. If your team needs expertise in publishing
# evidence whose every digest is verified before it is used, you can procure our
# services by sending an email to info@swedishembedded.com.

"""The paper's evidence dataset: download it at the pinned revision, check every file and archive
member against the manifest, and unpack the archives.

    evidence.py fetch  --dir DIR [--only PATH ...] [--token-file FILE]
    evidence.py check  --dir DIR [--only PATH ...]
    evidence.py manifest --dir DIR --repo REPO --revision SHA   (writes reproduce/dataset.json)

`fetch` downloads each file the manifest lists (or the ones `--only` names) that DIR does not already
hold with its digest, checks it, and unpacks each archive into DIR, refusing a member whose path
leaves DIR or whose digest differs. `check` re-checks DIR without the network, and checks that every
candidate record's adapter has the digest the record declares. A file is downloaded from the
dataset's `resolve` endpoint at the manifest's revision; a private dataset needs an access token,
read from --token-file (default: the Hugging Face client's token file, when it exists).
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

MANIFEST = Path(__file__).resolve().parent / "dataset.json"
DEFAULT_TOKEN = Path.home() / ".cache" / "huggingface" / "token"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path=MANIFEST):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selected(manifest, only):
    files = manifest["files"]
    if not only:
        return files
    names = {f["path"] for f in files}
    unknown = [o for o in only if not any(n == o or n.startswith(o.rstrip("/") + "/") for n in names)]
    if unknown:
        raise SystemExit(f"evidence: the manifest lists no {unknown}")
    return [f for f in files if any(f["path"] == o or f["path"].startswith(o.rstrip("/") + "/") for o in only)]


def safe_member(name):
    """A member name that stays inside the directory it is unpacked into."""
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"archive member {name!r} leaves the directory")
    return p


def unpack(archive, dest, members):
    """Unpacks the regular-file members of `archive` into `dest`, each checked against its digest in
    `members`; any other member type, or a member the manifest does not list, is refused."""
    seen = set()
    with tarfile.open(archive, "r:xz") as tar:
        for info in tar:
            if info.isdir():
                continue
            if not info.isfile():
                raise ValueError(f"{archive.name}: member {info.name!r} is not a regular file")
            rel = safe_member(info.name)
            want = members.get(info.name)
            if want is None:
                raise ValueError(f"{archive.name}: member {info.name!r} is not in the manifest")
            data = tar.extractfile(info).read()
            if hashlib.sha256(data).hexdigest() != want:
                raise ValueError(f"{archive.name}: member {info.name} differs from the manifest")
            out = dest.joinpath(*rel.parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(out)
            seen.add(info.name)
    missing = sorted(set(members) - seen)
    if missing:
        raise ValueError(f"{archive.name}: lacks {missing[:3]}")


def download(repo, revision, path, dest, token):
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
    headers = {"User-Agent": "paper-evidence-fetch"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    tmp = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600) as resp, \
            open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, 1 << 24)
    tmp.replace(dest)


def fetch(args):
    manifest = load_manifest()
    token = None
    token_file = args.token_file or (DEFAULT_TOKEN if DEFAULT_TOKEN.is_file() else None)
    if token_file:
        token = Path(token_file).read_text().strip() or None
    root = Path(args.dir)
    for f in selected(manifest, args.only):
        dest = root / f["path"]
        if not (dest.is_file() and sha256_file(dest) == f["sha256"]):
            print(f"downloading {f['path']} ({f['size']} bytes)", flush=True)
            download(manifest["repo"], manifest["revision"], f["path"], dest, token)
        if sha256_file(dest) != f["sha256"]:
            dest.unlink()
            raise SystemExit(f"evidence: {f['path']} does not have its manifest digest; removed")
        if "members" in f:
            unpack(dest, root, f["members"])
    check(args)


def check(args):
    manifest = load_manifest()
    root = Path(args.dir)
    problems, checked = [], 0
    for f in selected(manifest, args.only):
        path = root / f["path"]
        if not path.is_file() or sha256_file(path) != f["sha256"]:
            problems.append(f"{f['path']}: missing or not its manifest digest")
            continue
        checked += 1
        for name, digest in f.get("members", {}).items():
            member = root / name
            if not member.is_file() or sha256_file(member) != digest:
                problems.append(f"{name} (of {f['path']}): missing or not its manifest digest")
            else:
                checked += 1
    for record in sorted(root.glob("*/handoff/candidates/*/*/candidate.json")):
        cand = json.loads(record.read_text(encoding="utf-8"))
        adapter = root / cand["adapter"]
        if adapter.is_file():
            checked += 1
            if "sha256:" + sha256_file(adapter) != cand["adapter_digest"]:
                problems.append(f"{cand['adapter']}: not the digest {record.relative_to(root)} declares")
    for p in problems:
        print(f"FAIL {p}")
    print(f"evidence {root}: {checked} files and members checked, {len(problems)} problems")
    if problems:
        raise SystemExit(1)


def manifest(args):
    """The manifest of a dataset copy: every file with its size and digest, and each archive's members
    with theirs. The maintainer writes it once per published revision."""
    root = Path(args.dir)
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".") or "/." in rel or rel == "README.md" or not rel.startswith(("records/", "adapters/", "exam")):
            continue
        entry = {"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)}
        if rel.endswith(".tar.xz"):
            with tarfile.open(path, "r:xz") as tar:
                entry["members"] = {m.name: hashlib.sha256(tar.extractfile(m).read()).hexdigest()
                                    for m in sorted(tar, key=lambda m: m.name) if m.isfile()}
        files.append(entry)
    out = {"repo": args.repo, "revision": args.revision, "files": files}
    MANIFEST.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{MANIFEST}: {len(files)} files")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("fetch", "check"):
        s = sub.add_parser(name)
        s.add_argument("--dir", required=True, type=Path)
        s.add_argument("--only", nargs="*", default=[], help="manifest paths (or directory prefixes) to use")
        if name == "fetch":
            s.add_argument("--token-file", type=Path, help=f"access token file (default {DEFAULT_TOKEN} if present)")
    s = sub.add_parser("manifest")
    s.add_argument("--dir", required=True, type=Path)
    s.add_argument("--repo", required=True)
    s.add_argument("--revision", required=True)
    args = p.parse_args(argv)
    {"fetch": fetch, "check": check, "manifest": manifest}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
