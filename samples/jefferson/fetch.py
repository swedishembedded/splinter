#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements polite, auditable acquisition of public-domain
# archives for model training for its clients. If your team needs expertise in
# building a corpus whose every byte can be traced to where it came from, you
# can procure our services by sending an email to info@swedishembedded.com.

"""Fetch the Thomas Jefferson corpus's plain texts, politely and with provenance.

Splinter itself refuses URLs, so a corpus is acquired here, once, and learned
from as local files. Every fetch honours robots.txt, spaces its requests,
names itself, stays on an allowlist of hosts (redirects included), caps the
size it will accept, and records where each byte came from in MANIFEST.tsv.
A text already stored whose hash matches its record is not fetched again.

The layout is the one `splinter-jefferson` reads: `DIR/thomas-jefferson/<stem>.txt`
for Jefferson's letters and works, `DIR/founding-america/<stem>.txt` for the works
he argued from and with. A download is kept only when it names the work the
file name promises (the right volume, not merely a reachable one).

Usage: python3 samples/jefferson/fetch.py --resources DIR [--only ID ...]
                                      [--dry-run] [--list]
"""
import argparse
import collections
import hashlib
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser

USER_AGENT = "splinter-jefferson-fetch/1 (research corpus; info@swedishembedded.com)"

# A host is allowed when it is one of these or a subdomain of one: the
# Internet Archive serves a download from a data-centre host it redirects to.
ALLOWED_DOMAINS = ("archive.org", "gutenberg.org", "loc.gov")

MANIFEST_COLUMNS = ("source_id", "url", "role", "licence", "retrieved_at", "sha256", "bytes", "robots_ok")

# 429 and 5xx answer "not now", not "no": worth another try.
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})
RETRY_AFTER_CAP_SECONDS = 60.0

# `id` is `<directory>/<stem>`: the file is written to `DIR/<id>.txt`, where the
# Rust side reads it. `markers` are phrases the downloaded text must contain
# (compared case-blind with runs of whitespace collapsed), so a wrong edition
# or volume is refused instead of stored.
Source = collections.namedtuple("Source", "id url role licence description markers")

_PD = "public domain (US, published before 1929)"
_PG_LICENCE = _PD + "; Project Gutenberg licence wrapper applies to the file"


def _pg(book):
    return f"https://www.gutenberg.org/cache/epub/{book}/pg{book}.txt"


def _ia(identifier):
    return f"https://archive.org/download/{identifier}/{identifier}_djvu.txt"


def _gutenberg(directory, stem, book, role, description, title):
    return Source(f"{directory}/{stem}", _pg(book), role, _PG_LICENCE, description, (title,))


def _archive(directory, stem, identifier, description, *markers):
    return Source(f"{directory}/{stem}", _ia(identifier), "primary", _PD + "; OCR text", description, markers)


_WASHINGTON_VOLUMES = (45847, 50046, 52878, 53603, 53767, 55075, 56035, 56313, 56578)
_RANDOLPH_VOLUMES = (16781, 16782, 16783, 16784)

# The letters come from three clean editions (the Washington edition of 1853,
# Randolph's 1829 Memoir, and volume 6 of the Library Edition of 1903); the
# editions overlap, which the Rust side resolves by family.
SOURCES = (
    *(
        _gutenberg(
            "thomas-jefferson", f"writings-washington-ed-v{n}", book, "primary",
            f"The Writings of Thomas Jefferson (H. A. Washington, 1853), vol. {n} of 9",
            f"The Writings of Thomas Jefferson, Vol. {n} (of 9)",
        )
        for n, book in enumerate(_WASHINGTON_VOLUMES, start=1)
    ),
    *(
        _gutenberg(
            "thomas-jefferson", f"memoir-correspondence-miscellanies-v{n}", book, "primary",
            f"Memoir, Correspondence, and Miscellanies from the Papers of Thomas Jefferson (Randolph, 1829), vol. {n} of 4",
            f"Memoir, Correspondence, and Miscellanies, From the Papers of Thomas Jefferson, Volume {n}",
        )
        for n, book in enumerate(_RANDOLPH_VOLUMES, start=1)
    ),
    Source(
        "thomas-jefferson/writings-library-ed-v6", _pg(21002), "primary", _PG_LICENCE,
        "The Writings of Thomas Jefferson (Lipscomb and Bergh, Library Edition, 1903), vol. VI",
        ("Title: The Writings of Thomas Jefferson", "Library Edition", "VOL. VI."),
    ),
    _archive(
        "thomas-jefferson", "notes-on-the-state-of-virginia-1853", "notesonstateofvi01jeff",
        "Notes on the State of Virginia (J. W. Randolph, Richmond, 1853)", "J. W. RANDOLPH", "QUERY I.",
    ),
    _archive(
        "thomas-jefferson", "a-summary-view-of-the-rights-of-british-america-1774", "summaryviewofrig00jeff",
        "A Summary View of the Rights of British America (Dunlap, Philadelphia, 1774)",
        "SUMMARY VIEW", "PEOPLE OF VIRGINIA", "M,DCC,LXXIV",
    ),
    _archive(
        "thomas-jefferson", "manual-of-parliamentary-practice-1820", "amanualparliame02jeffgoog",
        "A Manual of Parliamentary Practice (Davis and Force, Washington, 1820)", "BY THOMAS JEFFERSON", "PREFACE",
    ),
    _archive(
        "thomas-jefferson", "life-and-morals-of-jesus-of-nazareth-1904", "thelifeandmorals00jeffuoft",
        "The Life and Morals of Jesus of Nazareth (1904)", "Greek, Latin, French, and English", "THOMAS JEFFERSON",
    ),
    _gutenberg(
        "thomas-jefferson", "declaration-of-independence", 1, "primary",
        "The Declaration of Independence", "The Declaration of Independence of the United States of America",
    ),
    # The works Jefferson argued from, with and against: context for the
    # "which work is this passage from" question, never a belief of his.
    _gutenberg("founding-america", "the-federalist-papers", 1404, "secondary", "The Federalist Papers", "The Federalist Papers"),
    _gutenberg("founding-america", "common-sense", 147, "secondary", "Paine, Common Sense", "Common Sense"),
    _gutenberg(
        "founding-america", "paine-writings-v2-rights-of-man", 3742, "secondary",
        "Paine, Writings vol. 2: The Rights of Man", "Volume 2 (1779-1792): The Rights of Man",
    ),
    _gutenberg(
        "founding-america", "paine-writings-v4-age-of-reason", 3743, "secondary",
        "Paine, Writings vol. 4: The Age of Reason", "Volume 4 (1794-1796): The Age of Reason",
    ),
    _gutenberg(
        "founding-america", "locke-second-treatise-of-government", 7370, "secondary",
        "Locke, Second Treatise of Government", "Second Treatise of Government",
    ),
    _gutenberg("founding-america", "hobbes-leviathan", 3207, "secondary", "Hobbes, Leviathan", "Leviathan"),
    _gutenberg(
        "founding-america", "rousseau-social-contract-and-discourses", 46333, "secondary",
        "Rousseau, The Social Contract and Discourses", "The social contract & discourses",
    ),
    _gutenberg(
        "founding-america", "blackstone-commentaries-book-1", 30802, "secondary",
        "Blackstone, Commentaries on the Laws of England, Book the First", "Commentaries on the Laws of England, Book the First",
    ),
    _gutenberg(
        "founding-america", "smith-wealth-of-nations", 3300, "secondary",
        "Smith, The Wealth of Nations", "An Inquiry into the Nature and Causes of the Wealth of Nations",
    ),
    _gutenberg(
        "founding-america", "franklin-autobiography", 148, "secondary",
        "Franklin, Autobiography", "The Autobiography of Benjamin Franklin",
    ),
    _gutenberg(
        "founding-america", "dickinson-letters-from-a-farmer", 47111, "secondary",
        "Dickinson, Letters from a Farmer in Pennsylvania",
        "Letters from a Farmer in Pennsylvania, to the Inhabitants of the British Colonies",
    ),
    _gutenberg(
        "founding-america", "adams-novanglus-and-massachusettensis", 45205, "secondary",
        "John Adams, Novanglus, and Daniel Leonard, Massachusettensis", "Novanglus, and Massachusettensis",
    ),
    # works.rs credits this file to Chittenden, whose report is of the secret
    # sessions of the Conference Convention of 1861, not of the Federal
    # Convention of 1787 that its title there names; the source follows the
    # author, and the mismatch is for works.rs to resolve.
    _gutenberg(
        "founding-america", "secret-debates-of-the-federal-convention", 24561, "secondary",
        "Chittenden, Report of the Debates and Proceedings in the Secret Sessions of the Conference Convention",
        "A Report of the Debates and Proceedings in the Secret Sessions of the Conference Convention",
    ),
)


def host_of(url):
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def host_allowed(host):
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


def missing_markers(data, markers):
    """The markers the text does not contain, compared case-blind, whitespace collapsed."""
    text = " ".join(data.decode("utf-8", "replace").lower().split())
    return [m for m in markers if " ".join(m.lower().split()) not in text]


class FetchError(Exception):
    """A source that could not be fetched, and why."""


class UrllibResponse:
    def __init__(self, status, headers, stream, final_url):
        self.status = status
        self.headers = headers
        self.stream = stream
        self.final_url = final_url

    def read(self, n):
        return self.stream.read(n)

    def close(self):
        self.stream.close()


class UrllibNetwork:
    """The real transport: one GET, redirects followed, a status for every outcome."""

    def __init__(self, timeout=60.0):
        self.timeout = timeout

    def open(self, url, user_agent):
        request = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)  # noqa: S310 - host checked by the caller
        except urllib.error.HTTPError as e:
            return UrllibResponse(e.code, dict(e.headers), e, url)
        return UrllibResponse(response.status, dict(response.headers), response, response.geturl())


def _clean(field):
    return str(field).replace("\t", " ").replace("\n", " ")


class Fetcher:
    def __init__(self, resources, network, clock=time.time, sleep=time.sleep, min_delay=2.0, max_bytes=64 * 1024 * 1024, max_attempts=4, backoff=2.0):
        self.resources = resources
        self.network = network
        self.clock = clock
        self.sleep = sleep
        self.min_delay = min_delay
        self.max_bytes = max_bytes
        self.max_attempts = max_attempts
        self.backoff = backoff
        self._last_request = {}
        self._robots = {}

    def run(self, sources, dry_run=False):
        """Fetch every source; return the (source, reason) pairs that failed."""
        manifest = self._read_manifest()
        failures = []
        for source in sources:
            try:
                if self._is_stored(source, manifest):
                    print(f"{source.id}: stored, hash matches its record")
                elif dry_run:
                    print(f"{source.id}: would fetch {source.url}")
                else:
                    manifest[source.id] = self._fetch_one(source)
                    self._write_manifest(manifest)
                    print(f"{source.id}: fetched {manifest[source.id]['bytes']} bytes")
            except FetchError as e:
                failures.append((source, str(e)))
                print(f"{source.id}: FAILED: {e}", file=sys.stderr)
        return failures

    def _stored_path(self, source):
        return os.path.join(self.resources, *source.id.split("/")) + ".txt"

    def _is_stored(self, source, manifest):
        row = manifest.get(source.id)
        path = self._stored_path(source)
        return bool(row) and os.path.isfile(path) and _sha256_file(path) == row["sha256"]

    def _fetch_one(self, source):
        if not host_allowed(host_of(source.url)):
            raise FetchError(f"host {host_of(source.url)!r} is not on the allowlist")
        allowed = self._robots_allow(source.url)
        if allowed is None:
            raise FetchError("robots.txt could not be read, and an unreadable file is not permission")
        if not allowed:
            raise FetchError("robots.txt does not allow this path")
        data, final_url = self._download(source.url)
        if not host_allowed(host_of(final_url)):
            raise FetchError(f"redirected to {host_of(final_url)!r}, which is not on the allowlist")
        missing = missing_markers(data, source.markers)
        if missing:
            raise FetchError(f"not the expected work: the text lacks {missing!r}")
        os.makedirs(os.path.dirname(self._stored_path(source)), exist_ok=True)
        _atomic_write(self._stored_path(source), data)
        return {
            "source_id": source.id,
            "url": source.url,
            "role": source.role,
            "licence": source.licence,
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.clock())),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": str(len(data)),
            "robots_ok": "yes",
        }

    def _robots_allow(self, url):
        """True or False by robots.txt, or None when it could not be read."""
        parts = urllib.parse.urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            self._robots[origin] = self._read_robots(origin + "/robots.txt")
        parser = self._robots[origin]
        return None if parser is None else parser.can_fetch(USER_AGENT, url)

    def _read_robots(self, robots_url):
        """The parsed robots.txt, an allow-all parser when there is none, or None
        when it cannot be read (an unreadable file is not permission)."""
        try:
            data, _ = self._download(robots_url, missing_ok=True)
        except FetchError:
            return None
        parser = urllib.robotparser.RobotFileParser()
        if data is None:
            parser.parse(["User-agent: *", "Allow: /"])
        else:
            parser.parse(data.decode("utf-8", "replace").splitlines())
        return parser

    def _download(self, url, missing_ok=False):
        """The body and final URL, retrying transient answers a bounded number of times."""
        last = "no attempt made"
        for attempt in range(self.max_attempts):
            self._space_requests(url)
            response = self.network.open(url, USER_AGENT)
            try:
                if response.status == 200:
                    return self._read_capped(response), response.final_url
                if response.status == 404 and missing_ok:
                    return None, response.final_url
                last = f"HTTP {response.status}"
                if response.status not in TRANSIENT_STATUSES:
                    raise FetchError(last)
                delay = self._retry_delay(response, attempt)
            finally:
                response.close()
            if attempt + 1 < self.max_attempts:
                self.sleep(delay)
        raise FetchError(f"{last} after {self.max_attempts} attempts")

    def _retry_delay(self, response, attempt):
        retry_after = response.headers.get("Retry-After", "")
        if retry_after.isdigit():
            return min(float(retry_after), RETRY_AFTER_CAP_SECONDS)
        return self.backoff * (2 ** attempt)

    def _read_capped(self, response):
        declared = response.headers.get("Content-Length", "")
        if declared.isdigit() and int(declared) > self.max_bytes:
            raise FetchError(f"{declared} bytes exceeds the {self.max_bytes}-byte cap")
        chunks, total = [], 0
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if total > self.max_bytes:
                raise FetchError(f"more than the {self.max_bytes}-byte cap")
            chunks.append(chunk)

    def _space_requests(self, url):
        host = host_of(url)
        last = self._last_request.get(host)
        if last is not None:
            wait = last + self.min_delay - self.clock()
            if wait > 0:
                self.sleep(wait)
        self._last_request[host] = self.clock()

    def _manifest_path(self):
        return os.path.join(self.resources, "MANIFEST.tsv")

    def _read_manifest(self):
        if not os.path.isfile(self._manifest_path()):
            return {}
        with open(self._manifest_path(), encoding="utf-8") as f:
            header, *rows = [line.rstrip("\n").split("\t") for line in f]
        return {r[0]: dict(zip(header, r)) for r in rows}

    def _write_manifest(self, manifest):
        lines = ["\t".join(MANIFEST_COLUMNS)]
        for source_id in sorted(manifest):
            lines.append("\t".join(_clean(manifest[source_id][c]) for c in MANIFEST_COLUMNS))
        _atomic_write(self._manifest_path(), ("\n".join(lines) + "\n").encode("utf-8"))


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path, data):
    """Write beside the target and rename, so a reader never sees half a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resources", help="directory the corpus is written under")
    parser.add_argument("--only", nargs="+", metavar="ID", help="fetch just these sources")
    parser.add_argument("--dry-run", action="store_true", help="say what would be fetched; contact nothing")
    parser.add_argument("--list", action="store_true", help="list the sources and exit")
    parser.add_argument("--min-delay", type=float, default=2.0, help="seconds between requests to one host")
    parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024, help="refuse a larger response")
    args = parser.parse_args(argv)
    if args.list:
        for s in SOURCES:
            print(f"{s.id}\t{s.role}\t{s.url}\t{s.description}")
        return 0
    if not args.resources:
        parser.error("--resources DIR is required")
    chosen = [s for s in SOURCES if not args.only or s.id in args.only]
    unknown = set(args.only or ()) - {s.id for s in SOURCES}
    if unknown:
        parser.error(f"unknown source(s): {', '.join(sorted(unknown))}")
    fetcher = Fetcher(args.resources, UrllibNetwork(), min_delay=args.min_delay, max_bytes=args.max_bytes)
    return 1 if fetcher.run(chosen, dry_run=args.dry_run) else 0


if __name__ == "__main__":
    sys.exit(main())
