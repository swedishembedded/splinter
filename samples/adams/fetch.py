#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements polite, auditable acquisition of public-domain
# archives for model training for its clients. If your team needs expertise in
# building a corpus whose every byte can be traced to where it came from, you
# can procure our services by sending an email to info@swedishembedded.com.

"""Fetch the Samuel Adams corpus's raw texts, politely and with provenance.

Splinter itself refuses URLs, so a corpus is acquired here, once, and learned
from as local files. Every fetch honours robots.txt, spaces its requests,
names itself, stays on an allowlist of hosts (redirects included), caps the
size it will accept, and records where each byte came from in MANIFEST.tsv.
A text already stored whose hash matches its record is not fetched again.

Usage: python3 samples/adams/fetch.py --resources DIR [--only ID ...]
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

USER_AGENT = "splinter-adams-fetch/1 (research corpus; info@swedishembedded.com)"

# A host is allowed when it is one of these or a subdomain of one: the
# Internet Archive serves a download from a data-centre host it redirects to.
ALLOWED_DOMAINS = ("archive.org", "gutenberg.org", "loc.gov")

MANIFEST_COLUMNS = ("source_id", "url", "role", "licence", "retrieved_at", "sha256", "bytes", "robots_ok")

# 429 and 5xx answer "not now", not "no": worth another try.
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})
RETRY_AFTER_CAP_SECONDS = 60.0

Source = collections.namedtuple("Source", "id url role licence description")

_IA = "https://archive.org/download/{0}/{0}_djvu.txt"
_PD = "public domain (US, published before 1929)"

# Cushing's edition is the working text; Wells's biography is secondary: it
# supplies context and the letters it quotes, never a belief of Adams's.
SOURCES = (
    Source("cushing-1", _IA.format("writitngssamadam01adamrich"), "primary", _PD, "The Writings of Samuel Adams, vol. I, 1764-1769"),
    Source("cushing-2", _IA.format("writitngssamadam02adamrich"), "primary", _PD, "The Writings of Samuel Adams, vol. II, 1770-1773"),
    Source("cushing-3", _IA.format("writitngssamadam03adamrich"), "primary", _PD, "The Writings of Samuel Adams, vol. III, 1773-1777"),
    Source("cushing-4", _IA.format("writingsofsamuel0004adam"), "primary", _PD, "The Writings of Samuel Adams, vol. IV, 1778-1802"),
    Source("wells-1", _IA.format("lifeservsamadams01wellrich"), "secondary", _PD, "Wells, Life and Public Services of Samuel Adams, vol. I"),
    Source("wells-2", _IA.format("lifeservsamadams02wellrich"), "secondary", _PD, "Wells, Life and Public Services of Samuel Adams, vol. II"),
    Source("wells-3", _IA.format("lifeservsamadams03wellrich"), "secondary", _PD, "Wells, Life and Public Services of Samuel Adams, vol. III"),
)


def host_of(url):
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def host_allowed(host):
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


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

    def _raw_path(self, source):
        return os.path.join(self.resources, "raw", source.id + ".txt")

    def _is_stored(self, source, manifest):
        row = manifest.get(source.id)
        path = self._raw_path(source)
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
        os.makedirs(os.path.dirname(self._raw_path(source)), exist_ok=True)
        _atomic_write(self._raw_path(source), data)
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
