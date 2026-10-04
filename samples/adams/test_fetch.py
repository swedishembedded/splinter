#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements polite, auditable acquisition of public-domain
# archives for model training for its clients. If your team needs expertise in
# building a corpus whose every byte can be traced to where it came from, you
# can procure our services by sending an email to info@swedishembedded.com.

"""Specification of the corpus fetcher, against a fake network.

No case touches the network: the transport, the clock and the sleep are
injected, so politeness (robots, delay, backoff) is asserted by what the
fetcher asked for, not by waiting.

Usage: python3 samples/adams/test_fetch.py
"""
import hashlib
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch  # noqa: E402

TEXT = b"To the Inhabitants of the Colony of Massachusetts Bay.\n" * 4
URL = "https://archive.org/download/vol1/vol1_djvu.txt"
SOURCE = fetch.Source("cushing-1", URL, "primary", "public domain (US, pre-1929)", "Writings, vol. I")


class Response:
    def __init__(self, status=200, body=b"", headers=None, final_url=None):
        self.status = status
        self.headers = headers or {}
        self.final_url = final_url
        self.stream = io.BytesIO(body)

    def read(self, n):
        return self.stream.read(n)

    def close(self):
        pass


class FakeNetwork:
    """Routes a URL to a list of responses, one per request, the last repeating."""

    def __init__(self, routes):
        self.routes = routes
        self.requests = []

    def open(self, url, user_agent):
        self.requests.append((url, user_agent))
        queue = self.routes.get(url)
        if queue is None:
            return Response(404)
        response = queue.pop(0) if len(queue) > 1 else queue[0]
        response.final_url = response.final_url or url
        return response


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def robots_allowing():
    return {"https://archive.org/robots.txt": [Response(200, b"User-agent: *\nAllow: /\n")]}


class FetcherTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.clock = Clock()

    def fetcher(self, routes, **kw):
        self.network = FakeNetwork(routes)
        return fetch.Fetcher(self.dir.name, self.network, self.clock.time, self.clock.sleep, **kw)

    def manifest(self):
        with open(os.path.join(self.dir.name, "MANIFEST.tsv")) as f:
            header, *rows = [line.rstrip("\n").split("\t") for line in f]
        return [dict(zip(header, row)) for row in rows]

    def test_a_fetched_document_is_stored_and_recorded_with_its_provenance(self):
        f = self.fetcher({**robots_allowing(), URL: [Response(200, TEXT)]})
        failures = f.run([SOURCE])
        self.assertEqual(failures, [])
        raw = open(os.path.join(self.dir.name, "raw", "cushing-1.txt"), "rb").read()
        self.assertEqual(raw, TEXT)
        row = self.manifest()[0]
        self.assertEqual(row["sha256"], hashlib.sha256(TEXT).hexdigest())
        self.assertEqual(row["bytes"], str(len(TEXT)))
        self.assertEqual(row["url"], URL)
        self.assertEqual(row["licence"], "public domain (US, pre-1929)")
        self.assertEqual(row["role"], "primary")
        self.assertEqual(row["robots_ok"], "yes")

    def test_a_host_outside_the_allowlist_is_never_contacted(self):
        elsewhere = fetch.Source("x", "https://example.com/a.txt", "primary", "?", "?")
        f = self.fetcher({})
        failures = f.run([elsewhere])
        self.assertEqual(len(failures), 1)
        self.assertEqual(self.network.requests, [])

    def test_a_redirect_to_a_host_outside_the_allowlist_is_refused(self):
        routes = {**robots_allowing(), URL: [Response(200, TEXT, final_url="https://evil.example/x")]}
        failures = self.fetcher(routes).run([SOURCE])
        self.assertEqual(len(failures), 1)
        self.assertFalse(os.path.exists(os.path.join(self.dir.name, "raw", "cushing-1.txt")))

    def test_robots_disallowing_the_path_stops_the_fetch(self):
        routes = {"https://archive.org/robots.txt": [Response(200, b"User-agent: *\nDisallow: /download/\n")]}
        failures = self.fetcher(routes).run([SOURCE])
        self.assertEqual(len(failures), 1)
        self.assertEqual([u for u, _ in self.network.requests], ["https://archive.org/robots.txt"])

    def test_an_unreachable_robots_file_is_treated_as_disallowing(self):
        routes = {"https://archive.org/robots.txt": [Response(503)]}
        failures = self.fetcher(routes, max_attempts=2).run([SOURCE])
        self.assertEqual(len(failures), 1)
        self.assertNotIn(URL, [u for u, _ in self.network.requests])

    def test_a_missing_robots_file_allows_the_fetch(self):
        f = self.fetcher({"https://archive.org/robots.txt": [Response(404)], URL: [Response(200, TEXT)]})
        self.assertEqual(f.run([SOURCE]), [])

    def test_requests_to_one_host_are_spaced_by_the_minimum_delay(self):
        other = fetch.Source("cushing-2", URL.replace("vol1", "vol2"), "primary", "pd", "vol II")
        routes = {**robots_allowing(), URL: [Response(200, TEXT)], other.url: [Response(200, TEXT + b"2")]}
        self.fetcher(routes, min_delay=2.0).run([SOURCE, other])
        self.assertGreaterEqual(sum(self.clock.slept), 4.0 - 1e-9)

    def test_every_request_names_the_project_and_a_contact(self):
        self.fetcher({**robots_allowing(), URL: [Response(200, TEXT)]}).run([SOURCE])
        agents = {agent for _, agent in self.network.requests}
        self.assertEqual(len(agents), 1)
        self.assertIn("info@swedishembedded.com", agents.pop())

    def test_a_document_already_stored_and_recorded_is_not_fetched_again(self):
        routes = {**robots_allowing(), URL: [Response(200, TEXT)]}
        self.fetcher(routes).run([SOURCE])
        again = self.fetcher({})
        self.assertEqual(again.run([SOURCE]), [])
        self.assertEqual(again.network.requests, [])

    def test_a_stored_file_that_no_longer_matches_its_record_is_fetched_again(self):
        routes = {**robots_allowing(), URL: [Response(200, TEXT)]}
        self.fetcher(routes).run([SOURCE])
        with open(os.path.join(self.dir.name, "raw", "cushing-1.txt"), "ab") as f:
            f.write(b"tampered")
        routes = {**robots_allowing(), URL: [Response(200, TEXT)]}
        self.assertEqual(self.fetcher(routes).run([SOURCE]), [])
        self.assertEqual(open(os.path.join(self.dir.name, "raw", "cushing-1.txt"), "rb").read(), TEXT)

    def test_a_response_over_the_size_cap_is_refused_and_leaves_nothing(self):
        routes = {**robots_allowing(), URL: [Response(200, TEXT)]}
        failures = self.fetcher(routes, max_bytes=len(TEXT) - 1).run([SOURCE])
        self.assertEqual(len(failures), 1)
        raw = os.path.join(self.dir.name, "raw")
        self.assertEqual(os.listdir(raw) if os.path.isdir(raw) else [], [])
        self.assertFalse(os.path.exists(os.path.join(self.dir.name, "MANIFEST.tsv")) and self.manifest())

    def test_a_transient_failure_is_retried_with_growing_backoff_then_succeeds(self):
        routes = {**robots_allowing(), URL: [Response(503), Response(429, headers={"Retry-After": "7"}), Response(200, TEXT)]}
        self.assertEqual(self.fetcher(routes, min_delay=0.0, backoff=1.0).run([SOURCE]), [])
        self.assertEqual([s for s in self.clock.slept if s > 0], [1.0, 7.0])

    def test_retries_are_bounded_and_the_failure_is_reported_not_hidden(self):
        routes = {**robots_allowing(), URL: [Response(503)]}
        failures = self.fetcher(routes, max_attempts=3, min_delay=0.0).run([SOURCE])
        self.assertEqual(len(failures), 1)
        self.assertEqual(sum(1 for u, _ in self.network.requests if u == URL), 3)

    def test_a_dry_run_contacts_nothing_and_writes_nothing(self):
        f = self.fetcher({})
        plan = f.run([SOURCE], dry_run=True)
        self.assertEqual(plan, [])
        self.assertEqual(self.network.requests, [])
        self.assertEqual(os.listdir(self.dir.name), [])

    def test_one_failing_source_does_not_lose_the_others(self):
        bad = fetch.Source("cushing-9", URL.replace("vol1", "vol9"), "primary", "pd", "missing")
        routes = {**robots_allowing(), URL: [Response(200, TEXT)]}
        failures = self.fetcher(routes, min_delay=0.0).run([bad, SOURCE])
        self.assertEqual([s.id for s, _ in failures], ["cushing-9"])
        self.assertEqual([r["source_id"] for r in self.manifest()], ["cushing-1"])


class SourceTableTest(unittest.TestCase):
    def test_every_source_is_on_an_allowed_host_with_a_licence_and_a_role(self):
        for s in fetch.SOURCES:
            self.assertTrue(fetch.host_allowed(fetch.host_of(s.url)), s.id)
            self.assertTrue(s.licence, s.id)
            self.assertIn(s.role, {"primary", "secondary"}, s.id)

    def test_source_ids_are_unique_and_safe_as_file_names(self):
        ids = [s.id for s in fetch.SOURCES]
        self.assertEqual(len(ids), len(set(ids)))
        for i in ids:
            self.assertRegex(i, r"^[a-z0-9][a-z0-9-]*$")

    def test_the_four_volumes_of_the_writings_are_all_present(self):
        self.assertEqual({f"cushing-{n}" for n in (1, 2, 3, 4)} - {s.id for s in fetch.SOURCES}, set())


if __name__ == "__main__":
    unittest.main()
