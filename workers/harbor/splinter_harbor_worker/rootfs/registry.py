# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for reproducible container-image based
# build and test environments on hosts without a container daemon for its clients.
# If your team needs expertise in hermetic benchmark environments then you can
# procure our services by sending an email to info@swedishembedded.com.
"""Anonymous pulls from a Docker Registry HTTP API v2 (Docker Hub), without a daemon.

``pull(reference)`` resolves an image reference to the manifest of the host's platform
(a multi-architecture index is searched for the host's platform, ``linux/arm64`` here, or the
platform asked for; a single-platform image is checked against its config), downloads its config and layer blobs into a content-addressed cache
(``<cache>/blobs/sha256/<hex>``; every blob is verified against its digest while it
streams in, so a cached blob is always intact) and returns them. A reference pinned by
digest (``name@sha256:...``) never asks the registry which image a tag means.

The network is behind one callable (``Transport``), so tests substitute canned
responses. Only what the benchmark needs is supported: public repositories on a
registry that answers Bearer-token challenges (Docker Hub), gzip or uncompressed layer
tarballs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from splinter_harbor_worker.errors import ExternalToolError, UsageError

DEFAULT_REGISTRY = "registry-1.docker.io"
INDEX_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
)
MANIFEST_TYPES = (
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
)
LAYER_TYPES = {
    "application/vnd.oci.image.layer.v1.tar",
    "application/vnd.oci.image.layer.v1.tar+gzip",
    "application/vnd.docker.image.rootfs.diff.tar.gzip",
}
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict[str, str]  # lower-case names
    body: BinaryIO


# (url, request headers, method) -> Response. Raises OSError when the registry cannot be reached.
Transport = Callable[..., Response]


class PlatformUnavailable(ExternalToolError):
    """The image exists but has no build for the platform asked for."""


def host_platform() -> tuple[str, str]:
    """The image platform this host runs natively: ``("linux", "arm64")`` on aarch64, ``("linux", "amd64")`` on x86-64."""
    machine = platform.machine()
    return "linux", {"aarch64": "arm64", "arm64": "arm64", "x86_64": "amd64", "amd64": "amd64"}.get(machine, machine)


def urllib_transport(url: str, headers: dict[str, str], method: str = "GET") -> Response:
    request = urllib.request.Request(url, headers=headers, method=method)
    try:
        reply = urllib.request.urlopen(request, timeout=60)  # noqa: S310 - https registry URLs only
    except urllib.error.HTTPError as exc:
        return Response(exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc)
    return Response(reply.status, {k.lower(): v for k, v in reply.headers.items()}, reply)


RATE_LIMIT_RETRIES = 3
RATE_LIMIT_MAX_WAIT_SEC = 120


def _retry_after(header: str | None, attempt: int) -> float:
    """Seconds to wait before retrying a rate-limited request: the registry's ``Retry-After`` (seconds), else a backoff."""
    try:
        wait = float(header) if header else 15.0 * 2 ** (attempt - 1)
    except ValueError:
        wait = 15.0 * 2 ** (attempt - 1)
    return min(max(wait, 1.0), RATE_LIMIT_MAX_WAIT_SEC)


@dataclass(frozen=True)
class Reference:
    registry: str
    repository: str
    tag: str | None
    digest: str | None

    @property
    def name(self) -> str:
        return f"{self.registry}/{self.repository}"

    @property
    def target(self) -> str:
        return self.digest or self.tag or "latest"


def parse_reference(text: str) -> Reference:
    """``ubuntu:24.04``, ``python:3.13-slim-bookworm``, ``ubuntu@sha256:...``,
    ``org/name:tag``. Docker Hub is the registry, official images live under ``library/``."""
    name, _, digest = text.partition("@")
    if digest and not _DIGEST.fullmatch(digest):
        raise UsageError(f"image reference {text!r}: the digest must be sha256:<64 hex>")
    tag = None
    head, _, last = name.rpartition("/")
    if ":" in last:
        last, _, tag = last.partition(":")
        name = f"{head}/{last}" if head else last
    if not name or "." in name.split("/")[0] or not re.fullmatch(r"[a-z0-9][a-z0-9._/-]*", name):
        raise UsageError(f"image reference {text!r}: only Docker Hub repositories are supported")
    repository = name if "/" in name else f"library/{name}"
    return Reference(DEFAULT_REGISTRY, repository, tag or (None if digest else "latest"), digest or None)


@dataclass(frozen=True)
class Image:
    reference: str
    manifest_digest: str  # of the platform manifest
    config: dict  # the image config JSON ({"config": {"Env": ..., "WorkingDir": ...}, ...})
    layers: tuple[Path, ...]  # cached layer tarballs, lowest first


class RegistryClient:
    def __init__(self, cache: Path, transport: Transport = urllib_transport,
                 sleep: Callable[[float], None] = time.sleep, rate_limit_retries: int = RATE_LIMIT_RETRIES,
                 mirror: str | None = None) -> None:
        if mirror is not None and not re.fullmatch(r"[a-z0-9][a-z0-9.-]*(:[0-9]+)?", mirror):
            raise UsageError(f"registry mirror {mirror!r}: give a host name, such as mirror.gcr.io")
        self._mirror = mirror
        self._cache = cache
        self._transport = transport
        self._sleep = sleep
        self._rate_limit_retries = rate_limit_retries
        self._tokens: dict[str, str] = {}

    def host(self, reference: Reference) -> str:
        """Where requests for ``reference`` go: its registry, or the mirror that serves Docker Hub's repositories
        (every manifest and blob is still checked against its digest, so a mirror can withhold an image, never alter
        one; a tag it resolves may lag Docker Hub's, and the trial's record names the mirror)."""
        return self._mirror if self._mirror and reference.registry == DEFAULT_REGISTRY else reference.registry

    def blob_path(self, digest: str) -> Path:
        return self._cache / "blobs" / "sha256" / digest.removeprefix("sha256:")

    def resolve(self, text: str, platform: tuple[str, str] | None = None) -> tuple[str, str]:
        """(index or manifest digest the reference names, platform manifest digest). ``platform`` defaults to the
        host's; an image without a build for it raises ``PlatformUnavailable`` naming the platforms it has."""
        platform = platform or host_platform()
        reference = parse_reference(text)
        document, digest = self._manifest(reference, reference.target)
        if document.get("mediaType") in INDEX_TYPES or "manifests" in document:
            offered = []
            for entry in document["manifests"]:
                found = entry.get("platform", {})
                if (found.get("os"), found.get("architecture")) == platform:
                    return digest, entry["digest"]
                offered.append(f"{found.get('os')}/{found.get('architecture')}")
            raise PlatformUnavailable(f"{text}: the image has no {platform[0]}/{platform[1]} build (it has "
                                      f"{', '.join(o for o in offered if o != 'unknown/unknown') or 'none'})")
        # A single-platform manifest says its platform in its config.
        config = json.loads(self._blob(reference, document["config"]["digest"]).read_text())
        found = (config.get("os"), config.get("architecture"))
        if found != platform:
            raise PlatformUnavailable(f"{text}: the image is built for {found[0]}/{found[1]} only, not "
                                      f"{platform[0]}/{platform[1]}")
        return digest, digest

    def pull(self, text: str, platform: tuple[str, str] | None = None) -> Image:
        reference = parse_reference(text)
        _, manifest_digest = self.resolve(text, platform)
        manifest, _ = self._manifest(reference, manifest_digest)
        config_path = self._blob(reference, manifest["config"]["digest"])
        layers = []
        for layer in manifest["layers"]:
            if layer["mediaType"] not in LAYER_TYPES:
                raise ExternalToolError(f"{text}: unsupported layer type {layer['mediaType']}")
            layers.append(self._blob(reference, layer["digest"]))
        return Image(text, manifest_digest, json.loads(config_path.read_text()), tuple(layers))

    # -- registry calls ----------------------------------------------------------

    def _manifest(self, reference: Reference, target: str) -> tuple[dict, str]:
        """The manifest (or index) ``target`` names, and its digest. Manifests are cached by digest like blobs, and a
        tag is resolved with a HEAD request, which a registry's pull limit does not count: a tag whose manifest is
        cached costs no counted pull, a digest none at all."""
        accept = ",".join(INDEX_TYPES + MANIFEST_TYPES)
        url = f"https://{self.host(reference)}/v2/{reference.repository}/manifests/{target}"
        digest = target if _DIGEST.fullmatch(target) else self._tag_digest(reference, target, url, accept)
        cached = self.blob_path(digest)
        if cached.is_file():
            return json.loads(cached.read_bytes()), digest
        return self._fetched_manifest(reference, url.rsplit("/", 1)[0] + "/" + digest, accept, digest)

    def _fetched_manifest(self, reference: Reference, url: str, accept: str, digest: str | None) -> tuple[dict, str]:
        response = self._get(reference, url, {"Accept": accept})
        try:
            body = response.body.read()
        finally:
            response.body.close()
        found = "sha256:" + hashlib.sha256(body).hexdigest()
        if digest and found != digest:
            raise ExternalToolError(f"{url}: the manifest hashes to {found}, not {digest}")
        self._store(found, body)
        return json.loads(body), found

    def _tag_digest(self, reference: Reference, tag: str, url: str, accept: str) -> str | None:
        """The digest ``tag`` was resolved to the first time this cache asked (``tags/``), so a tag means one image
        for every trial of every run on this cache, and is not asked again; delete the record to resolve it anew."""
        record = self._cache / "tags" / f"{hashlib.sha256(f'{reference.name}:{tag}'.encode()).hexdigest()}.json"
        try:
            return json.loads(record.read_text())["digest"]
        except (OSError, ValueError, KeyError):
            pass
        digest = self._head_digest(reference, url, accept)
        if digest is None:
            _, digest = self._fetched_manifest(reference, url, accept, None)
        record.parent.mkdir(parents=True, exist_ok=True)
        partial = record.with_name(f"{record.name}.partial-{os.getpid()}")
        partial.write_text(json.dumps({"reference": f"{reference.name}:{tag}", "digest": digest,
                                       "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        partial.replace(record)
        return digest

    def _head_digest(self, reference: Reference, url: str, accept: str) -> str | None:
        """The digest a tag names now (``Docker-Content-Digest``), or None when the registry does not say."""
        try:
            response = self._get(reference, url, {"Accept": accept}, method="HEAD")
        except ExternalToolError:
            return None
        response.body.close()
        digest = response.headers.get("docker-content-digest", "")
        return digest if _DIGEST.fullmatch(digest) else None

    def _store(self, digest: str, body: bytes) -> None:
        path = self.blob_path(digest)
        if path.is_file():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.partial-{os.getpid()}-{os.urandom(4).hex()}")
        partial.write_bytes(body)
        partial.replace(path)

    def _blob(self, reference: Reference, digest: str) -> Path:
        path = self.blob_path(digest)
        if path.is_file():
            return path
        url = f"https://{self.host(reference)}/v2/{reference.repository}/blobs/{digest}"
        response = self._get(reference, url, {})
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique per download: concurrent pulls of images that share a layer each write their own.
        partial = path.with_name(f"{path.name}.partial-{os.getpid()}-{os.urandom(4).hex()}")
        hasher = hashlib.sha256()
        try:
            with partial.open("wb") as out:
                while chunk := response.body.read(_CHUNK):
                    hasher.update(chunk)
                    out.write(chunk)
        finally:
            response.body.close()
        if "sha256:" + hasher.hexdigest() != digest:
            partial.unlink(missing_ok=True)
            raise ExternalToolError(f"{url}: the download does not match its digest {digest}")
        partial.replace(path)
        return path

    def _get(self, reference: Reference, url: str, headers: dict[str, str], method: str = "GET") -> Response:
        scope = reference.repository
        authenticated, limited = False, 0
        while True:
            auth = {"Authorization": f"Bearer {self._tokens[scope]}"} if scope in self._tokens else {}
            try:
                response = self._transport(url, {**headers, **auth}, method)
            except OSError as exc:
                raise ExternalToolError(f"{url}: {exc}") from None
            if response.status == 401 and not authenticated:
                authenticated = True
                self._tokens[scope] = self._token(response.headers.get("www-authenticate", ""), scope)
                response.body.close()
                continue
            if response.status == 429 and limited < self._rate_limit_retries:
                # Anonymous pulls are rate limited per address (shared by every user of a node): wait as asked, a few times.
                response.body.close()
                limited += 1
                self._sleep(_retry_after(response.headers.get("retry-after"), limited))
                continue
            if response.status == 401:
                response.body.close()
                raise ExternalToolError(f"{url}: the registry refused the anonymous token")
            if response.status != 200:
                response.body.close()
                hint = " (the anonymous pull rate limit; try again later)" if response.status == 429 else ""
                raise ExternalToolError(f"{url}: the registry answered {response.status}{hint}")
            return response

    def _token(self, challenge: str, scope: str) -> str:
        fields = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
        if not challenge.lower().startswith("bearer") or "realm" not in fields:
            raise ExternalToolError(f"the registry asks for authentication it cannot be given: {challenge!r}")
        query = {"service": fields.get("service", ""), "scope": f"repository:{scope}:pull"}
        url = f"{fields['realm']}?{urllib.parse.urlencode(query)}"
        try:
            response = self._transport(url, {})
        except OSError as exc:
            raise ExternalToolError(f"{url}: {exc}") from None
        try:
            if response.status != 200:
                raise ExternalToolError(f"{url}: the token service answered {response.status}")
            return json.loads(response.body.read())["token"]
        finally:
            response.body.close()
