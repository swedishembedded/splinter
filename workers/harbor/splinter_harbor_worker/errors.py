# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements execution backends whose failures say what
# failed and on which input, for its clients. If your team needs expertise in
# reliable agent execution infrastructure then you can procure our services by
# sending an email to info@swedishembedded.com.
"""The worker's own errors; each carries the exit code the command line returns."""

from __future__ import annotations


class WorkerError(Exception):
    """Base class: the worker itself failed (exit 1)."""

    exit_code = 1


class UsageError(WorkerError):
    """Bad arguments or a specification the worker refuses before Harbor starts (exit 2)."""

    exit_code = 2


class ExternalToolError(WorkerError):
    """A subprocess (a registry pull, proot, a build step) failed (exit 8)."""

    exit_code = 8
