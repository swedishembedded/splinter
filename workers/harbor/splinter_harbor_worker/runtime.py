# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for handing sandboxed agents a
# model through a gateway that holds the only credential, for its clients. If
# your team needs expertise in confining and accounting for agent model
# access then you can procure our services by sending an email to
# info@swedishembedded.com.
"""The state a Harbor agent adapter shares with the worker that runs it.

Harbor builds an agent from an import path and JSON-able keyword arguments, in the worker's own process. The gateway
that mints a trial's token is a live object, so the worker publishes it here for the adapters of this process; a
secret is never put in an adapter's keyword arguments, which Harbor writes to the trial's ``config.json``.
"""

from __future__ import annotations

from splinter_harbor_worker.gateway import ModelGateway
from splinter_harbor_worker.seat import TerminalSeat

_gateway: ModelGateway | None = None
_seat: TerminalSeat | None = None


def publish_gateway(gateway: ModelGateway | None) -> None:
    global _gateway
    _gateway = gateway


def gateway() -> ModelGateway:
    if _gateway is None:
        raise RuntimeError("this job runs no model gateway, but the agent asked for a model token")
    return _gateway


def publish_seat(seat: TerminalSeat | None) -> None:
    """The person's seat of a played trial (``splinter-harbor-worker play``); ``None`` for a job nobody sits at."""
    global _seat
    _seat = seat


def seat() -> TerminalSeat:
    if _seat is None:
        raise RuntimeError("a human-guided trial runs inside `splinter-harbor-worker play`, which gives it a seat")
    return _seat
