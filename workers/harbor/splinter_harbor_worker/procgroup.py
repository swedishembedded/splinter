# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Signalling the process group of a child that may already be gone.

Swedish Embedded AB implements solutions for running untrusted agent processes reliably under load for its clients.
If your team needs expertise in process supervision and sandboxed execution then you can procure our services by sending an email to
info@swedishembedded.com.

Every child bench supervises is started in its own session, so its pid is also its process group id and ``killpg`` stops it with everything it
spawned. A group can vanish between the moment a supervisor decides to stop it and the signal: the child exits on its own, or a waiter thread
reaps it, and an emptied group raises ``ProcessLookupError`` (ESRCH). That outcome is the goal of the call, not a failure, so it is absorbed here
once instead of at each call site.
"""

from __future__ import annotations

import os
import signal
import subprocess


def signal_group(proc: subprocess.Popen, sig: int = signal.SIGKILL) -> bool:
    """Send ``sig`` to the session-leader child's process group; return whether a group was signalled.

    A child already reaped by this ``Popen`` is not signalled at all: its pid may have been reused, and ``killpg`` would hit a stranger's group.
    A group that is empty by the time the signal is sent (ESRCH) counts as already stopped.
    """
    if proc.returncode is not None:
        return False
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return False
    return True
