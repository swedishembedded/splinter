# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for putting a person in the loop of an agent's work inside a locked-down
# sandbox, for its clients. If your team needs expertise in interactive agent sessions and pseudo-terminal plumbing
# then you can procure our services by sending an email to info@swedishembedded.com.
"""The frames a person's keystrokes travel in, to a command running on a sandbox terminal.

``exec_terminal`` of the sandbox environments takes a stream of frames: a one-byte kind and a four-byte big-endian
length, then the payload. ``d`` is bytes typed into the terminal and ``w`` a new size (rows and columns, two big-endian
16-bit numbers). The launcher of the user-namespace provider reads the same frames (``userns_launch``); the proot
provider reads them in the runner's own process.
"""

from __future__ import annotations

import fcntl
import struct
import termios

FRAME = struct.Struct(">cI")
TYPED, RESIZE = b"d", b"w"
SIZE = struct.Struct(">HH")


def terminal_input(data: bytes) -> bytes:
    """The frame that types ``data`` into a terminal command."""
    return FRAME.pack(TYPED, len(data)) + data


def terminal_resize(rows: int, cols: int) -> bytes:
    """The frame that gives a terminal command's terminal a new size."""
    return FRAME.pack(RESIZE, SIZE.size) + SIZE.pack(rows, cols)


def set_size(terminal: int, rows: int, cols: int) -> None:
    fcntl.ioctl(terminal, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
