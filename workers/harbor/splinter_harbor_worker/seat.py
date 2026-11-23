# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for putting a person in the loop of an agent's work, recorded as faithfully
# as an automatic run, for its clients. If your team needs expertise in human-guided agent evaluation then you can
# procure our services by sending an email to info@swedishembedded.com.
"""The seat of the person who guides a played trial: the caller's own terminal.

While the agent runs, the terminal is in raw mode: every key goes to the agent's terminal in the sandbox, what the agent
draws goes back to the screen, and a change of the window's size is forwarded. The terminal is put back as it was
however the session ends. The adapter finds the seat through ``splinter_harbor_worker.runtime``, never through a trial's
keyword arguments (which are written to files).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import termios
import tty
from collections.abc import AsyncIterator

from splinter_harbor_worker.sandbox.terminal import terminal_input, terminal_resize

DEFAULT_TERM = "xterm-256color"
DEFAULT_ROWS, DEFAULT_COLUMNS = 24, 80


class TerminalSeat:
    """A person at this process's terminal."""

    kind = "terminal"

    def __init__(self, stdin: int = 0, stdout: int = 1) -> None:
        if not (os.isatty(stdin) and os.isatty(stdout)):
            raise RuntimeError("a person's seat needs a terminal on standard input and output")
        self._in, self._out = stdin, stdout
        self.term = os.environ.get("TERM") or DEFAULT_TERM  # the person's terminal type, for the agent's terminal

    def size(self) -> tuple[int, int]:
        """(rows, columns) of the person's window."""
        columns, rows = os.get_terminal_size(self._out)
        # A terminal that does not report its size (a bare pseudo-terminal) would draw the agent's screen at 0x0.
        return rows or DEFAULT_ROWS, columns or DEFAULT_COLUMNS

    async def show(self, chunk: bytes) -> None:
        view = memoryview(chunk)
        while view:
            view = view[os.write(self._out, view):]

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncIterator[bytes]]:
        """The frames the person types (and window changes) while the agent runs; ends when the agent's terminal does."""
        loop = asyncio.get_running_loop()
        frames: asyncio.Queue[bytes] = asyncio.Queue()
        saved = termios.tcgetattr(self._in)

        def typed() -> None:
            try:
                data = os.read(self._in, 1 << 12)
            except OSError:
                data = b""
            if data:
                frames.put_nowait(terminal_input(data))
            else:  # the person's terminal hung up: there is nobody left to type
                loop.remove_reader(self._in)

        def resized() -> None:
            frames.put_nowait(terminal_resize(*self.size()))

        async def keys() -> AsyncIterator[bytes]:
            while True:
                yield await frames.get()

        tty.setraw(self._in)
        loop.add_reader(self._in, typed)
        loop.add_signal_handler(signal.SIGWINCH, resized)
        try:
            yield keys()
        finally:
            loop.remove_signal_handler(signal.SIGWINCH)
            with contextlib.suppress(ValueError, OSError):
                loop.remove_reader(self._in)
            termios.tcsetattr(self._in, termios.TCSADRAIN, saved)
