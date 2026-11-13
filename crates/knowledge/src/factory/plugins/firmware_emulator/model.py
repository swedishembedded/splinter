# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements emulator-verified firmware training
# environments, for its clients. If your team needs expertise in testing
# firmware against a reference model in simulation, you can procure our
# services by sending an email to info@swedishembedded.com.

"""The reference model of the firmware library and the harness that checks a
workspace against it.

`vectors(seed)` draws the inputs of one run and the console output the
correct firmware prints for them. `check(work, seed)` writes the vectors into
a copy of the workspace, builds it with the trusted flags and linker script,
runs it in the emulator and compares the console output line by line. The
inputs of the hidden seeds exist only here, never in the workspace.
"""
import math
import os
import subprocess
import sys
import tempfile

BUF_COUNT = 6
BUF_MAX = 16
PAIR_COUNT = 8
WORD_COUNT = 8
OP_COUNT = 24
RING_CAPACITY = 8
BOARD = "arm,vemu-cortex-m33"
CYCLES = "4000000"
HERE = os.path.dirname(os.path.abspath(__file__))
CFLAGS = [
    "--target=thumbv8m.main-none-eabi", "-mcpu=cortex-m33", "-mthumb", "-mfloat-abi=soft",
    "-Os", "-g", "-ffreestanding", "-fno-builtin", "-Wall", "-Wextra", "-std=c11",
    "-nostdlib", "-fuse-ld=lld", "-Wl,--gc-sections",
]


class Rng:
    """SplitMix64: the same stream on every platform and Python version."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def between(self, low, high):
        return low + self.next() % (high - low + 1)


def crc16(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def rle(data):
    out = []
    i = 0
    while i < len(data):
        run = 1
        while i + run < len(data) and data[i + run] == data[i] and run < 255:
            run += 1
        out += [run, data[i]]
        i += run
    return out


def div_round(a, b):
    q = (abs(a) + abs(b) // 2) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def bitrev32(x):
    return int("{:032b}".format(x)[::-1], 2)


def rotl32(x, n):
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def vectors(seed):
    """The C header of one run's inputs and the expected console lines."""
    rng = Rng(seed)
    bufs = []
    for _ in range(BUF_COUNT):
        data = []
        target = rng.between(4, BUF_MAX)
        while len(data) < target:
            value = rng.between(0, 5)
            data += [value] * min(rng.between(1, 5), target - len(data))
        bufs.append(data)
    pairs = []
    for i in range(PAIR_COUNT):
        b = rng.between(-20000, 20000) or 7
        a = rng.between(-20000, 20000)
        if i == 0:  # a half: rounds away from zero
            b = 2 * rng.between(3, 90)
            a = b * rng.between(-40, 40) + b // 2
        pairs.append((a, b))
    words = []
    for i in range(WORD_COUNT):
        if i == 0:
            words.append(rng.between(1, 65535) ** 2)
        elif i == 1:
            words.append(0xFFFFFFFF)
        else:
            words.append(rng.next() & 0xFFFFFFFF)
    ops = []
    for i in range(OP_COUNT):
        ops.append(rng.between(1, 255) if (i < RING_CAPACITY + 1 or rng.between(0, 99) < 55) else 0)

    lines = []
    for i, data in enumerate(bufs):
        lines.append("crc %d %04x" % (i, crc16(data)))
        lines.append("rle %d %s" % (i, "".join("%02x" % b for b in rle(data))))
    for i, (a, b) in enumerate(pairs):
        lines.append("mul %d %d" % (i, (a * b) >> 8))
        lines.append("div %d %d" % (i, div_round(a, b)))
    for i, w in enumerate(words):
        lines.append("sqrt %d %d" % (i, math.isqrt(w)))
        lines.append("pop %d %d" % (i, bin(w).count("1")))
        lines.append("rev %d %08x" % (i, bitrev32(w)))
        lines.append("rot %d %08x" % (i, rotl32(w, i % 31 + 1)))
    queue = []
    for i, op in enumerate(ops):
        if op == 0:
            lines.append("ring %d %s" % (i, queue.pop(0) if queue else "empty"))
        elif len(queue) == RING_CAPACITY:
            lines.append("ring %d full" % i)
        else:
            queue.append(op)
            lines.append("ring %d ok" % i)
    lines.append("END")

    header = "/* Inputs of one run; written by the grader for its own runs. */\n"
    header += "#define BUF_COUNT %d\n#define BUF_MAX %d\n" % (BUF_COUNT, BUF_MAX)
    header += "static const struct { uint8_t len; uint8_t data[BUF_MAX]; } BUFS[BUF_COUNT] = {\n"
    for data in bufs:
        header += "\t{ %d, { %s } },\n" % (len(data), ", ".join(str(b) for b in data))
    header += "};\n#define PAIR_COUNT %d\n" % PAIR_COUNT
    header += "static const struct { int32_t a; int32_t b; } PAIRS[PAIR_COUNT] = {\n"
    for a, b in pairs:
        header += "\t{ %d, %d },\n" % (a, b)
    header += "};\n#define WORD_COUNT %d\nstatic const uint32_t WORDS[WORD_COUNT] = {\n" % WORD_COUNT
    header += "".join("\t%du,\n" % w for w in words)
    header += "};\n#define OP_COUNT %d\nstatic const uint8_t OPS[OP_COUNT] = { %s };\n" % (
        OP_COUNT, ", ".join(str(o) for o in ops))
    return header, lines


def run(work, seed, vemu="vemu", clang="clang"):
    """Builds `work` with the vectors of `seed` and runs it. Returns
    ("built", console_lines) or ("build-failed", message)."""
    work = os.path.abspath(work)
    header, _ = vectors(seed)
    scratch = tempfile.mkdtemp(prefix="fw-")
    try:
        with open(os.path.join(work, "src", "vectors.h"), "w") as out:
            out.write(header)
        sources = sorted(
            os.path.join(work, "src", name) for name in os.listdir(os.path.join(work, "src")) if name.endswith(".c")
        )
        elf = os.path.join(scratch, "firmware.elf")
        built = subprocess.run(
            [clang] + CFLAGS + ["-Wl,-T," + os.path.join(HERE, "link-m33.ld")] + sources + ["-o", elf],
            capture_output=True, text=True, cwd=work, timeout=120,
        )
        if built.returncode != 0:
            return "build-failed", built.stderr.strip().splitlines()[-1:] and built.stderr.strip().splitlines()[-1]
        try:
            ran = subprocess.run(
                [vemu, "run", "--board", BOARD, "--kernel", elf, "--accel", "software", "--cycles", CYCLES],
                capture_output=True, stdin=subprocess.DEVNULL, timeout=60,
            )
        except subprocess.TimeoutExpired:
            return "built", []
        text = ran.stdout.decode("utf-8", "replace").replace("\r", "")
        return "built", [line.rstrip() for line in text.splitlines()]
    finally:
        subprocess.run(["rm", "-rf", scratch])


def differences(expected, got):
    """(line number, expected, got) for each console line that differs."""
    out = []
    for i in range(max(len(expected), len(got))):
        want = expected[i] if i < len(expected) else "<missing>"
        have = got[i] if i < len(got) else "<missing>"
        if want != have:
            out.append((i + 1, want, have))
    return out


def check(work, seed, vemu="vemu", clang="clang"):
    """("ok", []), or ("differs", [(line, expected, got), ...]) or
    ("build-failed", message)."""
    status, got = run(work, seed, vemu, clang)
    if status != "built":
        return status, got
    expected = vectors(seed)[1]
    diff = differences(expected, got)
    return ("ok", []) if not diff else ("differs", diff)


if __name__ == "__main__":
    # model.py vectors SEED OUT_HEADER   |   model.py check WORKDIR SEED
    if sys.argv[1] == "vectors":
        header, lines = vectors(int(sys.argv[2]))
        with open(sys.argv[3], "w") as out:
            out.write(header)
        print("\n".join(lines))
    else:
        status, detail = check(sys.argv[2], int(sys.argv[3]), os.environ.get("VEMU", "vemu"))
        if status == "differs":
            for line, want, have in detail[:4]:
                print("differs %d | %s | %s" % (line, want, have))
            print("count %d" % len(detail))
        else:
            print(status, detail if detail else "")
