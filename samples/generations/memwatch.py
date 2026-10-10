#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Samples the memory of a run once a second, for `generations.py report`.

    memwatch.py --pid PID --gpu INDEX --nvml LIBRARY --out FILE

Each line of FILE is `{"t", "gpu_used_mib": [per card], "rss_mib"}`: the card
memory in use as the driver (NVML) reports it - the whole card, so a run is
only measured fairly on a card nothing else uses - and the resident memory of
PID and everything it started. Stops when PID is gone."""

import argparse
import ctypes
import json
import os
import time


class Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


def descendants(root):
    """`root` and every process started under it, by walking /proc."""
    parents = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                stat = f.read()
            parents[int(name)] = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
    found = {root}
    grew = True
    while grew:
        grew = False
        for pid, parent in parents.items():
            if parent in found and pid not in found:
                found.add(pid)
                grew = True
    return found


def rss_mib(pids):
    total = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
        except OSError:
            continue
    return total // 1024


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--gpu", type=int, action="append", required=True)
    p.add_argument("--nvml", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    lib = ctypes.CDLL(args.nvml)
    lib.nvmlInit_v2()
    handles = []
    for index in args.gpu:
        handle = ctypes.c_void_p()
        lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle))
        handles.append(handle)
    with open(args.out, "a") as out:
        while os.path.exists(f"/proc/{args.pid}"):
            used = []
            for handle in handles:
                m = Memory()
                lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(m))
                used.append(m.used // 2**20)
            out.write(json.dumps({"t": time.time(), "gpu_used_mib": used,
                                  "rss_mib": rss_mib(descendants(args.pid))}) + "\n")
            out.flush()
            time.sleep(1)


if __name__ == "__main__":
    main()
