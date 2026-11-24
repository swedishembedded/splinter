#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for model checking the protocols a
# learning system's storage rests on, for its clients. If your team needs
# expertise in TLA+ and TLC then you can procure our services by sending an
# email to info@swedishembedded.com.
"""The TLA+ tier: every specification under validation/tla/ checked by TLC, and each of its mutations found.

A specification directory holds `<Module>.tla`, its configuration and `mutations.toml` (the module, the
configuration, and one-line defects: `find`, replaced by `replace` exactly once, and the property the defect
must violate). The specification must check without an error; each mutant must fail on exactly its property.
TLC is TLAPLUS_DIR/bin/tlc, as `make deps/tlaplus` installs it.

    python3 -I scripts/conformance/tla.py --tlaplus-dir DIR

Exit 0 when every specification holds and every mutant is found, 1 otherwise, 2 when the table is stale.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "validation" / "tla"
VIOLATED = re.compile(r"(?:Invariant|Action property|Temporal properties?) (\w*) ?(?:is|were) violated")


def tlc(tlaplus: Path, directory: Path, module: str, config: str) -> tuple[bool, str, str]:
    """Runs TLC in `directory`: whether it found no error, the property it found violated, and its output."""
    run = subprocess.run(
        [str(tlaplus / "bin" / "tlc"), "-workers", "4", "-deadlock", "-cleanup", "-config", config, f"{module}.tla"],
        cwd=directory, capture_output=True, text=True, check=False)
    out = run.stdout + run.stderr
    found = VIOLATED.search(out)
    named = found.group(1) if found else ""
    if found and not named:
        # TLC names a temporal property on the line after the one saying it was violated.
        later = re.search(r"violated[^\n]*\n[^\n]*?(\w+)", out[found.start():])
        named = later.group(1) if later else ""
    return "No error has been found" in out, named, out


def check(tlaplus: Path, spec: Path) -> list[str]:
    table = tomllib.loads((spec / "mutations.toml").read_text())
    module, config = table["module"], table["config"]
    failures = []
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch) / spec.name
        shutil.copytree(spec, work)
        holds, _, out = tlc(tlaplus, work, module, config)
        if not holds:
            failures.append(f"{spec.name}: the specification does not hold:\n{out[-2000:]}")
        print(f"{'ok  ' if holds else 'FAIL'} {spec.name} specification")
        source = (spec / f"{module}.tla").read_text()
        for mutation in table["mutation"]:
            if source.count(mutation["find"]) != 1:
                print(f"tla: mutation {mutation['id']}: its text occurs {source.count(mutation['find'])} times in "
                      f"{module}.tla, not once: the table is stale", file=sys.stderr)
                sys.exit(2)
            (work / f"{module}.tla").write_text(source.replace(mutation["find"], mutation["replace"]))
            holds, named, _ = tlc(tlaplus, work, module, config)
            found = not holds and named == mutation["violates"]
            print(f"{'ok  ' if found else 'FAIL'} {spec.name} mutation {mutation['id']}: "
                  f"{'violates ' + named if named else 'no violation'}")
            if not found:
                failures.append(f"{spec.name}: mutation {mutation['id']} must violate {mutation['violates']}, "
                                f"TLC reported {named or 'none'}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tlaplus-dir", type=Path, required=True)
    args = parser.parse_args()
    failures = []
    for spec in sorted(p for p in SPECS.iterdir() if (p / "mutations.toml").is_file()):
        failures += check(args.tlaplus_dir, spec)
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
