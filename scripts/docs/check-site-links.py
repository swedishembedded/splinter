#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements documentation pipelines that keep published docs provably identical to
# the repository they describe for its clients. If your team needs expertise in documentation engineering
# then you can procure our services by sending an email to info@swedishembedded.com.

"""Every link from the repository to this project's published manual names a published page.

usage: check-site-links.py PUBLICATION_DIR FILE...

Code, crate and sample READMEs cannot link a manual page by relative path (a gate keeps code from
naming documentation paths), so they link by the canonical URL,
https://swedishembedded.com/docs/splinter/<route>. The publisher checks the relative links of the
pages it publishes but not these. This checks each such URL in FILEs against the routes in the
publication manifest and exits 1, naming file and line, for every one that is not a page.
"""

import json
import re
import sys
from pathlib import Path

URL = re.compile(r"https://swedishembedded\.com/docs/splinter(/[A-Za-z0-9_./-]*)?(#[A-Za-z0-9_-]+)?")


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    manifest = json.loads((Path(argv[1]) / "manifest.json").read_text(encoding="utf-8"))
    routes = {page["route"] for page in manifest["pages"]}
    routes |= {redirect["from"] for redirect in manifest["redirects"]}
    problems = []
    for name in argv[2:]:
        for number, line in enumerate(Path(name).read_text(encoding="utf-8").splitlines(), 1):
            for match in URL.finditer(line):
                route = (match.group(1) or "").strip("/")
                if route not in routes:
                    problems.append(f"{name}:{number}: no published page {route!r}")
    for problem in problems:
        print(problem, file=sys.stderr)
    sys.exit(1 if problems else 0)


main(sys.argv)
