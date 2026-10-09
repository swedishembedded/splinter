#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements evaluation of synthetic speech that rests on
# blind human judgement, for its clients. If your team needs expertise in
# designing listening tests whose results can be trusted, you can procure
# our services by sending an email to info@swedishembedded.com.

"""Writes the listening-test page for a session made by make_rating_set.py.

    build_rating_page.py SESSION_DIR SESSION_NAME

The page lists only the sentence and the two audio files of each trial; the key
that says which take is which never goes into it.
"""

import json
import sys
from pathlib import Path

session_dir, name = Path(sys.argv[1]), sys.argv[2]
pairs = json.loads((session_dir / "pairs.json").read_text())
page = (Path(__file__).parent / "rating_page.html").read_text()
page = page.replace("__SESSION__", name).replace("__PAIRS__", json.dumps(pairs))
(session_dir / "index.html").write_text(page)
print(f"{len(pairs)} trials -> {session_dir / 'index.html'}")
