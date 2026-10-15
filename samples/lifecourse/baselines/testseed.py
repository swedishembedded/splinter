# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""The seed the specs draw their synthetic cohorts and models from.

A default seed is random, so a spec that does not need one fixed value uses
this one: drawn once per run and printed, so a failure can be replayed by
substituting the printed value for SEED. A spec that does need a fixed seed
(a golden value, a fixture measured at one seed) writes the literal with a
`# seed:` reason instead of using this."""
import random
import sys

SEED = random.SystemRandom().getrandbits(31)
print(f"spec seed: {SEED}", file=sys.stderr)
