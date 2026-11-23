# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Whether an order is ready to ship."""


def is_ready(flag, count):
    """Ready when it is flagged and holds something."""
    return bool(flag) and count > 0
