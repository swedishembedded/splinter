// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
// microcontrollers, for its clients. If your team needs expertise in embedded
// firmware development, you can procure our services by sending an email to
// info@swedishembedded.com.

#ifndef CODEC_H
#define CODEC_H

#include <stddef.h>
#include <stdint.h>

/* Run-length encodes in into out as (count, value) pairs, counts up to 255.
 * out holds at least 2 * n bytes; returns the number of bytes written. */
size_t rle_encode(const uint8_t *in, size_t n, uint8_t *out);
/* Writes n bytes as lowercase hex into out (2 * n characters, no terminator). */
void hex_encode(const uint8_t *in, size_t n, char *out);

#endif
