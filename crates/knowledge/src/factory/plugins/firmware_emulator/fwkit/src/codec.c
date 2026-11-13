// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
// microcontrollers, for its clients. If your team needs expertise in embedded
// firmware development, you can procure our services by sending an email to
// info@swedishembedded.com.

#include "codec.h"

size_t rle_encode(const uint8_t *in, size_t n, uint8_t *out)
{
	size_t w = 0;
	size_t i = 0;

	while (i < n) {
		uint8_t value = in[i];
		size_t run = 1;

		while (i + run < n && in[i + run] == value && run < 255u) {
			run++;
		}
		out[w++] = (uint8_t)run;
		out[w++] = value;
		i += run;
	}
	return w;
}

void hex_encode(const uint8_t *in, size_t n, char *out)
{
	static const char digits[] = "0123456789abcdef";

	for (size_t i = 0; i < n; i++) {
		out[2 * i] = digits[in[i] >> 4];
		out[2 * i + 1] = digits[in[i] & 0xFu];
	}
}
