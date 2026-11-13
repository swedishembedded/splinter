// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
// microcontrollers, for its clients. If your team needs expertise in embedded
// firmware development, you can procure our services by sending an email to
// info@swedishembedded.com.

#include "bits.h"

uint32_t popcount32(uint32_t x)
{
	uint32_t n = 0;

	while (x != 0) {
		n += x & 1u;
		x >>= 1;
	}
	return n;
}

uint32_t bitrev32(uint32_t x)
{
	uint32_t r = 0;

	for (int i = 0; i < 32; i++) {
		r = (r << 1) | (x & 1u);
		x >>= 1;
	}
	return r;
}

uint32_t rotl32(uint32_t x, uint32_t n)
{
	return (x << n) | (x >> (32u - n));
}
