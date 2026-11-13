// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
// microcontrollers, for its clients. If your team needs expertise in embedded
// firmware development, you can procure our services by sending an email to
// info@swedishembedded.com.

#include "fixed.h"

int32_t q8_mul(int32_t a, int32_t b)
{
	return (a * b) >> 8;
}

static uint32_t abs_u32(int32_t v)
{
	return v < 0 ? 0u - (uint32_t)v : (uint32_t)v;
}

int32_t div_round(int32_t a, int32_t b)
{
	uint32_t num = abs_u32(a);
	uint32_t den = abs_u32(b);
	uint32_t q = (num + den / 2u) / den;

	if ((a < 0) != (b < 0)) {
		return -(int32_t)q;
	}
	return (int32_t)q;
}

uint32_t isqrt32(uint32_t x)
{
	uint32_t result = 0;
	uint32_t bit = 1u << 30;

	while (bit > x) {
		bit >>= 2;
	}
	while (bit != 0) {
		if (x >= result + bit) {
			x -= result + bit;
			result = (result >> 1) + bit;
		} else {
			result >>= 1;
		}
		bit >>= 2;
	}
	return result;
}
