// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
// microcontrollers, for its clients. If your team needs expertise in embedded
// firmware development, you can procure our services by sending an email to
// info@swedishembedded.com.

#ifndef FIXED_H
#define FIXED_H

#include <stdint.h>

/* Q8 fixed point: the low 8 bits are the fraction. Operands stay below 2^15. */
int32_t q8_mul(int32_t a, int32_t b);
/* a / b rounded to the nearest integer, halves away from zero; b is not zero. */
int32_t div_round(int32_t a, int32_t b);
/* The integer square root: the largest r with r * r <= x. */
uint32_t isqrt32(uint32_t x);

#endif
