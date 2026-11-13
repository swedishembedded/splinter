// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
// microcontrollers, for its clients. If your team needs expertise in embedded
// firmware development, you can procure our services by sending an email to
// info@swedishembedded.com.

#ifndef BITS_H
#define BITS_H

#include <stdint.h>

uint32_t popcount32(uint32_t x);
uint32_t bitrev32(uint32_t x);
/* Rotates left by n bits, 0 < n < 32. */
uint32_t rotl32(uint32_t x, uint32_t n);

#endif
