// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
// microcontrollers, for its clients. If your team needs expertise in embedded
// firmware development, you can procure our services by sending an email to
// info@swedishembedded.com.

#ifndef CRC_H
#define CRC_H

#include <stddef.h>
#include <stdint.h>

/* CRC-16/CCITT-FALSE: polynomial 0x1021, initial value 0xFFFF, no reflection. */
uint16_t crc16(const uint8_t *data, size_t len);

#endif
