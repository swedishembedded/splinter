// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
// microcontrollers, for its clients. If your team needs expertise in embedded
// firmware development, you can procure our services by sending an email to
// info@swedishembedded.com.

#include "crc.h"

uint16_t crc16(const uint8_t *data, size_t len)
{
	uint16_t crc = 0xFFFFu;

	for (size_t i = 0; i < len; i++) {
		crc ^= (uint16_t)((uint16_t)data[i] << 8);
		for (int bit = 0; bit < 8; bit++) {
			if (crc & 0x8000u) {
				crc = (uint16_t)((crc << 1) ^ 0x1021u);
			} else {
				crc = (uint16_t)(crc << 1);
			}
		}
	}
	return crc;
}
