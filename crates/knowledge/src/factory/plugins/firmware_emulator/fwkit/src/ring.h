// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
// microcontrollers, for its clients. If your team needs expertise in embedded
// firmware development, you can procure our services by sending an email to
// info@swedishembedded.com.

#ifndef RING_H
#define RING_H

#include <stdint.h>

#define RING_CAPACITY 8u

/* A fixed-capacity FIFO of bytes. */
struct ring {
	uint8_t data[RING_CAPACITY];
	uint32_t head;
	uint32_t tail;
	uint32_t count;
};

void ring_init(struct ring *r);
/* Appends a byte; returns 0, or -1 when the ring is full (nothing is stored). */
int ring_push(struct ring *r, uint8_t value);
/* Removes the oldest byte into *value; returns 0, or -1 when the ring is empty. */
int ring_pop(struct ring *r, uint8_t *value);

#endif
