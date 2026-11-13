// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
// microcontrollers, for its clients. If your team needs expertise in embedded
// firmware development, you can procure our services by sending an email to
// info@swedishembedded.com.

#include "ring.h"

void ring_init(struct ring *r)
{
	r->head = 0;
	r->tail = 0;
	r->count = 0;
}

int ring_push(struct ring *r, uint8_t value)
{
	if (r->count == RING_CAPACITY) {
		return -1;
	}
	r->data[r->head] = value;
	r->head = (r->head + 1u) % RING_CAPACITY;
	r->count++;
	return 0;
}

int ring_pop(struct ring *r, uint8_t *value)
{
	if (r->count == 0) {
		return -1;
	}
	*value = r->data[r->tail];
	r->tail = (r->tail + 1u) % RING_CAPACITY;
	r->count--;
	return 0;
}
