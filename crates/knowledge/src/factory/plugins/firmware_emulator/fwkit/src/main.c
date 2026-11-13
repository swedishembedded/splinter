// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
// microcontrollers, for its clients. If your team needs expertise in embedded
// firmware development, you can procure our services by sending an email to
// info@swedishembedded.com.

#include <stddef.h>
#include <stdint.h>

#include "bits.h"
#include "codec.h"
#include "console.h"
#include "crc.h"
#include "fixed.h"
#include "ring.h"
#include "vectors.h"

static void line(const char *tag, int index)
{
	console_puts(tag);
	console_puts(" ");
	console_put_i32(index);
	console_puts(" ");
}

static void end_line(void)
{
	console_puts("\r\n");
}

int main(void)
{
	for (int i = 0; i < BUF_COUNT; i++) {
		const uint8_t *data = BUFS[i].data;
		size_t n = BUFS[i].len;
		uint8_t packed[2 * BUF_MAX];
		char text[2 * BUF_MAX + 1];

		line("crc", i);
		console_put_hex(crc16(data, n), 4);
		end_line();

		line("rle", i);
		size_t m = rle_encode(data, n, packed);
		hex_encode(packed, m, text);
		for (size_t k = 0; k < 2 * m; k++) {
			char one[2] = { text[k], 0 };
			console_puts(one);
		}
		end_line();
	}
	for (int i = 0; i < PAIR_COUNT; i++) {
		line("mul", i);
		console_put_i32(q8_mul(PAIRS[i].a, PAIRS[i].b));
		end_line();
		line("div", i);
		console_put_i32(div_round(PAIRS[i].a, PAIRS[i].b));
		end_line();
	}
	for (int i = 0; i < WORD_COUNT; i++) {
		line("sqrt", i);
		console_put_i32((int32_t)isqrt32(WORDS[i]));
		end_line();
		line("pop", i);
		console_put_i32((int32_t)popcount32(WORDS[i]));
		end_line();
		line("rev", i);
		console_put_hex(bitrev32(WORDS[i]), 8);
		end_line();
		line("rot", i);
		console_put_hex(rotl32(WORDS[i], (uint32_t)(i % 31) + 1u), 8);
		end_line();
	}
	struct ring ring;
	ring_init(&ring);
	for (int i = 0; i < OP_COUNT; i++) {
		line("ring", i);
		if (OPS[i] == 0) {
			uint8_t value = 0;

			if (ring_pop(&ring, &value) == 0) {
				console_put_i32(value);
			} else {
				console_puts("empty");
			}
		} else {
			if (ring_push(&ring, OPS[i]) == 0) {
				console_puts("ok");
			} else {
				console_puts("full");
			}
		}
		end_line();
	}
	console_puts("END\r\n");
	return 0;
}
