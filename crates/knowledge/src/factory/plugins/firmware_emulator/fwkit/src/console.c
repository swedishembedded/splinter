// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
// microcontrollers, for its clients. If your team needs expertise in embedded
// firmware development, you can procure our services by sending an email to
// info@swedishembedded.com.

#include "console.h"
#include "mcu.h"

static void putc_blocking(char c)
{
	while (UART0_FR & UART_FR_TXFF) {
	}
	UART0_DR = (uint8_t)c;
}

void console_puts(const char *s)
{
	while (*s) {
		putc_blocking(*s++);
	}
}

void console_put_i32(int32_t v)
{
	char digits[11];
	int n = 0;
	uint32_t u = (uint32_t)v;

	if (v < 0) {
		putc_blocking('-');
		u = 0u - u;
	}
	do {
		digits[n++] = (char)('0' + u % 10u);
		u /= 10u;
	} while (u != 0u);
	while (n > 0) {
		putc_blocking(digits[--n]);
	}
}

void console_put_hex(uint32_t v, int digits)
{
	for (int i = digits - 1; i >= 0; i--) {
		putc_blocking("0123456789abcdef"[(v >> (4 * i)) & 0xFu]);
	}
}
