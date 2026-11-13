// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
// microcontrollers, for its clients. If your team needs expertise in embedded
// firmware development, you can procure our services by sending an email to
// info@swedishembedded.com.

#include <stdint.h>

extern uint32_t _estack, _sidata, _sdata, _edata, _sbss, _ebss;

int main(void);

static void default_handler(void)
{
	for (;;) {
	}
}

void reset_handler(void)
{
	const uint32_t *src = &_sidata;

	for (uint32_t *dst = &_sdata; dst < &_edata;) {
		*dst++ = *src++;
	}
	for (uint32_t *dst = &_sbss; dst < &_ebss;) {
		*dst++ = 0;
	}
	main();
	for (;;) {
	}
}

__attribute__((section(".vectors"), used)) static const void *const vectors[16] = {
	[0] = &_estack,
	[1] = reset_handler,
	[2 ... 15] = default_handler,
};
