// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
// microcontrollers, for its clients. If your team needs expertise in embedded
// firmware development, you can procure our services by sending an email to
// info@swedishembedded.com.

#ifndef CONSOLE_H
#define CONSOLE_H

#include <stdint.h>

void console_puts(const char *s);
void console_put_i32(int32_t v);
void console_put_hex(uint32_t v, int digits);

#endif
