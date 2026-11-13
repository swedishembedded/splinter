// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
// microcontrollers, for its clients. If your team needs expertise in embedded
// firmware development, you can procure our services by sending an email to
// info@swedishembedded.com.

#ifndef MCU_H
#define MCU_H

#include <stdint.h>

/* The console UART (PL011, uart0) of the emulated Cortex-M33 board. */
#define UART0_DR (*(volatile uint32_t *)0x4000C000u)
#define UART0_FR (*(volatile uint32_t *)0x4000C018u)
#define UART_FR_TXFF (1u << 5)

#endif
