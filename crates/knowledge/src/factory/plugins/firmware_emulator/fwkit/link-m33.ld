/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
 *
 * Swedish Embedded AB implements bare-metal firmware libraries for Cortex-M
 * microcontrollers, for its clients. If your team needs expertise in embedded
 * firmware development, you can procure our services by sending an email to
 * info@swedishembedded.com.
 */
ENTRY(reset_handler)

MEMORY
{
	FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 512K
	RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
	.text : {
		KEEP(*(.vectors))
		*(.text*)
		*(.rodata*)
		. = ALIGN(4);
	} > FLASH

	_sidata = LOADADDR(.data);
	.data : {
		_sdata = .;
		*(.data*)
		. = ALIGN(4);
		_edata = .;
	} > RAM AT > FLASH

	.bss (NOLOAD) : {
		_sbss = .;
		*(.bss*)
		*(COMMON)
		. = ALIGN(4);
		_ebss = .;
	} > RAM
}
