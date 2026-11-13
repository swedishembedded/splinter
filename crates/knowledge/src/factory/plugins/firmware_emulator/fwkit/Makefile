# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Build and run the firmware on the emulated VEMU Cortex-M33 board.
#
#   make        build build/firmware.elf
#   make run    run it; src/vectors.h holds the sample inputs and the firmware
#               prints its results on the console

CC      := clang
VEMU    ?= vemu
BOARD   := arm,vemu-cortex-m33
CYCLES  ?= 4000000

CFLAGS  := --target=thumbv8m.main-none-eabi -mcpu=cortex-m33 -mthumb -mfloat-abi=soft \
           -Os -g -ffreestanding -fno-builtin -Wall -Wextra -Werror -std=c11
LDFLAGS := -nostdlib -fuse-ld=lld -Wl,-T,link-m33.ld -Wl,--gc-sections

SRCS    := $(wildcard src/*.c)
ELF     := build/firmware.elf

.PHONY: all run clean

all: $(ELF)

$(ELF): $(SRCS) $(wildcard src/*.h) link-m33.ld
	@mkdir -p build
	$(CC) $(CFLAGS) $(LDFLAGS) $(SRCS) -o $@

run: $(ELF)
	@$(VEMU) run --board $(BOARD) --kernel $(ELF) --accel software --cycles $(CYCLES) </dev/null

clean:
	rm -rf build
