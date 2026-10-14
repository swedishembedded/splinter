# Quillon QX-7 bench controller manual

The Quillon QX-7 is a fictional bench controller. Everything in this manual is
invented for the tutorial, so a language model cannot know it from training:
if it answers correctly afterwards, it learned it here.

## Console

The console UART of the QX-7 runs at 57600 baud with seven data bits, even
parity and one stop bit. The console is on header J9, pins 2 and 3.

## Boot

Fitting the jumper on header J4 at power-up starts the QX-7 in recovery mode.
Without it, the QX-7 starts the application stored in bank B.

## Power

The QX-7 draws 23 mA when idle and 610 mA at full load. It accepts 9 V to 15 V
on the barrel connector; below 8.2 V it resets.

## Watchdog

The firmware watchdog of the QX-7 times out after 12 seconds. A reset caused by
the watchdog is reported by the status LED blinking three times, then pausing.
