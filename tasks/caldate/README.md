<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# caldate

Civil calendar date arithmetic in Lua 5.1. A date is a plain table
`{year = 2024, month = 2, day = 29}`; every function returns new tables and
never mutates its arguments. Dates are proleptic Gregorian. Weekdays are
numbered 1 (Monday) to 7 (Sunday). Day numbers count days from 1970-01-01
(day 0), so earlier dates have negative numbers.

Errors are raised as plain strings without position information. The messages
below are part of the contract.

Common date errors (raised by every function that takes a date):
`date must be a table`, `date fields must be integers`, `month out of range`,
`day out of range`.

## Running the tests

The suite is Python `unittest` driving the `lua` interpreter, one process per
test, with only the standard library:

    python3 -m unittest

## Modules and public API

Modules live in `src/` and are loaded with `require("<module>")`.

### leapyear

- `cal_is_leap_year(year)` - Gregorian rule. Error: `year must be an integer`.
- `cal_days_in_month(year, month)` - 28 to 31. Errors: `year must be an
  integer`, `month out of range`.
- `cal_days_in_year(year)` - 365 or 366.
- `cal_month_lengths(year)` - list of the 12 month lengths.
- `cal_assert_date(date)` - returns nothing, raises the common date errors.
- `cal_is_valid_date(date)` - boolean, never raises.
- `cal_leap_years_between(first, last)` - leap years in the inclusive range.
  Error: `range start after end`.
- `cal_next_leap_year(year)` - the first leap year strictly after `year`.

### daynumber

- `cal_days_from_date(date)` - day number of a date (common date errors).
- `cal_date_from_days(number)` - inverse. Error: `day number must be an integer`.
- `cal_weekday(date)` - 1 to 7.
- `cal_weekday_name(date)` - `"Monday"` to `"Sunday"`.
- `cal_day_of_year(date)` - 1 to 366.
- `cal_date_from_year_day(year, ordinal)` - Error: `day of year out of range`.

### dateshift

- `cal_add_days(date, count)`, `cal_add_weeks(date, count)` - `count` may be
  negative. Error: `count must be an integer`.
- `cal_add_months(date, count)`, `cal_add_years(date, count)` - the day is
  clamped to the length of the target month (31 January plus one month is the
  last day of February; 29 February plus one year is 28 February).
- `cal_month_start(date)`, `cal_month_end(date)`.
- `cal_next_weekday_on_or_after(date, weekday)`,
  `cal_previous_weekday_on_or_before(date, weekday)` - Error: `weekday out of
  range`.

### datespan

- `cal_days_between(first, last)` - signed day count.
- `cal_compare_dates(left, right)` - -1, 0 or 1.
- `cal_weeks_and_days_between(first, last)` - two results, both carrying the
  sign of the span.
- `cal_whole_months_between(first, last)` - a month is complete when the later
  day-of-month is not smaller than the earlier one; negative when reversed.
- `cal_age_in_years(birth, on)` - whole years. Error: `date is before birth
  date`.
- `cal_is_date_within(date, first, last)` - inclusive range test. Error:
  `range start after end`.
- `cal_overlap_days(first_a, last_a, first_b, last_b)` - days shared by two
  inclusive ranges (0 when disjoint). Error: `range start after end`.
- `cal_earliest_date(dates)`, `cal_latest_date(dates)` - the first such entry
  on ties. Error: `no dates given`.

### isoweek

- `cal_iso_week_year(date)`, `cal_iso_week_number(date)`.
- `cal_iso_weeks_in_year(year)` - 52 or 53.
- `cal_iso_week_monday(year, week)` - Error: `week out of range`.
- `cal_iso_date_from_parts(year, week, weekday)` - Errors: `week out of
  range`, `weekday out of range`.
- `cal_iso_week_dates(year, week)` - list of 7 dates, Monday first.
- `cal_iso_week_label(date)` - for example `"2024-W09-4"`.

### dateiso

- `cal_month_name(month)`, `cal_month_abbreviation(month)` - Error: `month out
  of range`.
- `cal_month_number_from_name(text)` - full or three-letter name, any case;
  `nil` when unknown. Error: `month name must be a string`.
- `cal_format_iso_date(date)` - `YYYY-MM-DD`. Error: `year out of range for
  formatting` outside 0 to 9999.
- `cal_parse_iso_date(text)` - strict `YYYY-MM-DD`. Errors: `date text must be
  a string`, `malformed date text`, then the common date errors.
- `cal_try_parse_iso_date(text)` - the date, or `nil` and the error message.
- `cal_parse_compact_date(text)` - strict `YYYYMMDD`, same errors.
- `cal_format_day_ordinal(day)` - `"1st"`, `"12th"`, `"23rd"`. Error: `day out
  of range`.
- `cal_format_long_date(date)` - `"3 March 2025"`.
- `cal_format_ordinal_date(date)` - `"March 3rd, 2025"`.

### bizdays

A holiday set comes from `cal_holiday_set(dates)`; weekends are Saturday and
Sunday.

- `cal_is_weekend(date)`, `cal_is_holiday(holidays, date)`,
  `cal_is_business_day(holidays, date)`.
- `cal_next_business_day(holidays, date)`,
  `cal_previous_business_day(holidays, date)` - strictly after or before.
- `cal_roll_forward_to_business_day(holidays, date)` - the date itself when it
  is already a business day.
- `cal_add_business_days(holidays, date, count)` - negative counts go back; a
  count of 0 returns a copy of the date. Error: `count must be an integer`.
- `cal_business_days_between(holidays, first, last)` - business days in the
  half-open range `[first, last)`, negative when reversed.

### recurrence

- `cal_every_n_days(start, step, count)` - Errors: `step must be a positive
  integer`, `count must be a non-negative integer`.
- `cal_every_n_days_until(start, step, last)` - inclusive of `last`, empty when
  `last` precedes `start`.
- `cal_nth_weekday_of_month(year, month, weekday, occurrence)` - occurrence 1
  to 5; `nil` when the month has no such occurrence. Errors: `weekday out of
  range`, `occurrence must be between 1 and 5`.
- `cal_last_weekday_of_month(year, month, weekday)`.
- `cal_monthly_nth_weekday(start, weekday, occurrence, count)` - the next
  `count` occurrences on or after `start`, skipping months that lack one.
- `cal_weekly_on_days(start, weekdays, count)` - the next `count` dates on or
  after `start` whose weekday is listed. Error: `weekday list is empty`.
- `cal_count_weekday_in_range(first, last, weekday)` - inclusive range.

## Example

    package.path = "src/?.lua;" .. package.path
    local shift = require("dateshift")
    local iso = require("dateiso")
    local renewal = shift.cal_add_months(iso.cal_parse_iso_date("2024-01-31"), 1)
    print(iso.cal_format_iso_date(renewal))  --> 2024-02-29

Swedish Embedded AB implements calendar and scheduling logic for its clients.
If your team needs expertise in date arithmetic, business calendars or
recurring schedules you can procure our services by sending an email to
info@swedishembedded.com.
