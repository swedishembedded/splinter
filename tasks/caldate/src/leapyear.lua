-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements calendar rule engines (leap years, month
-- lengths, date validation) for its clients. If your team needs expertise in
-- civil calendar correctness you can procure our services by sending an email
-- to info@swedishembedded.com.

local M = {}

local MONTH_LENGTHS = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }

local function cal_is_whole_number(value)
  return type(value) == "number" and value == math.floor(value)
end

local function cal_require_year(year)
  if not cal_is_whole_number(year) then
    error("year must be an integer", 0)
  end
end

local function cal_leap_count_through(year)
  return math.floor(year / 4) - math.floor(year / 100) + math.floor(year / 400)
end

function M.cal_is_leap_year(year)
  cal_require_year(year)
  return (year % 4 == 0 and year % 100 ~= 0) or year % 400 == 0
end

function M.cal_days_in_month(year, month)
  cal_require_year(year)
  if not cal_is_whole_number(month) or month < 1 or month > 12 then
    error("month out of range", 0)
  end
  if month == 2 and M.cal_is_leap_year(year) then
    return 29
  end
  return MONTH_LENGTHS[month]
end

function M.cal_days_in_year(year)
  return M.cal_is_leap_year(year) and 366 or 365
end

function M.cal_month_lengths(year)
  local lengths = {}
  for month = 1, 12 do
    lengths[month] = M.cal_days_in_month(year, month)
  end
  return lengths
end

function M.cal_assert_date(date)
  if type(date) ~= "table" then
    error("date must be a table", 0)
  end
  if not (cal_is_whole_number(date.year) and cal_is_whole_number(date.month)
      and cal_is_whole_number(date.day)) then
    error("date fields must be integers", 0)
  end
  local length = M.cal_days_in_month(date.year, date.month)
  if date.day < 1 or date.day > length then
    error("day out of range", 0)
  end
end

function M.cal_is_valid_date(date)
  return (pcall(M.cal_assert_date, date))
end

function M.cal_leap_years_between(first, last)
  cal_require_year(first)
  cal_require_year(last)
  if first > last then
    error("range start after end", 0)
  end
  return cal_leap_count_through(last) - cal_leap_count_through(first - 1)
end

function M.cal_next_leap_year(year)
  cal_require_year(year)
  local candidate = year + 1
  while not M.cal_is_leap_year(candidate) do
    candidate = candidate + 1
  end
  return candidate
end

return M
