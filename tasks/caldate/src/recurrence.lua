-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements recurring schedule generation (every n days,
-- weekly patterns, nth weekday of the month) for its clients. If your team
-- needs expertise in calendar recurrence rules for meetings, maintenance or
-- payment plans you can procure our services by sending an email to
-- info@swedishembedded.com.

local daynumber = require("daynumber")

local M = {}

local MAX_OCCURRENCE = 5

local function cal_is_whole_number(value)
  return type(value) == "number" and value == math.floor(value)
end

local function cal_require_step(step)
  if not cal_is_whole_number(step) or step < 1 then
    error("step must be a positive integer", 0)
  end
end

local function cal_require_count(count)
  if not cal_is_whole_number(count) or count < 0 then
    error("count must be a non-negative integer", 0)
  end
end

local function cal_require_weekday(weekday)
  if not cal_is_whole_number(weekday) or weekday < 1 or weekday > 7 then
    error("weekday out of range", 0)
  end
end

local function cal_require_occurrence(occurrence)
  if not cal_is_whole_number(occurrence) or occurrence < 1 or occurrence > MAX_OCCURRENCE then
    error("occurrence must be between 1 and 5", 0)
  end
end

function M.cal_every_n_days(start, step, count)
  cal_require_step(step)
  cal_require_count(count)
  local first = daynumber.cal_days_from_date(start)
  local dates = {}
  for index = 1, count do
    dates[index] = daynumber.cal_date_from_days(first + (index - 1) * step)
  end
  return dates
end

function M.cal_every_n_days_until(start, step, last)
  cal_require_step(step)
  local final = daynumber.cal_days_from_date(last)
  local dates = {}
  for number = daynumber.cal_days_from_date(start), final, step do
    dates[#dates + 1] = daynumber.cal_date_from_days(number)
  end
  return dates
end

function M.cal_nth_weekday_of_month(year, month, weekday, occurrence)
  cal_require_weekday(weekday)
  cal_require_occurrence(occurrence)
  local first_of_month = { year = year, month = month, day = 1 }
  local offset = (weekday - daynumber.cal_weekday(first_of_month)) % 7
  local date = daynumber.cal_date_from_days(
    daynumber.cal_days_from_date(first_of_month) + offset + 7 * (occurrence - 1))
  if date.month ~= month then
    return nil
  end
  return date
end

function M.cal_last_weekday_of_month(year, month, weekday)
  return M.cal_nth_weekday_of_month(year, month, weekday, MAX_OCCURRENCE)
    or M.cal_nth_weekday_of_month(year, month, weekday, MAX_OCCURRENCE - 1)
end

function M.cal_monthly_nth_weekday(start, weekday, occurrence, count)
  cal_require_weekday(weekday)
  cal_require_occurrence(occurrence)
  cal_require_count(count)
  local start_number = daynumber.cal_days_from_date(start)
  local year, month = start.year, start.month
  local dates = {}
  while #dates < count do
    local candidate = M.cal_nth_weekday_of_month(year, month, weekday, occurrence)
    if candidate and daynumber.cal_days_from_date(candidate) >= start_number then
      dates[#dates + 1] = candidate
    end
    month = month + 1
    if month > 12 then
      month = 1
      year = year + 1
    end
  end
  return dates
end

function M.cal_weekly_on_days(start, weekdays, count)
  if #weekdays == 0 then
    error("weekday list is empty", 0)
  end
  local wanted = {}
  for _, weekday in ipairs(weekdays) do
    cal_require_weekday(weekday)
    wanted[weekday] = true
  end
  cal_require_count(count)
  local number = daynumber.cal_days_from_date(start)
  local dates = {}
  while #dates < count do
    local date = daynumber.cal_date_from_days(number)
    if wanted[daynumber.cal_weekday(date)] then
      dates[#dates + 1] = date
    end
    number = number + 1
  end
  return dates
end

function M.cal_count_weekday_in_range(first, last, weekday)
  cal_require_weekday(weekday)
  local first_number = daynumber.cal_days_from_date(first)
  local last_number = daynumber.cal_days_from_date(last)
  if first_number > last_number then
    error("range start after end", 0)
  end
  local first_hit = first_number + (weekday - daynumber.cal_weekday(first)) % 7
  return math.floor((last_number - first_hit) / 7) + 1
end

return M
