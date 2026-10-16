-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements business-day arithmetic with holiday lists
-- (settlement dates, delivery promises, SLA clocks) for its clients. If your
-- team needs expertise in working-day calendars you can procure our services
-- by sending an email to info@swedishembedded.com.

local daynumber = require("daynumber")

local M = {}

local FIRST_WEEKEND_DAY = 6

local function cal_is_open_number(holidays, number)
  local date = daynumber.cal_date_from_days(number)
  return not M.cal_is_weekend(date) and not M.cal_is_holiday(holidays, date)
end

local function cal_walk_to_open_day(holidays, date, step)
  local number = daynumber.cal_days_from_date(date) + step
  while not cal_is_open_number(holidays, number) do
    number = number + step
  end
  return daynumber.cal_date_from_days(number)
end

function M.cal_holiday_set(dates)
  local holidays = {}
  for _, date in ipairs(dates) do
    holidays[daynumber.cal_days_from_date(date)] = true
  end
  return holidays
end

function M.cal_is_weekend(date)
  return daynumber.cal_weekday(date) >= FIRST_WEEKEND_DAY
end

function M.cal_is_holiday(holidays, date)
  return holidays[daynumber.cal_days_from_date(date)] == true
end

function M.cal_is_business_day(holidays, date)
  return cal_is_open_number(holidays, daynumber.cal_days_from_date(date))
end

function M.cal_next_business_day(holidays, date)
  return cal_walk_to_open_day(holidays, date, 1)
end

function M.cal_previous_business_day(holidays, date)
  return cal_walk_to_open_day(holidays, date, -1)
end

function M.cal_roll_forward_to_business_day(holidays, date)
  if M.cal_is_business_day(holidays, date) then
    return date
  end
  return M.cal_next_business_day(holidays, date)
end

function M.cal_add_business_days(holidays, date, count)
  if type(count) ~= "number" or count ~= math.floor(count) then
    error("count must be an integer", 0)
  end
  local result = date
  local step = count < 0 and -1 or 1
  for _ = 1, math.abs(count) do
    result = cal_walk_to_open_day(holidays, result, step)
  end
  return daynumber.cal_date_from_days(daynumber.cal_days_from_date(result))
end

function M.cal_business_days_between(holidays, first, last)
  local from = daynumber.cal_days_from_date(first)
  local to = daynumber.cal_days_from_date(last)
  local sign = 1
  if from > to then
    from, to, sign = to, from, -1
  end
  local count = 0
  for number = from, to - 1 do
    if cal_is_open_number(holidays, number) then
      count = count + 1
    end
  end
  return sign * count
end

return M
