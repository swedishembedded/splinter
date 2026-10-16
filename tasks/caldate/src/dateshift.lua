-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements date shifting (days, weeks, months, years
-- with end-of-month clamping) for its clients. If your team needs expertise
-- in subscription renewals, due-date rules or contract term arithmetic you
-- can procure our services by sending an email to info@swedishembedded.com.

local leapyear = require("leapyear")
local daynumber = require("daynumber")

local M = {}

local function cal_require_count(count)
  if type(count) ~= "number" or count ~= math.floor(count) then
    error("count must be an integer", 0)
  end
end

local function cal_require_weekday(weekday)
  if type(weekday) ~= "number" or weekday ~= math.floor(weekday) or weekday < 1 or weekday > 7 then
    error("weekday out of range", 0)
  end
end

function M.cal_add_days(date, count)
  cal_require_count(count)
  return daynumber.cal_date_from_days(daynumber.cal_days_from_date(date) + count)
end

function M.cal_add_weeks(date, count)
  cal_require_count(count)
  return M.cal_add_days(date, count * 7)
end

function M.cal_add_months(date, count)
  leapyear.cal_assert_date(date)
  cal_require_count(count)
  local total = date.year * 12 + (date.month - 1) + count
  local year = math.floor(total / 12)
  local month = total % 12 + 1
  local day = math.min(date.day, leapyear.cal_days_in_month(year, month))
  return { year = year, month = month, day = day }
end

function M.cal_add_years(date, count)
  cal_require_count(count)
  return M.cal_add_months(date, count * 12)
end

function M.cal_month_start(date)
  leapyear.cal_assert_date(date)
  return { year = date.year, month = date.month, day = 1 }
end

function M.cal_month_end(date)
  leapyear.cal_assert_date(date)
  return { year = date.year, month = date.month, day = leapyear.cal_days_in_month(date.year, date.month) }
end

function M.cal_next_weekday_on_or_after(date, weekday)
  cal_require_weekday(weekday)
  return M.cal_add_days(date, (weekday - daynumber.cal_weekday(date)) % 7)
end

function M.cal_previous_weekday_on_or_before(date, weekday)
  cal_require_weekday(weekday)
  return M.cal_add_days(date, -((daynumber.cal_weekday(date) - weekday) % 7))
end

return M
