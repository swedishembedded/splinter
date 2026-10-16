-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements week-numbering rules (week-based years,
-- week labels, week dates) for its clients. If your team needs expertise in
-- payroll weeks, production calendars or reporting by week you can procure
-- our services by sending an email to info@swedishembedded.com.

local daynumber = require("daynumber")

local M = {}

local function cal_thursday_of_week(date)
  local number = daynumber.cal_days_from_date(date)
  return daynumber.cal_date_from_days(number - (daynumber.cal_weekday(date) - 1) + 3)
end

function M.cal_iso_week_year(date)
  return cal_thursday_of_week(date).year
end

function M.cal_iso_week_number(date)
  local thursday = cal_thursday_of_week(date)
  return math.floor((daynumber.cal_day_of_year(thursday) - 1) / 7) + 1
end

function M.cal_iso_weeks_in_year(year)
  return M.cal_iso_week_number({ year = year, month = 12, day = 28 })
end

local function cal_first_monday(year)
  local fourth_of_january = { year = year, month = 1, day = 4 }
  return daynumber.cal_days_from_date(fourth_of_january) - (daynumber.cal_weekday(fourth_of_january) - 1)
end

function M.cal_iso_week_monday(year, week)
  local first_monday = cal_first_monday(year)
  if type(week) ~= "number" or week ~= math.floor(week) or week < 1 or week > M.cal_iso_weeks_in_year(year) then
    error("week out of range", 0)
  end
  return daynumber.cal_date_from_days(first_monday + (week - 1) * 7)
end

function M.cal_iso_date_from_parts(year, week, weekday)
  local monday = M.cal_iso_week_monday(year, week)
  if type(weekday) ~= "number" or weekday ~= math.floor(weekday) or weekday < 1 or weekday > 7 then
    error("weekday out of range", 0)
  end
  return daynumber.cal_date_from_days(daynumber.cal_days_from_date(monday) + weekday - 1)
end

function M.cal_iso_week_dates(year, week)
  local monday_number = daynumber.cal_days_from_date(M.cal_iso_week_monday(year, week))
  local dates = {}
  for offset = 0, 6 do
    dates[offset + 1] = daynumber.cal_date_from_days(monday_number + offset)
  end
  return dates
end

function M.cal_iso_week_label(date)
  return string.format("%04d-W%02d-%d", M.cal_iso_week_year(date), M.cal_iso_week_number(date),
    daynumber.cal_weekday(date))
end

return M
