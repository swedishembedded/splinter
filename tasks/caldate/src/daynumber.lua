-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements exact day-count conversions between civil
-- dates and linear day numbers for its clients. If your team needs expertise
-- in scheduling, billing periods or timeline arithmetic you can procure our
-- services by sending an email to info@swedishembedded.com.

local M = {}

local WEEKDAY_NAMES = {
  "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}
local DAYS_PER_FOUR_CENTURIES = 146097
local EPOCH_SHIFT = 719468

local function cal_is_whole_number(value)
  return type(value) == "number" and value == math.floor(value)
end

local function cal_split_checked(date)
  if type(date) ~= "table" then
    error("date must be a table", 0)
  end
  if not (cal_is_whole_number(date.year) and cal_is_whole_number(date.month)
      and cal_is_whole_number(date.day)) then
    error("date fields must be integers", 0)
  end
  if date.month < 1 or date.month > 12 then
    error("month out of range", 0)
  end
  return date.year, date.month, date.day
end

-- Years are counted from March so the leap day closes the year.
local function cal_civil_from_number(number)
  local shifted = number + EPOCH_SHIFT
  local era = math.floor(shifted / DAYS_PER_FOUR_CENTURIES)
  local day_of_era = shifted - era * DAYS_PER_FOUR_CENTURIES
  local year_of_era = math.floor((day_of_era - math.floor(day_of_era / 1460)
    + math.floor(day_of_era / 36524) - math.floor(day_of_era / 146096)) / 365)
  local day_of_year = day_of_era - (365 * year_of_era + math.floor(year_of_era / 4)
    - math.floor(year_of_era / 100))
  local month_index = math.floor((5 * day_of_year + 2) / 153)
  local day = day_of_year - math.floor((153 * month_index + 2) / 5) + 1
  local month = month_index < 10 and month_index + 3 or month_index - 9
  local year = year_of_era + era * 400 + (month <= 2 and 1 or 0)
  return year, month, day
end

local function cal_number_from_civil(year, month, day)
  local shifted_year = month <= 2 and year - 1 or year
  local era = math.floor(shifted_year / 400)
  local year_of_era = shifted_year - era * 400
  local month_index = (month + 9) % 12
  local day_of_year = math.floor((153 * month_index + 2) / 5) + day - 1
  local day_of_era = year_of_era * 365 + math.floor(year_of_era / 4)
    - math.floor(year_of_era / 100) + day_of_year
  return era * DAYS_PER_FOUR_CENTURIES + day_of_era - EPOCH_SHIFT
end

function M.cal_days_from_date(date)
  local year, month, day = cal_split_checked(date)
  local number = cal_number_from_civil(year, month, day)
  local _, _, back_day = cal_civil_from_number(number)
  if back_day ~= day then
    error("day out of range", 0)
  end
  return number
end

function M.cal_date_from_days(number)
  if not cal_is_whole_number(number) then
    error("day number must be an integer", 0)
  end
  local year, month, day = cal_civil_from_number(number)
  return { year = year, month = month, day = day }
end

function M.cal_weekday(date)
  return (M.cal_days_from_date(date) + 3) % 7 + 1
end

function M.cal_weekday_name(date)
  return WEEKDAY_NAMES[M.cal_weekday(date)]
end

function M.cal_day_of_year(date)
  local number = M.cal_days_from_date(date)
  local first_of_year = { year = date.year, month = 1, day = 1 }
  return number - M.cal_days_from_date(first_of_year) + 1
end

function M.cal_date_from_year_day(year, ordinal)
  local first_number = M.cal_days_from_date({ year = year, month = 1, day = 1 })
  if not cal_is_whole_number(ordinal) then
    error("day of year out of range", 0)
  end
  local date = M.cal_date_from_days(first_number + ordinal - 1)
  if date.year ~= year then
    error("day of year out of range", 0)
  end
  return date
end

return M
