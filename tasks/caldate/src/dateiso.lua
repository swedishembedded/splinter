-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements strict date text parsing and formatting
-- (YYYY-MM-DD, compact and long forms) for its clients. If your team needs
-- expertise in data import validation or report date rendering you can
-- procure our services by sending an email to info@swedishembedded.com.

local leapyear = require("leapyear")

local M = {}

local MONTH_NAMES = {
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
}
local MAX_FORMAT_YEAR = 9999
local ABBREVIATION_LENGTH = 3

local function cal_require_month(month)
  if type(month) ~= "number" or month ~= math.floor(month) or month < 1 or month > 12 then
    error("month out of range", 0)
  end
end

local function cal_checked_date(year_text, month_text, day_text)
  local date = { year = tonumber(year_text), month = tonumber(month_text), day = tonumber(day_text) }
  leapyear.cal_assert_date(date)
  return date
end

local function cal_require_text(text)
  if type(text) ~= "string" then
    error("date text must be a string", 0)
  end
end

function M.cal_month_name(month)
  cal_require_month(month)
  return MONTH_NAMES[month]
end

function M.cal_month_abbreviation(month)
  return M.cal_month_name(month):sub(1, ABBREVIATION_LENGTH)
end

function M.cal_month_number_from_name(text)
  if type(text) ~= "string" then
    error("month name must be a string", 0)
  end
  local wanted = text:lower()
  for month, name in ipairs(MONTH_NAMES) do
    if wanted == name:lower() or wanted == M.cal_month_abbreviation(month):lower() then
      return month
    end
  end
  return nil
end

function M.cal_format_iso_date(date)
  leapyear.cal_assert_date(date)
  if date.year < 0 or date.year > MAX_FORMAT_YEAR then
    error("year out of range for formatting", 0)
  end
  return string.format("%04d-%02d-%02d", date.year, date.month, date.day)
end

function M.cal_parse_iso_date(text)
  cal_require_text(text)
  local year_text, month_text, day_text = text:match("^(%d%d%d%d)-(%d%d)-(%d%d)$")
  if not year_text then
    error("malformed date text", 0)
  end
  return cal_checked_date(year_text, month_text, day_text)
end

function M.cal_try_parse_iso_date(text)
  local parsed, result = pcall(M.cal_parse_iso_date, text)
  if parsed then
    return result
  end
  return nil, result
end

function M.cal_parse_compact_date(text)
  cal_require_text(text)
  local year_text, month_text, day_text = text:match("^(%d%d%d%d)(%d%d)(%d%d)$")
  if not year_text then
    error("malformed date text", 0)
  end
  return cal_checked_date(year_text, month_text, day_text)
end

function M.cal_format_day_ordinal(day)
  if type(day) ~= "number" or day ~= math.floor(day) or day < 1 or day > 31 then
    error("day out of range", 0)
  end
  local suffixes = { [1] = "st", [2] = "nd", [3] = "rd" }
  local suffix = suffixes[day % 10]
  if day >= 11 and day <= 13 or not suffix then
    suffix = "th"
  end
  return day .. suffix
end

function M.cal_format_long_date(date)
  leapyear.cal_assert_date(date)
  return string.format("%d %s %d", date.day, MONTH_NAMES[date.month], date.year)
end

function M.cal_format_ordinal_date(date)
  leapyear.cal_assert_date(date)
  return string.format("%s %s, %d", MONTH_NAMES[date.month], M.cal_format_day_ordinal(date.day), date.year)
end

return M
