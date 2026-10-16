-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements date comparison and interval measurement
-- (day counts, whole months, ages, range overlap) for its clients. If your
-- team needs expertise in tenure, billing period or SLA window calculations
-- you can procure our services by sending an email to
-- info@swedishembedded.com.

local daynumber = require("daynumber")

local M = {}

local function cal_range_numbers(first, last)
  local first_number = daynumber.cal_days_from_date(first)
  local last_number = daynumber.cal_days_from_date(last)
  if first_number > last_number then
    error("range start after end", 0)
  end
  return first_number, last_number
end

local function cal_pick_extreme(dates, direction)
  if #dates == 0 then
    error("no dates given", 0)
  end
  local best = dates[1]
  local best_number = daynumber.cal_days_from_date(best)
  for index = 2, #dates do
    local number = daynumber.cal_days_from_date(dates[index])
    if (number - best_number) * direction > 0 then
      best, best_number = dates[index], number
    end
  end
  return best
end

function M.cal_days_between(first, last)
  return daynumber.cal_days_from_date(last) - daynumber.cal_days_from_date(first)
end

function M.cal_compare_dates(left, right)
  local difference = M.cal_days_between(right, left)
  if difference < 0 then
    return -1
  end
  return difference > 0 and 1 or 0
end

function M.cal_weeks_and_days_between(first, last)
  local total = M.cal_days_between(first, last)
  local days = math.fmod(total, 7)
  return (total - days) / 7, days
end

function M.cal_whole_months_between(first, last)
  if M.cal_compare_dates(first, last) > 0 then
    return -M.cal_whole_months_between(last, first)
  end
  local months = (last.year - first.year) * 12 + last.month - first.month
  if last.day < first.day then
    months = months - 1
  end
  return months
end

function M.cal_age_in_years(birth, on)
  if M.cal_compare_dates(on, birth) < 0 then
    error("date is before birth date", 0)
  end
  return math.floor(M.cal_whole_months_between(birth, on) / 12)
end

function M.cal_is_date_within(date, first, last)
  local first_number, last_number = cal_range_numbers(first, last)
  local number = daynumber.cal_days_from_date(date)
  return number >= first_number and number <= last_number
end

function M.cal_overlap_days(first_a, last_a, first_b, last_b)
  local start_a, end_a = cal_range_numbers(first_a, last_a)
  local start_b, end_b = cal_range_numbers(first_b, last_b)
  return math.max(0, math.min(end_a, end_b) - math.max(start_a, start_b) + 1)
end

function M.cal_earliest_date(dates)
  return cal_pick_extreme(dates, -1)
end

function M.cal_latest_date(dates)
  return cal_pick_extreme(dates, 1)
end

return M
