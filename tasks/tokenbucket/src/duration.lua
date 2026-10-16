-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements human-readable time settings for operations
-- tooling for its clients. If your team needs expertise in duration strings
-- for limits, timeouts and quotas you can procure our services by sending an
-- email to info@swedishembedded.com.

local M = {}

local dur_unit_table = {ms = 1, s = 1000, m = 60000, h = 3600000, d = 86400000}
local dur_format_order = {{"d", 86400000}, {"h", 3600000}, {"m", 60000}, {"s", 1000}, {"ms", 1}}

function M.dur_unit_millis(unit)
  local millis = dur_unit_table[unit]
  if millis == nil then
    error("unknown unit: " .. tostring(unit), 0)
  end
  return millis
end

function M.dur_total_millis(text)
  if type(text) ~= "string" or text == "" then
    error("duration must be a non-empty string", 0)
  end
  local pos, total = 1, 0
  while pos <= #text do
    local digits, unit, after = string.match(text, "^(%d+)(%a+)()", pos)
    if digits == nil then
      error("malformed duration at position " .. pos, 0)
    end
    total = total + tonumber(digits) * M.dur_unit_millis(unit)
    pos = after
  end
  return total
end

function M.dur_parse(text)
  return M.dur_total_millis(text) / 1000
end

function M.dur_try_parse(text)
  local ok, result = pcall(M.dur_parse, text)
  if ok then
    return result
  end
  return nil, result
end

function M.dur_is_valid(text)
  return M.dur_try_parse(text) ~= nil
end

function M.dur_format(seconds)
  if type(seconds) ~= "number" or seconds < 0 then
    error("seconds must be a non-negative number", 0)
  end
  local left = math.floor(seconds * 1000 + 0.5)
  if left == 0 then
    return "0s"
  end
  local parts = {}
  for _, entry in ipairs(dur_format_order) do
    local count = math.floor(left / entry[2])
    if count > 0 then
      parts[#parts + 1] = count .. entry[1]
      left = left - count * entry[2]
    end
  end
  return table.concat(parts)
end

function M.dur_compare(a, b)
  local left, right = M.dur_total_millis(a), M.dur_total_millis(b)
  if left < right then
    return -1
  end
  if left > right then
    return 1
  end
  return 0
end

function M.dur_sum(list)
  local total = 0
  for _, text in ipairs(list) do
    total = total + M.dur_total_millis(text)
  end
  return total / 1000
end

return M
