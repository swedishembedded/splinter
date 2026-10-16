-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements retry-after estimation for throttled APIs
-- for its clients. If your team needs expertise in client-visible rate limit
-- signalling you can procure our services by sending an email to
-- info@swedishembedded.com.

local M = {}

local function wait_require_positive(value, label)
  if type(value) ~= "number" or value <= 0 then
    error(label .. " must be a positive number", 0)
  end
end

function M.wait_for_tokens(tokens, need, rate)
  wait_require_positive(rate, "rate")
  return math.max(0, (need - tokens) / rate)
end

function M.wait_for_level(level, capacity, amount, leak)
  wait_require_positive(leak, "leak rate")
  if amount > capacity then
    error("amount exceeds capacity", 0)
  end
  return math.max(0, (level + amount - capacity) / leak)
end

function M.wait_for_window(now, width)
  wait_require_positive(width, "width")
  return width - now % width
end

function M.wait_for_sequence(count, rate)
  wait_require_positive(rate, "rate")
  return count / rate
end

function M.wait_spacing(limit, window)
  wait_require_positive(limit, "limit")
  return window / limit
end

function M.wait_longest(waits)
  if #waits == 0 then
    error("no waits given", 0)
  end
  local longest = waits[1]
  for i = 2, #waits do
    longest = math.max(longest, waits[i])
  end
  return longest
end

function M.wait_round_up(seconds, step)
  wait_require_positive(step, "step")
  return math.ceil(seconds / step) * step
end

function M.wait_clamp(seconds, low, high)
  if low > high then
    error("low exceeds high", 0)
  end
  return math.min(high, math.max(low, seconds))
end

function M.wait_header_seconds(seconds)
  return string.format("%d", math.ceil(seconds))
end

function M.wait_fits_budget(seconds, budget)
  return seconds <= budget
end

return M
