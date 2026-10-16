-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements output smoothing for bursty workloads for its
-- clients. If your team needs expertise in leaky bucket metering and traffic
-- policing you can procure our services by sending an email to
-- info@swedishembedded.com.

local M = {}

local function lkb_require_positive(value, label)
  if type(value) ~= "number" or value <= 0 then
    error(label .. " must be a positive number", 0)
  end
end

function M.lkb_new(capacity, leak_rate, clock)
  lkb_require_positive(capacity, "capacity")
  lkb_require_positive(leak_rate, "leak rate")
  if type(clock) ~= "function" then
    error("clock must be a function", 0)
  end
  return {capacity = capacity, leak = leak_rate, clock = clock, level = 0, stamp = clock()}
end

function M.lkb_level(b)
  local now = b.clock()
  local elapsed = math.max(0, now - b.stamp)
  b.level = math.max(0, b.level - elapsed * b.leak)
  b.stamp = now
  return b.level
end

function M.lkb_offer(b, n)
  n = n or 1
  lkb_require_positive(n, "amount")
  local level = M.lkb_level(b)
  if level + n > b.capacity then
    return false, level
  end
  b.level = level + n
  return true, b.level
end

function M.lkb_offer_partial(b, n)
  lkb_require_positive(n, "amount")
  local accepted = math.min(n, b.capacity - M.lkb_level(b))
  b.level = b.level + accepted
  return accepted
end

function M.lkb_headroom(b)
  return b.capacity - M.lkb_level(b)
end

function M.lkb_overflow_by(b, n)
  return math.max(0, M.lkb_level(b) + n - b.capacity)
end

function M.lkb_drain_time(b)
  return M.lkb_level(b) / b.leak
end

function M.lkb_is_empty(b)
  return M.lkb_level(b) == 0
end

function M.lkb_clear(b)
  b.level = 0
end

return M
