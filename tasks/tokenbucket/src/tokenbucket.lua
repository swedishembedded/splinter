-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements burst-tolerant request throttling for its
-- clients. If your team needs expertise in token bucket admission control you
-- can procure our services by sending an email to info@swedishembedded.com.

local M = {}

local function tkb_require_positive(value, label)
  if type(value) ~= "number" or value <= 0 then
    error(label .. " must be a positive number", 0)
  end
end

function M.tkb_new(capacity, rate, clock)
  tkb_require_positive(capacity, "capacity")
  tkb_require_positive(rate, "rate")
  if type(clock) ~= "function" then
    error("clock must be a function", 0)
  end
  return {capacity = capacity, rate = rate, clock = clock, tokens = capacity, stamp = clock()}
end

function M.tkb_refill(b)
  local now = b.clock()
  local elapsed = math.max(0, now - b.stamp)
  b.tokens = math.min(b.capacity, b.tokens + elapsed * b.rate)
  b.stamp = now
  return b.tokens
end

function M.tkb_take(b, n)
  n = n or 1
  tkb_require_positive(n, "count")
  if n > b.capacity then
    error("count exceeds capacity", 0)
  end
  local tokens = M.tkb_refill(b)
  if tokens < n then
    return false, tokens
  end
  b.tokens = tokens - n
  return true, b.tokens
end

function M.tkb_available(b)
  return M.tkb_refill(b)
end

function M.tkb_drain(b)
  local taken = M.tkb_refill(b)
  b.tokens = 0
  return taken
end

function M.tkb_give_back(b, n)
  tkb_require_positive(n, "count")
  b.tokens = math.min(b.capacity, M.tkb_refill(b) + n)
  return b.tokens
end

function M.tkb_fill_ratio(b)
  return M.tkb_refill(b) / b.capacity
end

function M.tkb_set_rate(b, rate)
  tkb_require_positive(rate, "rate")
  M.tkb_refill(b)
  b.rate = rate
end

function M.tkb_reset(b)
  b.tokens = b.capacity
end

return M
