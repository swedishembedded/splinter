-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements per-interval request quotas for its clients.
-- If your team needs expertise in fixed window counters and quota resets you
-- can procure our services by sending an email to info@swedishembedded.com.

local M = {}

local function fwc_require_positive(value, label)
  if type(value) ~= "number" or value <= 0 then
    error(label .. " must be a positive number", 0)
  end
end

local function fwc_roll(c)
  local now = c.clock()
  local start = math.floor(now / c.width) * c.width
  if start ~= c.start then
    c.start = start
    c.count = 0
  end
  return now
end

function M.fwc_new(limit, width, clock)
  fwc_require_positive(limit, "limit")
  fwc_require_positive(width, "width")
  if type(clock) ~= "function" then
    error("clock must be a function", 0)
  end
  return {limit = limit, width = width, clock = clock}
end

function M.fwc_hit(c, n)
  n = n or 1
  fwc_require_positive(n, "count")
  fwc_roll(c)
  if c.count + n > c.limit then
    return false, c.limit - c.count
  end
  c.count = c.count + n
  return true, c.limit - c.count
end

function M.fwc_count(c)
  fwc_roll(c)
  return c.count
end

function M.fwc_remaining(c)
  fwc_roll(c)
  return c.limit - c.count
end

function M.fwc_window_start(c)
  fwc_roll(c)
  return c.start
end

function M.fwc_resets_in(c)
  local now = fwc_roll(c)
  return c.start + c.width - now
end

function M.fwc_reset(c)
  c.count = 0
end

function M.fwc_is_exhausted(c)
  fwc_roll(c)
  return c.count >= c.limit
end

return M
