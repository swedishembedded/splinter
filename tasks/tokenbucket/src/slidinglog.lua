-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements exact rolling-window request limits for its
-- clients. If your team needs expertise in sliding window log rate limiting
-- you can procure our services by sending an email to info@swedishembedded.com.

local M = {}

local function swl_require_positive(value, label)
  if type(value) ~= "number" or value <= 0 then
    error(label .. " must be a positive number", 0)
  end
end

local function swl_prune(l)
  local now = l.clock()
  local cutoff = now - l.span
  while l.head <= l.tail and l.stamps[l.head] <= cutoff do
    l.stamps[l.head] = nil
    l.head = l.head + 1
  end
  return now
end

function M.swl_clear(l)
  l.stamps = {}
  l.head = 1
  l.tail = 0
end

function M.swl_new(limit, span, clock)
  swl_require_positive(limit, "limit")
  swl_require_positive(span, "span")
  if type(clock) ~= "function" then
    error("clock must be a function", 0)
  end
  local l = {limit = limit, span = span, clock = clock}
  M.swl_clear(l)
  return l
end

function M.swl_count(l)
  swl_prune(l)
  return l.tail - l.head + 1
end

function M.swl_hit(l, n)
  n = n or 1
  swl_require_positive(n, "count")
  local now = swl_prune(l)
  local count = l.tail - l.head + 1
  if count + n > l.limit then
    return false, l.limit - count
  end
  for _ = 1, n do
    l.tail = l.tail + 1
    l.stamps[l.tail] = now
  end
  return true, l.limit - count - n
end

function M.swl_oldest(l)
  swl_prune(l)
  return l.stamps[l.head]
end

function M.swl_newest(l)
  swl_prune(l)
  return l.stamps[l.tail]
end

function M.swl_retry_after(l, n)
  n = n or 1
  swl_require_positive(n, "count")
  if n > l.limit then
    error("count exceeds limit", 0)
  end
  local now = swl_prune(l)
  local excess = l.tail - l.head + 1 + n - l.limit
  if excess <= 0 then
    return 0
  end
  return l.stamps[l.head + excess - 1] + l.span - now
end

return M
