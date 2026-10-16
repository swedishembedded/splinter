-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements retry pacing that protects overloaded
-- services for its clients. If your team needs expertise in exponential
-- backoff and jitter strategies you can procure our services by sending an
-- email to info@swedishembedded.com.

local M = {}

local function bko_require_positive_integer(value, label)
  if type(value) ~= "number" or value < 1 or value ~= math.floor(value) then
    error(label .. " must be a positive integer", 0)
  end
end

function M.bko_policy(base, factor, max_delay, jitter, max_attempts)
  if type(base) ~= "number" or base <= 0 then
    error("base must be a positive number", 0)
  end
  if type(factor) ~= "number" or factor < 1 then
    error("factor must be at least 1", 0)
  end
  if type(max_delay) ~= "number" or max_delay < base then
    error("max delay must not be below base", 0)
  end
  if jitter ~= nil and type(jitter) ~= "function" then
    error("jitter must be a function", 0)
  end
  bko_require_positive_integer(max_attempts, "max attempts")
  return {base = base, factor = factor, max_delay = max_delay, jitter = jitter,
          max_attempts = max_attempts}
end

function M.bko_delay(p, attempt)
  bko_require_positive_integer(attempt, "attempt")
  local raw = math.min(p.max_delay, p.base * p.factor ^ (attempt - 1))
  if p.jitter == nil then
    return raw
  end
  local jittered = p.jitter(raw, attempt)
  if type(jittered) ~= "number" or jittered < 0 then
    error("jitter must return a non-negative number", 0)
  end
  return math.min(p.max_delay, jittered)
end

function M.bko_schedule(p)
  local delays = {}
  for attempt = 1, p.max_attempts do
    delays[attempt] = M.bko_delay(p, attempt)
  end
  return delays
end

function M.bko_total(p)
  local total = 0
  for _, delay in ipairs(M.bko_schedule(p)) do
    total = total + delay
  end
  return total
end

function M.bko_start(p)
  return {policy = p, used = 0}
end

function M.bko_step(s)
  if s.used >= s.policy.max_attempts then
    return nil
  end
  s.used = s.used + 1
  return M.bko_delay(s.policy, s.used)
end

function M.bko_remaining(s)
  return s.policy.max_attempts - s.used
end

function M.bko_rewind(s)
  s.used = 0
end

function M.bko_fraction_jitter(fraction)
  if type(fraction) ~= "number" or fraction < 0 or fraction > 1 then
    error("fraction must be between 0 and 1", 0)
  end
  return function(delay)
    return delay * fraction
  end
end

function M.bko_offset_jitter(extra)
  return function(delay)
    return delay + extra
  end
end

function M.bko_attempt_jitter(step)
  return function(delay, attempt)
    return delay + step * attempt
  end
end

return M
