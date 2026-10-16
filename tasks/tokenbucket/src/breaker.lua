-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements failure isolation between services for its
-- clients. If your team needs expertise in circuit breakers and resilient
-- service-to-service calls you can procure our services by sending an email
-- to info@swedishembedded.com.

local M = {}

local function cbk_require_count(value, label)
  if type(value) ~= "number" or value < 1 or value ~= math.floor(value) then
    error(label .. " must be a positive integer", 0)
  end
end

function M.cbk_new(opts)
  local threshold = opts.failure_threshold or 5
  local open_seconds = opts.open_seconds or 30
  local probes = opts.probe_limit or 1
  cbk_require_count(threshold, "failure threshold")
  cbk_require_count(probes, "probe limit")
  if type(open_seconds) ~= "number" or open_seconds <= 0 then
    error("open seconds must be a positive number", 0)
  end
  if type(opts.clock) ~= "function" then
    error("clock must be a function", 0)
  end
  return {threshold = threshold, open_seconds = open_seconds, probe_limit = probes,
          clock = opts.clock, state = "closed", failures = 0, inflight = 0, probe_ok = 0}
end

local function cbk_trip_open(b)
  b.state = "open"
  b.opened_at = b.clock()
  b.failures = 0
end

local function cbk_settle(b)
  if b.state == "open" and b.clock() - b.opened_at >= b.open_seconds then
    b.state = "half_open"
    b.inflight = 0
    b.probe_ok = 0
  end
end

function M.cbk_state(b)
  cbk_settle(b)
  return b.state
end

function M.cbk_allow(b)
  cbk_settle(b)
  if b.state == "closed" then
    return true
  end
  if b.state == "half_open" and b.inflight < b.probe_limit then
    b.inflight = b.inflight + 1
    return true
  end
  return false
end

function M.cbk_failure(b)
  cbk_settle(b)
  if b.state == "half_open" then
    cbk_trip_open(b)
  elseif b.state == "closed" then
    b.failures = b.failures + 1
    if b.failures >= b.threshold then
      cbk_trip_open(b)
    end
  end
end

function M.cbk_success(b)
  cbk_settle(b)
  if b.state == "half_open" then
    b.probe_ok = b.probe_ok + 1
    if b.probe_ok >= b.probe_limit then
      b.state = "closed"
    end
  elseif b.state == "closed" then
    b.failures = 0
  end
end

function M.cbk_failures(b)
  return b.failures
end

function M.cbk_retry_in(b)
  cbk_settle(b)
  if b.state ~= "open" then
    return 0
  end
  return b.opened_at + b.open_seconds - b.clock()
end

function M.cbk_trip(b)
  cbk_trip_open(b)
end

function M.cbk_reset(b)
  b.state = "closed"
  b.failures = 0
end

function M.cbk_call(b, f, ...)
  if not M.cbk_allow(b) then
    return nil, "circuit open"
  end
  local ok, result = pcall(f, ...)
  if ok then
    M.cbk_success(b)
    return true, result
  end
  M.cbk_failure(b)
  return false, result
end

return M
