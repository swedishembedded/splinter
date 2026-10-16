-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements per-client limiter bookkeeping that does not
-- grow without bound for its clients. If your team needs expertise in
-- multi-tenant rate limiting and idle state eviction you can procure our
-- services by sending an email to info@swedishembedded.com.

local M = {}

function M.rlreg_new(factory, idle_ttl, clock)
  if type(factory) ~= "function" then
    error("factory must be a function", 0)
  end
  if type(idle_ttl) ~= "number" or idle_ttl <= 0 then
    error("idle ttl must be a positive number", 0)
  end
  if type(clock) ~= "function" then
    error("clock must be a function", 0)
  end
  return {factory = factory, ttl = idle_ttl, clock = clock, items = {}, used = {}}
end

function M.rlreg_get(r, key)
  if key == nil then
    error("key must not be nil", 0)
  end
  local item = r.items[key]
  if item == nil then
    item = r.factory(key)
    if item == nil then
      error("factory returned nothing for key " .. tostring(key), 0)
    end
    r.items[key] = item
  end
  r.used[key] = r.clock()
  return item
end

function M.rlreg_peek(r, key)
  return r.items[key]
end

function M.rlreg_has(r, key)
  return r.items[key] ~= nil
end

function M.rlreg_size(r)
  local total = 0
  for _ in pairs(r.items) do
    total = total + 1
  end
  return total
end

function M.rlreg_keys(r)
  local keys = {}
  for key in pairs(r.items) do
    keys[#keys + 1] = key
  end
  table.sort(keys)
  return keys
end

function M.rlreg_idle_for(r, key)
  local last = r.used[key]
  if last == nil then
    return nil
  end
  return r.clock() - last
end

function M.rlreg_sweep(r)
  local now = r.clock()
  local gone = {}
  for key, last in pairs(r.used) do
    if now - last >= r.ttl then
      gone[#gone + 1] = key
    end
  end
  table.sort(gone)
  for _, key in ipairs(gone) do
    r.items[key] = nil
    r.used[key] = nil
  end
  return #gone, gone
end

function M.rlreg_forget(r, key)
  local existed = r.items[key] ~= nil
  r.items[key] = nil
  r.used[key] = nil
  return existed
end

return M
