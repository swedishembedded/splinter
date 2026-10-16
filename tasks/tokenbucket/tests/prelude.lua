-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Printing helpers that give Lua values one canonical text form for the tests.

function show(v)
  local t = type(v)
  if t == "number" then
    if v == math.huge then return "inf" end
    if v == -math.huge then return "-inf" end
    if v == math.floor(v) and math.abs(v) < 1e15 then return string.format("%d", v) end
    return string.format("%.10g", v)
  elseif t == "string" then
    return '"' .. v .. '"'
  elseif t == "function" then
    return "fn"
  elseif t == "table" then
    local parts, n = {}, #v
    for i = 1, n do parts[#parts + 1] = show(v[i]) end
    local keys = {}
    for k in pairs(v) do
      if not (type(k) == "number" and k >= 1 and k <= n and k == math.floor(k)) then
        keys[#keys + 1] = k
      end
    end
    table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
    for _, k in ipairs(keys) do parts[#parts + 1] = tostring(k) .. "=" .. show(v[k]) end
    return "{" .. table.concat(parts, ",") .. "}"
  end
  return tostring(v)
end

function say(...)
  local parts = {}
  for i = 1, select("#", ...) do parts[i] = show((select(i, ...))) end
  print(table.concat(parts, " "))
end

local function pack(...)
  return {n = select("#", ...), ...}
end

function try(f, ...)
  local r = pack(pcall(f, ...))
  if not r[1] then return "err:" .. tostring(r[2]) end
  local parts = {}
  for i = 2, r.n do parts[#parts + 1] = show(r[i]) end
  return "ok:" .. table.concat(parts, " ")
end

function fake(start)
  local c = {now = start or 0}
  c.read = function() return c.now end
  return c
end
