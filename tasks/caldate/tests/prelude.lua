-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Printing prelude for the caldate tests: renders Lua values as canonical
-- text so the Python tests can assert on exact output.

local function show(value)
  local kind = type(value)
  if kind == "nil" then
    return "nil"
  elseif kind == "string" then
    return "'" .. value .. "'"
  elseif kind == "number" then
    return string.format("%.14g", value + 0)
  elseif kind ~= "table" then
    return tostring(value)
  end
  if type(value.year) == "number" and type(value.month) == "number" and type(value.day) == "number" then
    return string.format("<%04d-%02d-%02d>", value.year, value.month, value.day)
  end
  local parts = {}
  if #value > 0 or next(value) == nil then
    for i = 1, #value do
      parts[#parts + 1] = show(value[i])
    end
    return "[" .. table.concat(parts, ", ") .. "]"
  end
  local keys = {}
  for key in pairs(value) do
    keys[#keys + 1] = key
  end
  table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
  for _, key in ipairs(keys) do
    parts[#parts + 1] = tostring(key) .. "=" .. show(value[key])
  end
  return "{" .. table.concat(parts, ", ") .. "}"
end

function D(year, month, day)
  return { year = year, month = month, day = day }
end

function say(...)
  local shown = {}
  for i = 1, select("#", ...) do
    shown[i] = show((select(i, ...)))
  end
  print(#shown == 0 and "(none)" or table.concat(shown, ", "))
end

function run(f, ...)
  local function report(ok, ...)
    if ok then
      say(...)
    else
      print("error: " .. tostring((...)))
    end
  end
  report(pcall(f, ...))
end
