-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Printing and graph-building helpers shared by the test host.

function show(v)
  local kind = type(v)
  if kind == "number" then
    return string.format("%.14g", v)
  end
  if kind ~= "table" then
    return tostring(v)
  end
  local keys, parts = {}, {}
  for k in pairs(v) do
    keys[#keys + 1] = k
  end
  if #keys == #v then
    for i = 1, #v do
      parts[i] = show(v[i])
    end
    return "[" .. table.concat(parts, ",") .. "]"
  end
  table.sort(keys, function(a, b)
    return tostring(a) < tostring(b)
  end)
  for i, k in ipairs(keys) do
    parts[i] = tostring(k) .. "=" .. show(v[k])
  end
  return "{" .. table.concat(parts, ",") .. "}"
end

function fail(f, ...)
  local ok, err = pcall(f, ...)
  if ok then
    return "no error"
  end
  return tostring(err)
end

-- G("d" or "u", "a>b a>c:2 +z") builds a graph value without the adjacency
-- module: tokens are arcs with an optional weight, or +name for a vertex.
local function vertex_of(g, v)
  if g.out[v] == nil then
    g.order[#g.order + 1] = v
    g.out[v] = {}
    g.indeg[v] = 0
  end
end

function G(kind, text)
  local g = { directed = kind == "d", order = {}, out = {}, indeg = {}, edges = 0 }
  for token in text:gmatch("%S+") do
    local extra = token:match("^%+(.+)$")
    if extra then
      vertex_of(g, extra)
    else
      local a, b, w = token:match("^([^>:]+)>([^>:]+):(.+)$")
      if not a then
        a, b = token:match("^([^>:]+)>([^>:]+)$")
      end
      vertex_of(g, a)
      vertex_of(g, b)
      w = tonumber(w) or 1
      g.out[a][#g.out[a] + 1] = { to = b, w = w }
      g.indeg[b] = g.indeg[b] + 1
      if kind == "u" and a ~= b then
        g.out[b][#g.out[b] + 1] = { to = a, w = w }
        g.indeg[a] = g.indeg[a] + 1
      end
      g.edges = g.edges + 1
    end
  end
  return g
end
