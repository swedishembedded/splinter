-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements impact analysis and access-propagation
-- engines for its clients. If your team needs expertise in transitive closure,
-- reachability queries and transitive reduction of dependency graphs you can
-- procure our services by sending an email to info@swedishembedded.com.
--
-- Reachability means a path of one or more arcs, so a vertex reaches itself
-- only through a cycle.

local M = {}

local function gkreach_require_vertex(g, v)
  if g.out[v] == nil then
    error("unknown vertex: " .. tostring(v), 0)
  end
end

-- Boolean matrix m[i][j] over vertex positions plus the position index,
-- completed by letting every vertex inherit what its successors reach.
local function gkreach_matrix(g)
  local index, n, m = {}, #g.order, {}
  for i, v in ipairs(g.order) do
    index[v] = i
    m[i] = {}
  end
  for i, v in ipairs(g.order) do
    for _, arc in ipairs(g.out[v]) do
      m[i][index[arc.to]] = true
    end
  end
  for k = 1, n do
    for i = 1, n do
      if m[i][k] then
        for j = 1, n do
          if m[k][j] then
            m[i][j] = true
          end
        end
      end
    end
  end
  return m, index
end

-- vertex -> list of the vertices it reaches, in vertex order.
function M.gkreach_closure(g)
  local m = gkreach_matrix(g)
  local closure = {}
  for i, v in ipairs(g.order) do
    closure[v] = {}
    for j, u in ipairs(g.order) do
      if m[i][j] then
        table.insert(closure[v], u)
      end
    end
  end
  return closure
end

-- The closure as a flat list of {from, to} pairs.
function M.gkreach_pairs(g)
  local closure, found = M.gkreach_closure(g), {}
  for _, v in ipairs(g.order) do
    for _, u in ipairs(closure[v]) do
      found[#found + 1] = { from = v, to = u }
    end
  end
  return found
end

function M.gkreach_pair_count(g)
  return #M.gkreach_pairs(g)
end

-- Vertices reachable from v by a search that does not build the matrix.
function M.gkreach_reachable_from(g, v)
  gkreach_require_vertex(g, v)
  local seen, stack = {}, { v }
  while #stack > 0 do
    local w = table.remove(stack)
    for _, arc in ipairs(g.out[w]) do
      if not seen[arc.to] then
        seen[arc.to] = true
        stack[#stack + 1] = arc.to
      end
    end
  end
  local found = {}
  for _, u in ipairs(g.order) do
    if seen[u] then
      found[#found + 1] = u
    end
  end
  return found
end

function M.gkreach_can_reach(g, from, to)
  gkreach_require_vertex(g, to)
  for _, u in ipairs(M.gkreach_reachable_from(g, from)) do
    if u == to then
      return true
    end
  end
  return false
end

-- Vertices from which v can be reached.
function M.gkreach_ancestors(g, v)
  gkreach_require_vertex(g, v)
  local m, index = gkreach_matrix(g)
  local found = {}
  for i, u in ipairs(g.order) do
    if m[i][index[v]] then
      found[#found + 1] = u
    end
  end
  return found
end

function M.gkreach_mutual(g, a, b)
  return M.gkreach_can_reach(g, a, b) and M.gkreach_can_reach(g, b, a)
end

-- Arcs that are not implied by a longer path, for an acyclic graph.
function M.gkreach_reduction(g)
  local m, index = gkreach_matrix(g)
  for i = 1, #g.order do
    if m[i][i] then
      error("reduction needs an acyclic graph", 0)
    end
  end
  local kept = {}
  for _, v in ipairs(g.order) do
    for _, arc in ipairs(g.out[v]) do
      local implied = false
      for _, other in ipairs(g.out[v]) do
        implied = implied or m[index[other.to]][index[arc.to]] == true
      end
      if not implied then
        kept[#kept + 1] = { from = v, to = arc.to }
      end
    end
  end
  return kept
end

return M
