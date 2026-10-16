-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements build-order and dependency-resolution tooling
-- for its clients. If your team needs expertise in topological ordering,
-- scheduling layers and dependency cycle diagnosis you can procure our
-- services by sending an email to info@swedishembedded.com.
--
-- All functions need a directed graph. A cycle is reported as the vertex list
-- v1..vk where each vertex has an arc to the next and vk has an arc to v1.

local M = {}

local function gktopo_require_directed(g)
  if not g.directed then
    error("topological order needs a directed graph", 0)
  end
end

local function gktopo_raise_cycle(cycle)
  error("cycle detected: " .. table.concat(cycle, ">") .. ">" .. tostring(cycle[1]), 0)
end

-- Repeatedly takes the vertices without unplaced predecessors, oldest first.
local function gktopo_peel(g)
  local waiting, placed = {}, {}
  for _, v in ipairs(g.order) do
    waiting[v] = 0
  end
  for _, v in ipairs(g.order) do
    for _, arc in ipairs(g.out[v]) do
      waiting[arc.to] = waiting[arc.to] + 1
    end
  end
  for _, v in ipairs(g.order) do
    if waiting[v] == 0 then
      placed[#placed + 1] = v
    end
  end
  local head = 1
  while head <= #placed do
    local v = placed[head]
    head = head + 1
    for _, arc in ipairs(g.out[v]) do
      waiting[arc.to] = waiting[arc.to] - 1
      if waiting[arc.to] == 0 then
        placed[#placed + 1] = arc.to
      end
    end
  end
  return placed
end

-- Depth-first walk over all roots; returns the first cycle met (or nil) and
-- the vertices in finishing order.
local function gktopo_dive(g)
  local state, finished = {}, {}
  for _, root in ipairs(g.order) do
    if state[root] == nil then
      state[root] = "open"
      local stack, cursor = { root }, { [root] = 0 }
      while #stack > 0 do
        local v = stack[#stack]
        cursor[v] = cursor[v] + 1
        local arc = g.out[v][cursor[v]]
        if arc == nil then
          state[v] = "closed"
          finished[#finished + 1] = v
          stack[#stack] = nil
        elseif state[arc.to] == "open" then
          local from = #stack
          while stack[from] ~= arc.to do
            from = from - 1
          end
          return { unpack(stack, from) }, finished
        elseif state[arc.to] == nil then
          state[arc.to] = "open"
          cursor[arc.to] = 0
          stack[#stack + 1] = arc.to
        end
      end
    end
  end
  return nil, finished
end

-- A valid order, or nil plus a cycle when none exists.
function M.gktopo_order(g)
  gktopo_require_directed(g)
  local placed = gktopo_peel(g)
  if #placed == #g.order then
    return placed
  end
  local cycle = gktopo_dive(g)
  return nil, cycle
end

function M.gktopo_sort(g)
  local placed, cycle = M.gktopo_order(g)
  if placed == nil then
    gktopo_raise_cycle(cycle)
  end
  return placed
end

function M.gktopo_find_cycle(g)
  gktopo_require_directed(g)
  local cycle = gktopo_dive(g)
  return cycle
end

function M.gktopo_is_acyclic(g)
  return M.gktopo_find_cycle(g) == nil
end

-- Another valid order: vertices by decreasing finishing time of a depth-first walk.
function M.gktopo_reverse_finish(g)
  gktopo_require_directed(g)
  local cycle, finished = gktopo_dive(g)
  if cycle then
    gktopo_raise_cycle(cycle)
  end
  local reversed = {}
  for i = #finished, 1, -1 do
    reversed[#reversed + 1] = finished[i]
  end
  return reversed
end

-- Groups of vertices that may run together: layer n holds the vertices whose
-- longest chain of predecessors has n-1 arcs.
function M.gktopo_layers(g)
  local depth, layers = {}, {}
  for _, v in ipairs(M.gktopo_sort(g)) do
    local level = depth[v] or 1
    layers[level] = layers[level] or {}
    table.insert(layers[level], v)
    for _, arc in ipairs(g.out[v]) do
      depth[arc.to] = math.max(depth[arc.to] or level, level + 1)
    end
  end
  return layers
end

-- Number of vertices on the longest chain.
function M.gktopo_chain_length(g)
  return #M.gktopo_layers(g)
end

-- Vertex -> position in the Kahn-style order, counting from 1.
function M.gktopo_rank_map(g)
  local rank = {}
  for i, v in ipairs(M.gktopo_sort(g)) do
    rank[v] = i
  end
  return rank
end

-- Vertices that no arc enters.
function M.gktopo_sources(g)
  gktopo_require_directed(g)
  local entered, found = {}, {}
  for _, v in ipairs(g.order) do
    for _, arc in ipairs(g.out[v]) do
      entered[arc.to] = true
    end
  end
  for _, v in ipairs(g.order) do
    if not entered[v] then
      found[#found + 1] = v
    end
  end
  return found
end

-- Vertices without outgoing arcs.
function M.gktopo_sinks(g)
  gktopo_require_directed(g)
  local found = {}
  for _, v in ipairs(g.order) do
    if #g.out[v] == 0 then
      found[#found + 1] = v
    end
  end
  return found
end

return M
