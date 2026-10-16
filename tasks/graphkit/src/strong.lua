-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements module-cycle and feedback-loop analysis for
-- its clients. If your team needs expertise in strongly connected components
-- and condensation of directed dependency graphs you can procure our services
-- by sending an email to info@swedishembedded.com.
--
-- Components come from two depth-first passes: one over the graph to rank the
-- vertices by finishing time, one over the reversed arcs from the latest
-- finisher down. Components therefore appear in an order where every arc
-- between components points forward; members keep their vertex order.

local M = {}

local function gkscc_require_directed(g)
  if not g.directed then
    error("components need a directed graph", 0)
  end
end

-- vertex -> list of vertices with an arc into it, in vertex order.
local function gkscc_predecessors(g)
  local back = {}
  for _, v in ipairs(g.order) do
    back[v] = {}
  end
  for _, v in ipairs(g.order) do
    for _, arc in ipairs(g.out[v]) do
      table.insert(back[arc.to], v)
    end
  end
  return back
end

function M.gkscc_finish_order(g)
  gkscc_require_directed(g)
  local seen, finished = {}, {}
  for _, root in ipairs(g.order) do
    if not seen[root] then
      seen[root] = true
      local stack, cursor = { root }, { [root] = 0 }
      while #stack > 0 do
        local v = stack[#stack]
        cursor[v] = cursor[v] + 1
        local arc = g.out[v][cursor[v]]
        if arc == nil then
          finished[#finished + 1] = v
          stack[#stack] = nil
        elseif not seen[arc.to] then
          seen[arc.to] = true
          cursor[arc.to] = 0
          stack[#stack + 1] = arc.to
        end
      end
    end
  end
  return finished
end

function M.gkscc_components(g)
  local finished = M.gkscc_finish_order(g)
  local back = gkscc_predecessors(g)
  local position = {}
  for i, v in ipairs(g.order) do
    position[v] = i
  end
  local assigned, components = {}, {}
  for i = #finished, 1, -1 do
    local root = finished[i]
    if not assigned[root] then
      assigned[root] = true
      local members, head = { root }, 1
      while head <= #members do
        local v = members[head]
        head = head + 1
        for _, before in ipairs(back[v]) do
          if not assigned[before] then
            assigned[before] = true
            members[#members + 1] = before
          end
        end
      end
      table.sort(members, function(a, b)
        return position[a] < position[b]
      end)
      components[#components + 1] = members
    end
  end
  return components
end

-- vertex -> index of its component in gkscc_components.
function M.gkscc_component_map(g)
  local index = {}
  for i, members in ipairs(M.gkscc_components(g)) do
    for _, v in ipairs(members) do
      index[v] = i
    end
  end
  return index
end

function M.gkscc_component_count(g)
  return #M.gkscc_components(g)
end

function M.gkscc_is_strongly_connected(g)
  return #g.order > 0 and M.gkscc_component_count(g) == 1
end

function M.gkscc_same_component(g, a, b)
  local index = M.gkscc_component_map(g)
  if index[a] == nil or index[b] == nil then
    error("unknown vertex: " .. tostring(index[a] == nil and a or b), 0)
  end
  return index[a] == index[b]
end

-- Arcs between components, each pair once, in first-seen order.
function M.gkscc_condensation(g)
  local index = M.gkscc_component_map(g)
  local seen, arcs = {}, {}
  for _, v in ipairs(g.order) do
    for _, arc in ipairs(g.out[v]) do
      local from, to = index[v], index[arc.to]
      local key = from .. ">" .. to
      if from ~= to and not seen[key] then
        seen[key] = true
        arcs[#arcs + 1] = { from = from, to = to }
      end
    end
  end
  return arcs
end

-- The first component with the most members.
function M.gkscc_largest(g)
  local best = {}
  for _, members in ipairs(M.gkscc_components(g)) do
    if #members > #best then
      best = members
    end
  end
  return best
end

-- Vertices that lie on some cycle: in a component of two or more, or with a self-loop.
function M.gkscc_cyclic_vertices(g)
  local index = M.gkscc_component_map(g)
  local size = {}
  for _, i in pairs(index) do
    size[i] = (size[i] or 0) + 1
  end
  local found = {}
  for _, v in ipairs(g.order) do
    local looped = false
    for _, arc in ipairs(g.out[v]) do
      looped = looped or arc.to == v
    end
    if size[index[v]] > 1 or looped then
      found[#found + 1] = v
    end
  end
  return found
end

return M
