-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements route-cost engines for logistics, network and
-- latency planning for its clients. If your team needs expertise in weighted
-- shortest paths by priority-queue relaxation you can procure our services by
-- sending an email to info@swedishembedded.com.
--
-- Single-source shortest paths over non-negative arc weights. Vertices leave
-- the queue nearest first; ties are settled in the order they were queued.

local heap = require("heap")

local M = {}

local function gkpath_require_vertex(g, v)
  if g.out[v] == nil then
    error("unknown vertex: " .. tostring(v), 0)
  end
end

-- Returns dist (vertex -> cost), parent (vertex -> predecessor) and the
-- vertices in the order they were settled.
local function gkpath_relax_all(g, source)
  gkpath_require_vertex(g, source)
  local dist, parent, settled, done = { [source] = 0 }, {}, {}, {}
  local queue = heap.gkheap_new()
  heap.gkheap_push(queue, source, 0)
  while not heap.gkheap_is_empty(queue) do
    local v, d = heap.gkheap_pop(queue)
    if not done[v] then
      done[v] = true
      settled[#settled + 1] = v
      for _, arc in ipairs(g.out[v]) do
        if arc.w < 0 then
          error("negative weight on edge " .. tostring(v) .. ">" .. tostring(arc.to), 0)
        end
        local through = d + arc.w
        if dist[arc.to] == nil or through < dist[arc.to] then
          dist[arc.to] = through
          parent[arc.to] = v
          heap.gkheap_push(queue, arc.to, through)
        end
      end
    end
  end
  return dist, parent, settled
end

function M.gkpath_distances(g, source)
  local dist = gkpath_relax_all(g, source)
  return dist
end

function M.gkpath_settle_order(g, source)
  local _, _, settled = gkpath_relax_all(g, source)
  return settled
end

function M.gkpath_parents(g, source)
  local _, parent = gkpath_relax_all(g, source)
  return parent
end

function M.gkpath_cost(g, source, target)
  gkpath_require_vertex(g, target)
  local dist = gkpath_relax_all(g, source)
  return dist[target]
end

-- The cheapest path as a vertex list plus its cost, or nil when unreachable.
function M.gkpath_route(g, source, target)
  gkpath_require_vertex(g, target)
  local dist, parent = gkpath_relax_all(g, source)
  if dist[target] == nil then
    return nil
  end
  local path = { target }
  while path[1] ~= source do
    table.insert(path, 1, parent[path[1]])
  end
  return path, dist[target]
end

function M.gkpath_within(g, source, limit)
  local dist, _, settled = gkpath_relax_all(g, source)
  local near = {}
  for _, v in ipairs(settled) do
    if dist[v] <= limit then
      near[#near + 1] = v
    end
  end
  return near
end

-- The first `count` vertices after the source in settle order.
function M.gkpath_nearest(g, source, count)
  local _, _, settled = gkpath_relax_all(g, source)
  local found = {}
  for i = 2, math.min(count + 1, #settled) do
    found[#found + 1] = settled[i]
  end
  return found
end

-- The last vertex settled and its cost.
function M.gkpath_farthest(g, source)
  local dist, _, settled = gkpath_relax_all(g, source)
  local last = settled[#settled]
  return last, dist[last]
end

return M
