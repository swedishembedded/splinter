-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements reachability and fewest-step analysis for
-- networks, link graphs and workflow maps for its clients. If your team needs
-- expertise in unweighted shortest-hop queries and path reconstruction you can
-- procure our services by sending an email to info@swedishembedded.com.
--
-- Hop counts ignore arc weights: every arc counts as one step.

local M = {}

-- Breadth-first search returning hops (vertex -> steps), parent (vertex ->
-- predecessor on a fewest-step path) and the discovery order.
local function gkhops_search(g, source)
  if g.out[source] == nil then
    error("unknown vertex: " .. tostring(source), 0)
  end
  local hops, parent, order = { [source] = 0 }, {}, { source }
  local head = 1
  while head <= #order do
    local v = order[head]
    head = head + 1
    for _, arc in ipairs(g.out[v]) do
      if hops[arc.to] == nil then
        hops[arc.to] = hops[v] + 1
        parent[arc.to] = v
        order[#order + 1] = arc.to
      end
    end
  end
  return hops, parent, order
end

function M.gkhops_distance(g, source, target)
  local hops = gkhops_search(g, source)
  return hops[target]
end

function M.gkhops_all(g, source)
  local hops = gkhops_search(g, source)
  return hops
end

-- A fewest-step path as a vertex list, or nil when the target is unreachable.
function M.gkhops_path(g, source, target)
  local hops, parent = gkhops_search(g, source)
  if hops[target] == nil then
    return nil
  end
  local path = { target }
  while path[1] ~= source do
    table.insert(path, 1, parent[path[1]])
  end
  return path
end

-- The vertex discovered first among those farthest from source, and its hops.
function M.gkhops_farthest(g, source)
  local hops, _, order = gkhops_search(g, source)
  local best = source
  for _, v in ipairs(order) do
    if hops[v] > hops[best] then
      best = v
    end
  end
  return best, hops[best]
end

-- Vertices exactly `steps` away, in discovery order.
function M.gkhops_ring(g, source, steps)
  if type(steps) ~= "number" or steps < 0 then
    error("steps must be a non-negative number", 0)
  end
  local hops, _, order = gkhops_search(g, source)
  local ring = {}
  for _, v in ipairs(order) do
    if hops[v] == steps then
      ring[#ring + 1] = v
    end
  end
  return ring
end

-- How many vertices other than the source lie within `steps` steps.
function M.gkhops_reach_count(g, source, steps)
  local hops, _, order = gkhops_search(g, source)
  local count = 0
  for i = 2, #order do
    if hops[order[i]] <= steps then
      count = count + 1
    end
  end
  return count
end

-- How many different fewest-step paths join source to target.
function M.gkhops_count_paths(g, source, target)
  local hops, _, order = gkhops_search(g, source)
  if hops[target] == nil then
    return 0
  end
  local ways = { [source] = 1 }
  for _, v in ipairs(order) do
    for _, arc in ipairs(g.out[v]) do
      if hops[arc.to] == hops[v] + 1 then
        ways[arc.to] = (ways[arc.to] or 0) + ways[v]
      end
    end
  end
  return ways[target]
end

-- The longest fewest-step distance between any ordered pair that is joined.
function M.gkhops_diameter(g)
  local widest = 0
  for _, v in ipairs(g.order) do
    local _, steps = M.gkhops_farthest(g, v)
    widest = math.max(widest, steps)
  end
  return widest
end

return M
