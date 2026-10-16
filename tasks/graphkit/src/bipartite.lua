-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements assignment and pairing engines for rostering,
-- scheduling and marketplace matching for its clients. If your team needs
-- expertise in bipartite graph checks, two-colouring and maximum matching you
-- can procure our services by sending an email to info@swedishembedded.com.
--
-- Colours are 1 and 2. Each component is coloured from its first vertex in
-- vertex order, which always gets colour 1.

local M = {}

local function gkbip_require_undirected(g)
  if g.directed then
    error("bipartite check needs an undirected graph", 0)
  end
end

-- Returns the colours assigned so far and, if an edge joins two vertices of
-- the same colour, that edge as {a, b}.
local function gkbip_paint(g)
  gkbip_require_undirected(g)
  local colour = {}
  for _, root in ipairs(g.order) do
    if colour[root] == nil then
      colour[root] = 1
      local queue, head = { root }, 1
      while head <= #queue do
        local v = queue[head]
        head = head + 1
        for _, arc in ipairs(g.out[v]) do
          local seen = colour[arc.to]
          if seen == nil then
            colour[arc.to] = 3 - colour[v]
            queue[#queue + 1] = arc.to
          elseif seen == colour[v] then
            return colour, { v, arc.to }
          end
        end
      end
    end
  end
  return colour, nil
end

-- Tries to give left vertex v a partner, moving earlier pairings if needed.
local function gkbip_augment(g, v, visited, partner)
  for _, arc in ipairs(g.out[v]) do
    local u = arc.to
    if not visited[u] then
      visited[u] = true
      if partner[u] == nil or gkbip_augment(g, partner[u], visited, partner) then
        partner[u] = v
        return true
      end
    end
  end
  return false
end

function M.gkbip_colouring(g)
  local colour, conflict = gkbip_paint(g)
  if conflict then
    return nil
  end
  return colour
end

function M.gkbip_is_bipartite(g)
  local _, conflict = gkbip_paint(g)
  return conflict == nil
end

-- The first edge found whose ends have the same colour, or nil.
function M.gkbip_conflict(g)
  local _, conflict = gkbip_paint(g)
  return conflict
end

-- The two sides as vertex lists (colour 1, colour 2), or nil when not bipartite.
function M.gkbip_sides(g)
  local colour = M.gkbip_colouring(g)
  if colour == nil then
    return nil
  end
  local left, right = {}, {}
  for _, v in ipairs(g.order) do
    table.insert(colour[v] == 1 and left or right, v)
  end
  return left, right
end

-- A largest set of vertex-disjoint edges as {left, right} pairs, ordered by right vertex.
function M.gkbip_matching(g)
  local colour, conflict = gkbip_paint(g)
  if conflict then
    error("graph is not bipartite", 0)
  end
  local partner = {}
  for _, v in ipairs(g.order) do
    if colour[v] == 1 then
      gkbip_augment(g, v, {}, partner)
    end
  end
  local pairs_found = {}
  for _, u in ipairs(g.order) do
    if partner[u] ~= nil then
      pairs_found[#pairs_found + 1] = { left = partner[u], right = u }
    end
  end
  return pairs_found
end

function M.gkbip_matching_size(g)
  return #M.gkbip_matching(g)
end

return M
