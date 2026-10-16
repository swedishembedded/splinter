-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements cabling, pipeline and network-layout cost
-- optimisation for its clients. If your team needs expertise in minimum
-- spanning trees and bottleneck routing over weighted undirected graphs you
-- can procure our services by sending an email to info@swedishembedded.com.
--
-- Edges are taken lightest first (heaviest first for the maximum variants),
-- equal weights in the order the edges were stored, and kept when they join
-- two different trees of the growing forest.

local sets = require("unionfind")

local M = {}

local function gkmst_require_undirected(g)
  if g.directed then
    error("spanning tree needs an undirected graph", 0)
  end
end

-- Every edge once as {a, b, w, seq}, sorted by weight (descending when
-- `heaviest`), ties by storage order.
local function gkmst_sorted_edges(g, heaviest)
  gkmst_require_undirected(g)
  local position, edges = {}, {}
  for i, v in ipairs(g.order) do
    position[v] = i
  end
  for _, a in ipairs(g.order) do
    for _, arc in ipairs(g.out[a]) do
      if position[a] <= position[arc.to] then
        edges[#edges + 1] = { a = a, b = arc.to, w = arc.w, seq = #edges + 1 }
      end
    end
  end
  table.sort(edges, function(x, y)
    if x.w ~= y.w then
      return (x.w < y.w) ~= heaviest
    end
    return x.seq < y.seq
  end)
  return edges
end

-- Returns the chosen edges as {from, to, w}, their total weight and the
-- number of trees left in the forest.
local function gkmst_grow(g, heaviest)
  local forest = sets.gkdsu_from_list(g.order)
  local chosen, total = {}, 0
  for _, e in ipairs(gkmst_sorted_edges(g, heaviest)) do
    if sets.gkdsu_union(forest, e.a, e.b) then
      chosen[#chosen + 1] = { from = e.a, to = e.b, w = e.w }
      total = total + e.w
    end
  end
  return chosen, total, sets.gkdsu_set_count(forest)
end

-- Minimum spanning forest: edges and total weight.
function M.gkmst_forest(g)
  local chosen, total = gkmst_grow(g, false)
  return chosen, total
end

function M.gkmst_maximum_forest(g)
  local chosen, total = gkmst_grow(g, true)
  return chosen, total
end

-- Minimum spanning tree; the graph must be connected.
function M.gkmst_tree(g)
  local chosen, total, trees = gkmst_grow(g, false)
  if trees > 1 then
    error("graph is not connected: " .. trees .. " trees", 0)
  end
  return chosen, total
end

function M.gkmst_total_weight(g)
  local _, total = gkmst_grow(g, false)
  return total
end

function M.gkmst_tree_count(g)
  local _, _, trees = gkmst_grow(g, false)
  return trees
end

function M.gkmst_is_spanning_tree(g)
  return #g.order > 0 and M.gkmst_tree_count(g) == 1
end

-- Whether the minimum spanning forest contains the edge a-b.
function M.gkmst_contains_edge(g, a, b)
  for _, e in ipairs(M.gkmst_forest(g)) do
    if (e.from == a and e.to == b) or (e.from == b and e.to == a) then
      return true
    end
  end
  return false
end

-- Smallest possible heaviest edge on any path from a to b, or nil when none exists.
function M.gkmst_bottleneck(g, a, b)
  local edges = gkmst_sorted_edges(g, false)
  local forest = sets.gkdsu_from_list(g.order)
  if sets.gkdsu_connected(forest, a, b) then
    return 0
  end
  for _, e in ipairs(edges) do
    sets.gkdsu_union(forest, e.a, e.b)
    if sets.gkdsu_connected(forest, a, b) then
      return e.w
    end
  end
  return nil
end

return M
