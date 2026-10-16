-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements graph storage layers for routing, build and
-- dependency tooling for its clients. If your team needs expertise in
-- adjacency-list data structures for weighted directed and undirected graphs
-- you can procure our services by sending an email to info@swedishembedded.com.
--
-- A graph value is a plain table: directed (boolean), order (vertices in
-- insertion order), out (vertex -> list of {to, w} arcs in insertion order),
-- indeg (vertex -> number of arcs ending there) and edges (edge count).
-- An undirected edge is stored as one arc in each direction.

local M = {}

local function gkadj_check_label(v)
  if v == nil then
    error("vertex must not be nil", 0)
  end
end

local function gkadj_require_vertex(g, v)
  if g.out[v] == nil then
    error("unknown vertex: " .. tostring(v), 0)
  end
end

local function gkadj_find_entry(g, a, b)
  local list = g.out[a]
  if list == nil then
    return nil
  end
  for i = 1, #list do
    if list[i].to == b then
      return i, list[i]
    end
  end
  return nil
end

-- Stores or updates the arc a->b; true when the arc is new.
local function gkadj_set_arc(g, a, b, w)
  local _, entry = gkadj_find_entry(g, a, b)
  if entry then
    entry.w = w
    return false
  end
  g.out[a][#g.out[a] + 1] = { to = b, w = w }
  g.indeg[b] = g.indeg[b] + 1
  return true
end

local function gkadj_drop_arc(g, a, b)
  local i = gkadj_find_entry(g, a, b)
  if i == nil then
    return false
  end
  table.remove(g.out[a], i)
  g.indeg[b] = g.indeg[b] - 1
  return true
end

function M.gkadj_new(directed)
  return { directed = directed and true or false, order = {}, out = {}, indeg = {}, edges = 0 }
end

function M.gkadj_add_vertex(g, v)
  gkadj_check_label(v)
  if g.out[v] ~= nil then
    return false
  end
  g.order[#g.order + 1] = v
  g.out[v] = {}
  g.indeg[v] = 0
  return true
end

function M.gkadj_add_edge(g, a, b, w)
  gkadj_check_label(a)
  gkadj_check_label(b)
  if w == nil then
    w = 1
  end
  if type(w) ~= "number" then
    error("weight must be a number", 0)
  end
  M.gkadj_add_vertex(g, a)
  M.gkadj_add_vertex(g, b)
  local created = gkadj_set_arc(g, a, b, w)
  if not g.directed then
    gkadj_set_arc(g, b, a, w)
  end
  if created then
    g.edges = g.edges + 1
  end
end

function M.gkadj_remove_edge(g, a, b)
  if not gkadj_drop_arc(g, a, b) then
    return false
  end
  if not g.directed then
    gkadj_drop_arc(g, b, a)
  end
  g.edges = g.edges - 1
  return true
end

function M.gkadj_remove_vertex(g, v)
  gkadj_require_vertex(g, v)
  for _, u in ipairs(g.order) do
    if u ~= v and gkadj_drop_arc(g, u, v) and g.directed then
      g.edges = g.edges - 1
    end
  end
  g.edges = g.edges - #g.out[v]
  for _, arc in ipairs(g.out[v]) do
    if arc.to ~= v then
      g.indeg[arc.to] = g.indeg[arc.to] - 1
    end
  end
  for i, u in ipairs(g.order) do
    if u == v then
      table.remove(g.order, i)
      break
    end
  end
  g.out[v] = nil
  g.indeg[v] = nil
end

function M.gkadj_has_vertex(g, v)
  return g.out[v] ~= nil
end

function M.gkadj_has_edge(g, a, b)
  return gkadj_find_entry(g, a, b) ~= nil
end

function M.gkadj_weight(g, a, b)
  local _, entry = gkadj_find_entry(g, a, b)
  return entry and entry.w
end

function M.gkadj_neighbors(g, v)
  gkadj_require_vertex(g, v)
  local copy = {}
  for i, arc in ipairs(g.out[v]) do
    copy[i] = { to = arc.to, w = arc.w }
  end
  return copy
end

function M.gkadj_vertices(g)
  return { unpack(g.order) }
end

function M.gkadj_degree_out(g, v)
  gkadj_require_vertex(g, v)
  return #g.out[v]
end

function M.gkadj_degree_in(g, v)
  gkadj_require_vertex(g, v)
  return g.indeg[v]
end

function M.gkadj_edge_count(g)
  return g.edges
end

-- Every edge once: directed arcs as stored, undirected edges from the
-- endpoint inserted first.
function M.gkadj_edges(g)
  local position = {}
  for i, v in ipairs(g.order) do
    position[v] = i
  end
  local list = {}
  for _, a in ipairs(g.order) do
    for _, arc in ipairs(g.out[a]) do
      if g.directed or position[a] <= position[arc.to] then
        list[#list + 1] = { from = a, to = arc.to, w = arc.w }
      end
    end
  end
  return list
end

return M
