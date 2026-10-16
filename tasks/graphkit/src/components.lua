-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements network resilience and partition analysis for
-- its clients. If your team needs expertise in connected components, isolated
-- nodes and bridge detection in undirected networks you can procure our
-- services by sending an email to info@swedishembedded.com.
--
-- Components are lists of vertices in breadth-first discovery order, ordered
-- by their first vertex in vertex order.

local M = {}

local function gkcomp_require_undirected(g)
  if g.directed then
    error("components need an undirected graph", 0)
  end
end

function M.gkcomp_components(g)
  gkcomp_require_undirected(g)
  local seen, found = {}, {}
  for _, root in ipairs(g.order) do
    if not seen[root] then
      seen[root] = true
      local members, head = { root }, 1
      while head <= #members do
        local v = members[head]
        head = head + 1
        for _, arc in ipairs(g.out[v]) do
          if not seen[arc.to] then
            seen[arc.to] = true
            members[#members + 1] = arc.to
          end
        end
      end
      found[#found + 1] = members
    end
  end
  return found
end

function M.gkcomp_count(g)
  return #M.gkcomp_components(g)
end

function M.gkcomp_sizes(g)
  local sizes = {}
  for i, members in ipairs(M.gkcomp_components(g)) do
    sizes[i] = #members
  end
  return sizes
end

-- Index of the component that holds v.
function M.gkcomp_component_of(g, v)
  for i, members in ipairs(M.gkcomp_components(g)) do
    for _, u in ipairs(members) do
      if u == v then
        return i
      end
    end
  end
  error("unknown vertex: " .. tostring(v), 0)
end

-- An empty graph and a single-component graph both count as connected.
function M.gkcomp_is_connected(g)
  return M.gkcomp_count(g) <= 1
end

-- The first component with the most vertices.
function M.gkcomp_largest(g)
  local best = {}
  for _, members in ipairs(M.gkcomp_components(g)) do
    if #members > #best then
      best = members
    end
  end
  return best
end

-- Vertices without any arc (a self-loop is an arc).
function M.gkcomp_isolated(g)
  gkcomp_require_undirected(g)
  local found = {}
  for _, v in ipairs(g.order) do
    if #g.out[v] == 0 then
      found[#found + 1] = v
    end
  end
  return found
end

local function gkcomp_lowlink(g, v, parent, st)
  st.clock = st.clock + 1
  st.depth[v] = st.clock
  st.low[v] = st.clock
  for _, arc in ipairs(g.out[v]) do
    local u = arc.to
    if st.depth[u] == nil then
      gkcomp_lowlink(g, u, v, st)
      st.low[v] = math.min(st.low[v], st.low[u])
      if st.low[u] > st.depth[v] then
        st.bridges[#st.bridges + 1] = { v, u }
      end
    elseif u ~= parent then
      st.low[v] = math.min(st.low[v], st.depth[u])
    end
  end
end

-- Edges whose removal would split a component, as {a, b} in the order a
-- depth-first walk finishes them.
function M.gkcomp_bridges(g)
  gkcomp_require_undirected(g)
  local st = { clock = 0, depth = {}, low = {}, bridges = {} }
  for _, root in ipairs(g.order) do
    if st.depth[root] == nil then
      gkcomp_lowlink(g, root, nil, st)
    end
  end
  return st.bridges
end

return M
