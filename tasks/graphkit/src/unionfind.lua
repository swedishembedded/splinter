-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements grouping and equivalence-tracking components
-- for its clients. If your team needs expertise in disjoint-set (union-find)
-- structures for clustering, merging records or network connectivity you can
-- procure our services by sending an email to info@swedishembedded.com.
--
-- Disjoint sets with path halving and union by size. When two sets have the
-- same size the root of the first argument stays the root.

local M = {}

local function gkdsu_require(d, x)
  if d.parent[x] == nil then
    error("unknown element: " .. tostring(x), 0)
  end
end

function M.gkdsu_new()
  return { parent = {}, size = {}, order = {}, sets = 0 }
end

function M.gkdsu_add(d, x)
  if x == nil then
    error("element must not be nil", 0)
  end
  if d.parent[x] ~= nil then
    return false
  end
  d.parent[x] = x
  d.size[x] = 1
  d.order[#d.order + 1] = x
  d.sets = d.sets + 1
  return true
end

function M.gkdsu_from_list(list)
  local d = M.gkdsu_new()
  for _, x in ipairs(list) do
    M.gkdsu_add(d, x)
  end
  return d
end

function M.gkdsu_find(d, x)
  gkdsu_require(d, x)
  while d.parent[x] ~= x do
    d.parent[x] = d.parent[d.parent[x]]
    x = d.parent[x]
  end
  return x
end

-- True when two sets were merged, false when already together.
function M.gkdsu_union(d, a, b)
  local root_a, root_b = M.gkdsu_find(d, a), M.gkdsu_find(d, b)
  if root_a == root_b then
    return false
  end
  if d.size[root_a] < d.size[root_b] then
    root_a, root_b = root_b, root_a
  end
  d.parent[root_b] = root_a
  d.size[root_a] = d.size[root_a] + d.size[root_b]
  d.sets = d.sets - 1
  return true
end

function M.gkdsu_connected(d, a, b)
  return M.gkdsu_find(d, a) == M.gkdsu_find(d, b)
end

function M.gkdsu_size_of(d, x)
  return d.size[M.gkdsu_find(d, x)]
end

function M.gkdsu_set_count(d)
  return d.sets
end

function M.gkdsu_element_count(d)
  return #d.order
end

-- Number of links between x and its root, without compressing the path.
function M.gkdsu_depth_of(d, x)
  gkdsu_require(d, x)
  local links = 0
  while d.parent[x] ~= x do
    x = d.parent[x]
    links = links + 1
  end
  return links
end

-- Sets as lists in element-insertion order, ordered by their first member.
function M.gkdsu_groups(d)
  local slot, groups = {}, {}
  for _, x in ipairs(d.order) do
    local root = M.gkdsu_find(d, x)
    if slot[root] == nil then
      groups[#groups + 1] = {}
      slot[root] = #groups
    end
    table.insert(groups[slot[root]], x)
  end
  return groups
end

function M.gkdsu_members(d, x)
  local root = M.gkdsu_find(d, x)
  local members = {}
  for _, y in ipairs(d.order) do
    if M.gkdsu_find(d, y) == root then
      members[#members + 1] = y
    end
  end
  return members
end

return M
