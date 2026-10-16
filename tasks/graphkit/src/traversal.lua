-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements dependency and network exploration tooling
-- for its clients. If your team needs expertise in graph traversal orders for
-- build planning, crawling or impact analysis you can procure our services by
-- sending an email to info@swedishembedded.com.
--
-- Every function reads a graph value (fields order and out) and visits the
-- arcs of a vertex in their stored order, so results are fully determined.

local M = {}

local function gkwalk_check_start(g, v)
  if g.out[v] == nil then
    error("unknown vertex: " .. tostring(v), 0)
  end
end

-- Appends the vertices reachable from start, nearest first, to order.
local function gkwalk_breadth_from(g, start, seen, order)
  seen[start] = true
  local head = #order + 1
  order[head] = start
  while head <= #order do
    local v = order[head]
    head = head + 1
    for _, arc in ipairs(g.out[v]) do
      if not seen[arc.to] then
        seen[arc.to] = true
        order[#order + 1] = arc.to
      end
    end
  end
end

-- Depth-first walk with an explicit stack; fills pre (on entry) and post
-- (on exit) in the order the visits happen.
local function gkwalk_depth_from(g, start, seen, pre, post)
  seen[start] = true
  pre[#pre + 1] = start
  local stack, cursor = { start }, { [start] = 0 }
  while #stack > 0 do
    local v = stack[#stack]
    local i = cursor[v] + 1
    cursor[v] = i
    local arc = g.out[v][i]
    if arc == nil then
      post[#post + 1] = v
      stack[#stack] = nil
    elseif not seen[arc.to] then
      seen[arc.to] = true
      pre[#pre + 1] = arc.to
      cursor[arc.to] = 0
      stack[#stack + 1] = arc.to
    end
  end
end

function M.gkwalk_breadth_order(g, start)
  gkwalk_check_start(g, start)
  local order = {}
  gkwalk_breadth_from(g, start, {}, order)
  return order
end

function M.gkwalk_depth_preorder(g, start)
  gkwalk_check_start(g, start)
  local pre = {}
  gkwalk_depth_from(g, start, {}, pre, {})
  return pre
end

function M.gkwalk_depth_postorder(g, start)
  gkwalk_check_start(g, start)
  local post = {}
  gkwalk_depth_from(g, start, {}, {}, post)
  return post
end

function M.gkwalk_breadth_levels(g, start)
  gkwalk_check_start(g, start)
  local levels, seen = { { start } }, { [start] = true }
  while true do
    local following = {}
    for _, v in ipairs(levels[#levels]) do
      for _, arc in ipairs(g.out[v]) do
        if not seen[arc.to] then
          seen[arc.to] = true
          following[#following + 1] = arc.to
        end
      end
    end
    if #following == 0 then
      return levels
    end
    levels[#levels + 1] = following
  end
end

function M.gkwalk_breadth_within(g, start, limit)
  if type(limit) ~= "number" or limit < 0 then
    error("depth limit must be a non-negative number", 0)
  end
  local levels = M.gkwalk_breadth_levels(g, start)
  local order = {}
  for depth = 1, math.min(limit + 1, #levels) do
    for _, v in ipairs(levels[depth]) do
      order[#order + 1] = v
    end
  end
  return order
end

function M.gkwalk_breadth_order_all(g)
  local seen, order = {}, {}
  for _, v in ipairs(g.order) do
    if not seen[v] then
      gkwalk_breadth_from(g, v, seen, order)
    end
  end
  return order
end

function M.gkwalk_depth_preorder_all(g)
  local seen, pre = {}, {}
  for _, v in ipairs(g.order) do
    if not seen[v] then
      gkwalk_depth_from(g, v, seen, pre, {})
    end
  end
  return pre
end

return M
