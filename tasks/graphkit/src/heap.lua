-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements scheduling and work-queue components for its
-- clients. If your team needs expertise in priority queues and binary heaps
-- for routing, scheduling or event processing you can procure our services by
-- sending an email to info@swedishembedded.com.
--
-- A binary min-heap ordered by (priority, insertion sequence): entries of
-- equal priority leave in the order they arrived.

local M = {}

local function gkheap_before(a, b)
  if a.priority ~= b.priority then
    return a.priority < b.priority
  end
  return a.seq < b.seq
end

local function gkheap_swap(items, i, j)
  items[i], items[j] = items[j], items[i]
end

local function gkheap_sift_up(items, i)
  while i > 1 do
    local parent = math.floor(i / 2)
    if not gkheap_before(items[i], items[parent]) then
      return
    end
    gkheap_swap(items, i, parent)
    i = parent
  end
end

local function gkheap_sift_down(items, i, n)
  while true do
    local child = i * 2
    if child > n then
      return
    end
    if child < n and gkheap_before(items[child + 1], items[child]) then
      child = child + 1
    end
    if not gkheap_before(items[child], items[i]) then
      return
    end
    gkheap_swap(items, i, child)
    i = child
  end
end

local function gkheap_require_entry(h)
  if h.n == 0 then
    error("heap is empty", 0)
  end
end

function M.gkheap_new()
  return { items = {}, n = 0, seq = 0 }
end

function M.gkheap_push(h, value, priority)
  if value == nil then
    error("value must not be nil", 0)
  end
  if type(priority) ~= "number" then
    error("priority must be a number", 0)
  end
  h.n = h.n + 1
  h.seq = h.seq + 1
  h.items[h.n] = { value = value, priority = priority, seq = h.seq }
  gkheap_sift_up(h.items, h.n)
  return h.n
end

function M.gkheap_pop(h)
  gkheap_require_entry(h)
  local top = h.items[1]
  h.items[1] = h.items[h.n]
  h.items[h.n] = nil
  h.n = h.n - 1
  gkheap_sift_down(h.items, 1, h.n)
  return top.value, top.priority
end

function M.gkheap_peek(h)
  gkheap_require_entry(h)
  return h.items[1].value, h.items[1].priority
end

function M.gkheap_size(h)
  return h.n
end

function M.gkheap_is_empty(h)
  return h.n == 0
end

function M.gkheap_pop_many(h, count)
  if type(count) ~= "number" or count < 0 then
    error("count must be a non-negative number", 0)
  end
  local values = {}
  while #values < count and h.n > 0 do
    values[#values + 1] = M.gkheap_pop(h)
  end
  return values
end

function M.gkheap_drain(h)
  return M.gkheap_pop_many(h, h.n)
end

-- Builds a heap from a list of {value, priority} pairs.
function M.gkheap_from_pairs(pairs_list)
  local h = M.gkheap_new()
  for _, pair in ipairs(pairs_list) do
    M.gkheap_push(h, pair[1], pair[2])
  end
  return h
end

return M
