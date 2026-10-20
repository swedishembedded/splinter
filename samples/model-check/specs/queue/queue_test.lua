-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements model-checked verification of concurrent
-- protocols for its clients. If your team needs expertise in specifying
-- protocols in TLA+ and checking implementations against them, you can
-- procure our services by sending an email to info@swedishembedded.com.

local dir = arg[0]:match("^(.*)/[^/]*$") or "."
local queue = dofile(dir .. "/queue.lua")

local function deepcopy(x)
  if type(x) ~= "table" then return x end
  local t = {}
  for k, v in pairs(x) do t[k] = deepcopy(v) end
  return t
end

local function deepequal(a, b)
  if type(a) ~= "table" or type(b) ~= "table" then return a == b end
  for k, v in pairs(a) do
    if not deepequal(v, b[k]) then return false end
  end
  for k in pairs(b) do
    if a[k] == nil then return false end
  end
  return true
end

local function findaction(name)
  for _, a in ipairs(queue.actions) do
    if a.name == name then return a end
  end
  error("no action " .. name)
end

local enqueue = findaction("enqueue")
local dequeue = findaction("dequeue")

local function checkall(s)
  for _, inv in ipairs(queue.invariants) do
    assert(inv.check(s), "invariant " .. inv.name .. " violated")
  end
end

-- init state and invariants on it
local s0 = queue.init()
assert(deepequal(s0, { q = {}, sent = { 0, 0 }, recv = { 0, 0 } }), "init state wrong")
checkall(s0)

-- enqueue refuses when the queue is full
local full = {
  q = { { p = 1, n = 1 }, { p = 2, n = 1 } },
  sent = { 1, 1 },
  recv = { 0, 0 },
}
for p = 1, 2 do
  assert(not enqueue.enabled(full, p), "enqueue enabled on full queue")
end

-- enqueue refuses when a producer has sent MAX_ITEMS
local maxed = {
  q = { { p = 1, n = 1 }, { p = 2, n = 1 } },
  sent = { 2, 1 },
  recv = { 0, 0 },
}
assert(not enqueue.enabled(maxed, 1), "enqueue enabled after MAX_ITEMS")
assert(not enqueue.enabled(maxed, 2), "enqueue enabled on full queue (maxed case)")

-- dequeue refuses on an empty queue
assert(not dequeue.enabled(s0, nil), "dequeue enabled on empty queue")

-- FIFO order: only enabled actions are taken, so the capacity of 2 holds.
local function take(action, state, arg)
  assert(action.enabled(state, arg), "action not enabled")
  return action.step(state, arg)
end
local order = {}
local function pop(state)
  local h = state.q[1]
  order[#order + 1] = h.p * 10 + h.n
  return take(dequeue, state, nil)
end
local s = s0
s = take(enqueue, s, 1)
s = take(enqueue, s, 2)
checkall(s)
s = pop(s)
s = take(enqueue, s, 1)
checkall(s)
s = pop(s)
s = pop(s)
checkall(s)
assert(deepequal(order, { 11, 21, 12 }), "items did not come out in FIFO order")
assert(deepequal(s, { q = {}, sent = { 2, 1 }, recv = { 2, 1 } }), "final state wrong")

-- step leaves its input state unchanged
local before = deepcopy(s0)
local s1 = enqueue.step(s0, 1)
assert(deepequal(s0, before), "enqueue.step modified its input state")
local s1_before = deepcopy(s1)
dequeue.step(s1, nil)
assert(deepequal(s1, s1_before), "dequeue.step modified its input state")

-- a hand-made state with three items in q violates "Bounded"
local overfull = {
  q = { { p = 1, n = 1 }, { p = 2, n = 1 }, { p = 1, n = 2 } },
  sent = { 2, 1 },
  recv = { 0, 0 },
}
local function invariant(name)
  for _, inv in ipairs(queue.invariants) do
    if inv.name == name then return inv end
  end
  error("no invariant " .. name)
end

local bounded = invariant("Bounded")
assert(not bounded.check(overfull), "Bounded should fail with three items")

-- a state with sent = {0,0} but one item in q violates "Conservation"
local lost = { q = { { p = 1, n = 1 } }, sent = { 0, 0 }, recv = { 0, 0 } }
local conservation = invariant("Conservation")
assert(not conservation.check(lost), "Conservation should fail with unaccounted item")

print("ok")
