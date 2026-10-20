-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements model-checked verification of concurrent
-- protocols for its clients. If your team needs expertise in specifying
-- protocols in TLA+ and checking implementations against them, you can
-- procure our services by sending an email to info@swedishembedded.com.

-- tracegen_test.lua: tests for tracegen.lua. Run from anywhere:
--   lua5.1 samples/model-check/lua/tests/tracegen_test.lua
-- Prints "ok" at the end; exits non-zero when an assertion fails.

local dir = arg[0]:match("^(.*)/[^/]*$") or "."
local luadir = dir .. "/.."
local E = dofile(luadir .. "/explore.lua")
local T = dofile(luadir .. "/tracegen.lua")

local function checkeq(got, want, msg)
  if got ~= want then
    io.stderr:write("FAIL: " .. msg .. ": got " .. tostring(got)
      .. ", want " .. tostring(want) .. "\n")
    os.exit(1)
  end
end

local function checkerr(fn, sub, msg)
  local ok, err = pcall(fn)
  if ok or not tostring(err):find(sub, 1, true) then
    io.stderr:write("FAIL: " .. msg .. ": error " .. tostring(err)
      .. " does not contain " .. sub .. "\n")
    os.exit(1)
  end
end

-- tla_value
checkeq(T.tla_value(42), "42", "integer")
checkeq(T.tla_value(-3), "(-3)", "negative integer")
checkeq(T.tla_value(true), "TRUE", "true")
checkeq(T.tla_value(false), "FALSE", "false")
checkeq(T.tla_value('a"b\\c'), '"a\\"b\\\\c"', "string with quote and backslash")
checkeq(T.tla_value({}), "<<>>", "empty table")
checkeq(T.tla_value({ 1, 2, 3 }), "<<1, 2, 3>>", "array")
checkeq(T.tla_value({ b = { 1, { z = true } }, a = "x" }),
  '[a |-> "x", b |-> <<1, [z |-> TRUE]>>]', "nested record sorted by key")
checkerr(function() T.tla_value(1.5) end, "unencodable", "fraction")
checkerr(function() T.tla_value(print) end, "unencodable", "function")
checkerr(function() T.tla_value({ 1, a = 2 }) end, "unencodable", "mixed table")

-- generate on the counter fixture
local counter = dofile(dir .. "/fixtures/counter.lua")
local explored = E.explore(counter, { collect_edges = true })
checkeq(T.generate(counter, explored, "Counter"), [[
---- MODULE CounterTrace ----
EXTENDS Counter, Naturals, Sequences
VARIABLES pos, tr
tvars == <<vars, pos, tr>>
Nodes == <<
  [x |-> 0],
  [x |-> 1],
  [x |-> 2],
  [x |-> 3]
>>
Traces == {
  <<1, 2>>,
  <<1, 2, 3>>,
  <<1, 2, 3, 4>>
}
Match(i) == /\ x = Nodes[i].x
MatchNext(i) == /\ x' = Nodes[i].x
TInit == /\ Init /\ tr \in Traces /\ pos = 1 /\ Match(tr[1])
TStep == /\ pos < Len(tr) /\ Next /\ MatchNext(tr[pos + 1]) /\ pos' = pos + 1 /\ UNCHANGED tr
TDone == /\ pos = Len(tr) /\ UNCHANGED tvars
TNext == TStep \/ TDone
TSpec == TInit /\ [][TNext]_tvars
====
]], "counter trace module")

-- two variables, a diamond: node 4 has two parents
local diamond = {
  name = "diamond",
  init = function() return { a = 0, b = 0 } end,
  actions = {
    { name = "A", enabled = function(s) return s.a == 0 end,
      step = function(s) return { a = 1, b = s.b } end },
    { name = "B", enabled = function(s) return s.b == 0 end,
      step = function(s) return { a = s.a, b = 1 } end },
  },
  invariants = {},
}
local dexplored = E.explore(diamond, { collect_edges = true })
local text = T.generate(diamond, dexplored, "Diamond")
local want_traces = "Traces == {\n  <<1, 2>>,\n  <<1, 3>>,\n  <<1, 2, 4>>,\n  <<1, 3, 4>>\n}\n"
if not text:find(want_traces, 1, true) then
  io.stderr:write("FAIL: diamond traces\n" .. text)
  os.exit(1)
end
if not text:find("Match(i) == /\\ a = Nodes[i].a\n  /\\ b = Nodes[i].b\n", 1, true) then
  io.stderr:write("FAIL: diamond Match\n" .. text)
  os.exit(1)
end
if not text:find("MatchNext(i) == /\\ a' = Nodes[i].a\n  /\\ b' = Nodes[i].b\n", 1, true) then
  io.stderr:write("FAIL: diamond MatchNext\n" .. text)
  os.exit(1)
end

-- generate refuses an exploration that is not ok
local bad = { invariants = { { name = "Never", check = function() return false end } },
  init = counter.init, actions = counter.actions }
checkerr(function()
  T.generate(bad, E.explore(bad, { collect_edges = true }), "Counter")
end, "invariant-violated", "not ok exploration")

-- main with --no-invariants writes the trace module of a module whose invariant fails
local mod = os.tmpname()
local f = assert(io.open(mod, "w"))
f:write([[return {
  name = "bad",
  init = function() return { x = 0 } end,
  actions = { { name = "inc", enabled = function(s) return s.x < 2 end,
                step = function(s) return { x = s.x + 1 } end } },
  invariants = { { name = "Never", check = function(s) return s.x < 1 end } },
}]])
f:close()
local out = os.tmpname()
os.remove(out)
os.execute("mkdir " .. out)
-- main talks to stdout and stderr; keep the test's own output to its verdict
local real_print, real_stderr = print, io.stderr
print = function() end
io.stderr = { write = function() end }
local invariant_code = T.main({ mod, "Bad", out })
local free_code = T.main({ mod, "Bad", out, "--no-invariants" })
local usage_code = T.main({})
print, io.stderr = real_print, real_stderr
checkeq(invariant_code, 1, "main reports the invariant")
checkeq(free_code, 0, "main --no-invariants")
checkeq(usage_code, 64, "usage error")
local g = assert(io.open(out .. "/BadTrace.tla", "r"))
local body = g:read("*a")
g:close()
if not body:find("---- MODULE BadTrace ----", 1, true) then
  io.stderr:write("FAIL: written module\n" .. body)
  os.exit(1)
end
os.remove(out .. "/BadTrace.tla")
os.remove(out)
os.remove(mod)

print("ok")
