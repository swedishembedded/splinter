-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements model-checked verification of concurrent
-- protocols for its clients. If your team needs expertise in specifying
-- protocols in TLA+ and checking implementations against them, you can
-- procure our services by sending an email to info@swedishembedded.com.

-- explore_test.lua: tests for explore.lua. Run from anywhere:
--   lua5.1 samples/model-check/lua/tests/explore_test.lua
-- Prints "ok" at the end; exits non-zero when an assertion fails.

local dir = arg[0]:match("^(.*)/[^/]*$") or "."
local luadir = dir .. "/.."
local E = dofile(luadir .. "/explore.lua")

local function check(cond, msg)
  if not cond then
    io.stderr:write("FAIL: " .. msg .. "\n")
    os.exit(1)
  end
end

local function checkeq(got, want, msg)
  if got ~= want then
    io.stderr:write("FAIL: " .. msg .. ": got " .. tostring(got)
      .. ", want " .. tostring(want) .. "\n")
    os.exit(1)
  end
end

local function checkerr(fn, sub, msg)
  local ok, err = pcall(fn)
  if ok then
    io.stderr:write("FAIL: " .. msg .. ": expected an error, none raised\n")
    os.exit(1)
  end
  if not tostring(err):find(sub, 1, true) then
    io.stderr:write("FAIL: " .. msg .. ": error " .. tostring(err)
      .. " does not contain " .. sub .. "\n")
    os.exit(1)
  end
end

---------------------------------------------------------------------------
-- encode
---------------------------------------------------------------------------

-- Key ordering must not depend on insertion order.
checkeq(E.encode({ b = 1, a = 2 }), E.encode({ a = 2, b = 1 }),
  "encode key ordering")
checkeq(E.encode({ b = 1, a = 2 }), "{a=2,b=1}", "encode sorted keys")

-- Nested data.
checkeq(E.encode({ x = { y = { 1, true, 's"\\t' } } }),
  '{x={y=[1,true,"s\\"\\\\t"]}}', "encode nested")

-- Empty table.
checkeq(E.encode({}), "[]", "encode empty table")

-- Rejections.
checkerr(function() E.encode(function() end) end, "unencodable value: function",
  "encode rejects a function")
checkerr(function() E.encode(1.5) end, "unencodable value: number",
  "encode rejects 1.5")

---------------------------------------------------------------------------
-- counter fixture
---------------------------------------------------------------------------

local counter = dofile(dir .. "/fixtures/counter.lua")
local res = E.explore(counter)
checkeq(res.result, "ok", "counter result")
checkeq(res.states, 4, "counter states")
checkeq(res.transitions, 3, "counter transitions")
checkeq(res.depth, 3, "counter depth")

---------------------------------------------------------------------------
-- invariant violation with a trace
---------------------------------------------------------------------------

local badcounter = {
  name = "badcounter",
  init = function() return { x = 0 } end,
  actions = {
    {
      name = "inc",
      enabled = function(s, a) return s.x < 3 end,
      step = function(s, a) return { x = s.x + 1 } end,
    },
  },
  invariants = {
    { name = "Small", check = function(s) return s.x <= 1 end },
  },
}
res = E.explore(badcounter)
checkeq(res.result, "invariant-violated", "badcounter result")
checkeq(res.invariant, "Small", "badcounter invariant name")
checkeq(#res.trace, 3, "badcounter trace length")
checkeq(res.trace[1].state.x, 0, "badcounter trace init")
checkeq(res.trace[3].state.x, 2, "badcounter trace end")

---------------------------------------------------------------------------
-- runtime errors
---------------------------------------------------------------------------

local errmod = {
  name = "errmod",
  init = function() return { x = 0 } end,
  actions = {
    {
      name = "boom",
      enabled = function(s, a) return true end,
      step = function(s, a) error("kaboom") end,
    },
  },
  invariants = {},
}
res = E.explore(errmod)
checkeq(res.result, "runtime-error", "errmod result")
check(res.error:find("step", 1, true) ~= nil, "errmod message contains step")

local mutmod = {
  name = "mutmod",
  init = function() return { x = 0 } end,
  actions = {
    {
      name = "mut",
      enabled = function(s, a) return true end,
      step = function(s, a)
        s.x = s.x + 1
        return s
      end,
    },
  },
  invariants = {},
}
res = E.explore(mutmod)
checkeq(res.result, "runtime-error", "mutmod result")
check(res.error:find("modified", 1, true) ~= nil,
  "mutmod message contains modified")

---------------------------------------------------------------------------
-- bound exceeded
---------------------------------------------------------------------------

local unbounded = {
  name = "unbounded",
  init = function() return { x = 0 } end,
  actions = {
    {
      name = "inc",
      enabled = function(s, a) return true end,
      step = function(s, a) return { x = s.x + 1 } end,
    },
  },
  invariants = {},
}
res = E.explore(unbounded, { max_states = 5 })
checkeq(res.result, "bound-exceeded", "unbounded result")

---------------------------------------------------------------------------
-- collect_edges on the counter fixture
---------------------------------------------------------------------------

res = E.explore(counter, { collect_edges = true })
checkeq(res.result, "ok", "collect_edges result")
checkeq(#res.nodes, 4, "collect_edges nodes")
checkeq(#res.edges, 3, "collect_edges edges")

---------------------------------------------------------------------------
-- command line
---------------------------------------------------------------------------

-- invariants can be left out, and a violation in the initial state is reported
res = E.explore(badcounter, { check_invariants = false })
checkeq(res.result, "ok", "no invariants checked")
checkeq(res.states, 4, "no invariants checked: states")
local initbad = {
  name = "initbad",
  init = function() return { x = 5 } end,
  actions = {},
  invariants = { { name = "Small", check = function(s) return s.x <= 1 end } },
}
res = E.explore(initbad)
checkeq(res.result, "invariant-violated", "initial state violation")
checkeq(#res.trace, 1, "initial state violation trace")

-- the command line, run with the interpreter running this test
local lua = arg[-1] or "lua"
local explore = luadir .. "/explore.lua"
local fixture = dir .. "/fixtures/counter.lua"
local function cli(args)
  local pipe = io.popen(string.format("%q %q %q %s 2>&1; echo exit=$?", lua, explore, fixture, args))
  local out = pipe:read("*a")
  pipe:close()
  return out
end
local out = cli("")
check(out:find("result=ok", 1, true) ~= nil, "cli prints result=ok: " .. out)
check(out:find("states=4", 1, true) ~= nil, "cli prints states=4: " .. out)
check(out:find("exit=0", 1, true) ~= nil, "cli exits 0: " .. out)
out = cli("--max-states 2")
check(out:find("result=bound-exceeded", 1, true) ~= nil, "cli bound: " .. out)
check(out:find("exit=3", 1, true) ~= nil, "cli bound exit code: " .. out)
check(cli("--bogus"):find("exit=64", 1, true) ~= nil, "cli usage error exit code")

print("ok")
