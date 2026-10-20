-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements model-checked verification of concurrent
-- protocols for its clients. If your team needs expertise in specifying
-- protocols in TLA+ and checking implementations against them, you can
-- procure our services by sending an email to info@swedishembedded.com.

-- explore.lua: breadth-first state-space explorer for Lua state-machine
-- modules. A module is a table M with:
--   M.name        string
--   M.init()      returns the initial state
--   M.actions     array of { name, args?, enabled(s, a), step(s, a) }
--   M.invariants  array of { name, check(s) }
-- States are plain data (integers, booleans, strings, arrays, records).
--
-- The file is both a library (returns table E) and a command line program:
--   lua5.1 explore.lua <module.lua> [--max-states N]

local E = {}

---------------------------------------------------------------------------
-- E.encode: canonical compact text of plain data
---------------------------------------------------------------------------

-- Escape a string for double-quoted output.
local function escape(s)
  local out = {}
  for i = 1, #s do
    local c = s:sub(i, i)
    if c == "\\" then
      out[#out + 1] = "\\\\"
    elseif c == '"' then
      out[#out + 1] = '\\"'
    else
      out[#out + 1] = c
    end
  end
  return table.concat(out)
end

-- Collect the string keys of a table, sorted ascending.
local function sortedkeys(t)
  local ks = {}
  for k in pairs(t) do ks[#ks + 1] = k end
  table.sort(ks)
  return ks
end

local function encode(v)
  local tv = type(v)
  if tv == "number" then
    if v ~= v or v == math.huge or v == -math.huge then
      error("unencodable value: " .. tv)
    end
    if v ~= math.floor(v) then
      error("unencodable value: " .. tv)
    end
    return tostring(math.floor(v))
  elseif tv == "boolean" then
    return v and "true" or "false"
  elseif tv == "string" then
    return '"' .. escape(v) .. '"'
  elseif tv == "table" then
    -- Classify the keys: array part (1..n, no holes) and string keys.
    local n = 0
    local hasstring = false
    for k in pairs(v) do
      if type(k) == "number" then
        if k ~= math.floor(k) or k < 1 then
          error("unencodable value: " .. tv)
        end
        if k > n then n = k end
      elseif type(k) == "string" then
        hasstring = true
      else
        error("unencodable value: " .. tv)
      end
    end
    if hasstring and n > 0 then
      -- mixed array and string keys
      error("unencodable value: " .. tv)
    end
    if n > 0 then
      local parts = {}
      for i = 1, n do
        if v[i] == nil then
          -- nil hole in the array
          error("unencodable value: " .. tv)
        end
        parts[#parts + 1] = encode(v[i])
      end
      return "[" .. table.concat(parts, ",") .. "]"
    end
    if hasstring then
      local parts = {}
      for _, k in ipairs(sortedkeys(v)) do
        parts[#parts + 1] = k .. "=" .. encode(v[k])
      end
      return "{" .. table.concat(parts, ",") .. "}"
    end
    return "[]"
  end
  error("unencodable value: " .. tv)
end

E.encode = encode

-------------------------------------------------------------------------------
-- E.explore: breadth-first search over the module's state space
---------------------------------------------------------------------------

-- Runs fn(...) under pcall. Returns true and its value, or false and the
-- message prefixed with what was running.
local function guarded(what, fn, ...)
  local ok, res = pcall(fn, ...)
  if ok then return true, res end
  return false, what .. ": " .. tostring(res)
end

-- The shortest path to node `index` as trace steps: the initial state, then
-- one { action, arg, state } per step.
local function trace_to(index, states, parent)
  local steps = {}
  while parent[index] do
    local edge = parent[index]
    table.insert(steps, 1, { action = edge.action, arg = edge.arg, state = states[index] })
    index = edge.from
  end
  table.insert(steps, 1, { state = states[index] })
  return steps
end

-- Explores M breadth-first.
--   opts.max_states        stop with "bound-exceeded" beyond this many states (default 100000)
--   opts.collect_edges     also return nodes (states in discovery order) and edges
--   opts.check_invariants  false: do not check invariants (default true)
-- The result's `result` is "ok", "invariant-violated", "runtime-error" or "bound-exceeded".
function E.explore(M, opts)
  opts = opts or {}
  local max_states = opts.max_states or 100000
  local collect = opts.collect_edges or false
  local check_invariants = opts.check_invariants ~= false

  local result = { states = 0, transitions = 0, depth = 0 }
  if collect then
    result.nodes = {}
    result.edges = {}
  end
  local function fail(kind, message)
    result.result = kind
    if kind == "runtime-error" then result.error = message end
    return result
  end

  local states = {}  -- node index -> state
  local depth = {}   -- node index -> BFS depth
  local parent = {}  -- node index -> the edge that first reached it
  local seen = {}    -- encoded state -> node index

  -- Registers a new state as the next node and checks its invariants.
  -- Returns nil when fine, otherwise the finished result.
  local function discover(state, encoded, from_edge, node_depth)
    local index = result.states + 1
    result.states = index
    states[index] = state
    depth[index] = node_depth
    parent[index] = from_edge
    seen[encoded] = index
    if node_depth > result.depth then result.depth = node_depth end
    if collect then result.nodes[index] = state end
    if not check_invariants then return nil end
    for _, inv in ipairs(M.invariants or {}) do
      local ok, holds = guarded("check " .. inv.name, inv.check, state)
      if not ok then return fail("runtime-error", holds) end
      if type(holds) ~= "boolean" then
        return fail("runtime-error", "check " .. inv.name .. " returned a non-boolean")
      end
      if not holds then
        result.invariant = inv.name
        result.trace = trace_to(index, states, parent)
        return fail("invariant-violated")
      end
    end
    return nil
  end

  local ok, initial = guarded("init", M.init)
  if not ok then return fail("runtime-error", initial) end
  local encoded
  ok, encoded = pcall(encode, initial)
  if not ok then return fail("runtime-error", "init: " .. tostring(encoded)) end
  local stop = discover(initial, encoded, nil, 0)
  if stop then return stop end

  local head = 1
  while head <= result.states do
    local from = head
    head = head + 1
    local state = states[from]
    for _, act in ipairs(M.actions or {}) do
      local arglist = { n = 1 }  -- no `args`: one call with a nil argument
      if act.args then
        ok, arglist = guarded("args " .. act.name, act.args)
        if not ok then return fail("runtime-error", arglist) end
        arglist.n = #arglist
      end
      for k = 1, arglist.n do
        local arg = arglist[k]
        local enabled
        ok, enabled = guarded("enabled " .. act.name, act.enabled, state, arg)
        if not ok then return fail("runtime-error", enabled) end
        if type(enabled) ~= "boolean" then
          return fail("runtime-error", "enabled " .. act.name .. " returned a non-boolean")
        end
        if enabled then
          result.transitions = result.transitions + 1
          local before = encode(state)
          local next_state
          ok, next_state = guarded("step " .. act.name, act.step, state, arg)
          if not ok then return fail("runtime-error", next_state) end
          if encode(state) ~= before then
            return fail("runtime-error", "step " .. act.name .. " modified its input state")
          end
          ok, encoded = pcall(encode, next_state)
          if not ok then
            return fail("runtime-error", "step " .. act.name .. ": " .. tostring(encoded))
          end
          local to = seen[encoded]
          if not to then
            if result.states + 1 > max_states then return fail("bound-exceeded") end
            local edge = { from = from, action = act.name, arg = arg }
            stop = discover(next_state, encoded, edge, depth[from] + 1)
            to = result.states
            if collect then
              result.edges[#result.edges + 1] =
                { from = from, to = to, action = act.name, arg = arg }
            end
            if stop then return stop end
          elseif collect then
            result.edges[#result.edges + 1] =
              { from = from, to = to, action = act.name, arg = arg }
          end
        end
      end
    end
  end

  return fail("ok")
end

---------------------------------------------------------------------------
-- E.main: command line entry point
---------------------------------------------------------------------------

local USAGE = "usage: lua5.1 explore.lua <module.lua> [--max-states N] [--no-invariants]"

-- Returns module path and options, or nil when the command line is wrong.
local function parse_args(argv)
  local module
  local opts = {}
  local i = 1
  while i <= #argv do
    local a = argv[i]
    if a == "--max-states" then
      opts.max_states = tonumber(argv[i + 1])
      if not opts.max_states then return nil end
      i = i + 2
    elseif a == "--no-invariants" then
      opts.check_invariants = false
      i = i + 1
    elseif a:match("^%-%-") or module then
      return nil
    else
      module = a
      i = i + 1
    end
  end
  if not module then return nil end
  return module, opts
end

local EXIT_CODES = {
  ok = 0, ["invariant-violated"] = 1, ["runtime-error"] = 2, ["bound-exceeded"] = 3,
}

function E.main(argv)
  local module, opts = parse_args(argv)
  if not module then
    io.stderr:write(USAGE .. "\n")
    return 64
  end
  local res = E.explore(dofile(module), opts)

  print("result=" .. res.result)
  print("states=" .. res.states)
  print("transitions=" .. res.transitions)
  print("depth=" .. res.depth)
  if res.result == "invariant-violated" then
    print("invariant=" .. res.invariant)
    for i, step in ipairs(res.trace) do
      if i == 1 then
        print("trace.0=" .. encode(step.state))
      else
        local arg = (step.arg == nil) and "-" or encode(step.arg)
        print("trace." .. (i - 1) .. "=" .. step.action .. "(" .. arg .. ") -> " .. encode(step.state))
      end
    end
  elseif res.result == "runtime-error" then
    print("error=" .. res.error)
  end
  return EXIT_CODES[res.result]
end

-- Run main only when invoked as a script (not when dofile'd by a test).
if arg and arg[0] and arg[0]:match("explore%.lua$") then
  os.exit(E.main(arg))
end

return E
