-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements model-checked verification of concurrent
-- protocols for its clients. If your team needs expertise in specifying
-- protocols in TLA+ and checking implementations against them, you can
-- procure our services by sending an email to info@swedishembedded.com.

-- tracegen.lua: turns the behaviours of a Lua state-machine module into a TLA+
-- module that makes TLC check each of them against a specification. Every
-- edge of the explored state graph ends one trace: the first-discovery path
-- to the edge's source, then its target. TSpec follows a chosen trace step by
-- step through the specification's own Next, so TLC reaches a deadlock exactly
-- when the implementation takes a step the specification does not allow.
--
-- The file is both a library (returns table T) and a command line program:
--   lua5.1 tracegen.lua <module.lua> <SpecName> <out-dir> [--no-invariants]

local T = {}

local here = debug.getinfo(1, "S").source:match("^@(.*)/[^/]*$") or "."
local E = dofile(here .. "/explore.lua")

---------------------------------------------------------------------------
-- T.tla_value: TLA+ literal of plain data
---------------------------------------------------------------------------

local function sortedkeys(t)
  local ks = {}
  for k in pairs(t) do ks[#ks + 1] = k end
  table.sort(ks)
  return ks
end

local function literal(v)
  local tv = type(v)
  if tv == "number" then
    local text = string.format("%d", v)
    return v < 0 and "(" .. text .. ")" or text
  elseif tv == "boolean" then
    return v and "TRUE" or "FALSE"
  elseif tv == "string" then
    return '"' .. v:gsub("[\\\"]", "\\%0") .. '"'
  end
  local parts = {}
  if #v > 0 or next(v) == nil then
    for i = 1, #v do parts[i] = literal(v[i]) end
    return "<<" .. table.concat(parts, ", ") .. ">>"
  end
  for _, k in ipairs(sortedkeys(v)) do
    parts[#parts + 1] = k .. " |-> " .. literal(v[k])
  end
  return "[" .. table.concat(parts, ", ") .. "]"
end

-- Accepts exactly what E.encode accepts.
function T.tla_value(v)
  E.encode(v)
  return literal(v)
end

---------------------------------------------------------------------------
-- T.generate: the trace module
---------------------------------------------------------------------------

-- The first-discovery path from node 1 to `index` as node indices.
local function path_to(index, first_edge_into)
  local path = { index }
  while first_edge_into[index] do
    index = first_edge_into[index].from
    table.insert(path, 1, index)
  end
  return path
end

local function path_less(a, b)
  if #a ~= #b then return #a < #b end
  for i = 1, #a do
    if a[i] ~= b[i] then return a[i] < b[i] end
  end
  return false
end

-- Deduplicated, sorted traces of the explored graph.
local function traces_of(explored)
  local first_edge_into = {}
  for _, edge in ipairs(explored.edges) do
    if not first_edge_into[edge.to] then first_edge_into[edge.to] = edge end
  end
  local seen, traces = {}, {}
  local function add(trace)
    local key = table.concat(trace, ",")
    if not seen[key] then
      seen[key] = true
      traces[#traces + 1] = trace
    end
  end
  for _, edge in ipairs(explored.edges) do
    local trace = path_to(edge.from, first_edge_into)
    trace[#trace + 1] = edge.to
    add(trace)
  end
  if #traces == 0 then add({ 1 }) end  -- a module that never moves
  table.sort(traces, path_less)
  return traces
end

function T.generate(M, explored, spec_name)
  if explored.result ~= "ok" then
    error("cannot generate traces from an exploration that is " .. explored.result)
  end
  local nodes = explored.nodes
  local variables = sortedkeys(nodes[1])

  local lines = {
    "---- MODULE " .. spec_name .. "Trace ----",
    "EXTENDS " .. spec_name .. ", Naturals, Sequences",
    "VARIABLES pos, tr",
    "tvars == <<vars, pos, tr>>",
    "Nodes == <<",
  }
  for i, state in ipairs(nodes) do
    lines[#lines + 1] = "  " .. T.tla_value(state) .. (i < #nodes and "," or "")
  end
  lines[#lines + 1] = ">>"
  lines[#lines + 1] = "Traces == {"
  local traces = traces_of(explored)
  for i, trace in ipairs(traces) do
    local items = {}
    for j, node in ipairs(trace) do items[j] = tostring(node) end
    lines[#lines + 1] = "  <<" .. table.concat(items, ", ") .. ">>" .. (i < #traces and "," or "")
  end
  lines[#lines + 1] = "}"
  for _, def in ipairs({ { "Match(i)", "" }, { "MatchNext(i)", "'" } }) do
    for k, name in ipairs(variables) do
      local conjunct = "/\\ " .. name .. def[2] .. " = Nodes[i]." .. name
      lines[#lines + 1] = (k == 1) and (def[1] .. " == " .. conjunct) or ("  " .. conjunct)
    end
  end
  for _, line in ipairs({
    "TInit == /\\ Init /\\ tr \\in Traces /\\ pos = 1 /\\ Match(tr[1])",
    "TStep == /\\ pos < Len(tr) /\\ Next /\\ MatchNext(tr[pos + 1]) /\\ pos' = pos + 1 /\\ UNCHANGED tr",
    "TDone == /\\ pos = Len(tr) /\\ UNCHANGED tvars",
    "TNext == TStep \\/ TDone",
    "TSpec == TInit /\\ [][TNext]_tvars",
    "====",
  }) do
    lines[#lines + 1] = line
  end
  return table.concat(lines, "\n") .. "\n"
end

---------------------------------------------------------------------------
-- T.main: command line entry point
---------------------------------------------------------------------------

local USAGE = "usage: lua5.1 tracegen.lua <module.lua> <SpecName> <out-dir> [--no-invariants]"

local EXIT_CODES = {
  ["invariant-violated"] = 1, ["runtime-error"] = 2, ["bound-exceeded"] = 3,
}

function T.main(argv)
  local positional, check_invariants = {}, true
  for _, a in ipairs(argv) do
    if a == "--no-invariants" then
      check_invariants = false
    elseif a:match("^%-%-") then
      io.stderr:write(USAGE .. "\n")
      return 64
    else
      positional[#positional + 1] = a
    end
  end
  if #positional ~= 3 then
    io.stderr:write(USAGE .. "\n")
    return 64
  end
  local module, spec_name, out_dir = positional[1], positional[2], positional[3]
  local M = dofile(module)
  local explored = E.explore(M, { collect_edges = true, check_invariants = check_invariants })
  if explored.result ~= "ok" then
    print("result=" .. explored.result)
    if explored.invariant then print("invariant=" .. explored.invariant) end
    if explored.error then print("error=" .. explored.error) end
    return EXIT_CODES[explored.result]
  end
  local text = T.generate(M, explored, spec_name)
  local file = assert(io.open(out_dir .. "/" .. spec_name .. "Trace.tla", "w"))
  file:write(text)
  file:close()
  print("result=ok")
  print("states=" .. explored.states)
  print("transitions=" .. explored.transitions)
  print("traces=" .. select(2, text:gsub("\n  <<", "")))
  return 0
end

-- Run main only when invoked as a script (not when dofile'd by a test).
if arg and arg[0] and arg[0]:match("tracegen%.lua$") then
  os.exit(T.main(arg))
end

return T
