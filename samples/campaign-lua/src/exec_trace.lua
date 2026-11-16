-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- The execution tracer of exec_trace.rs (Lua 5.1). The program arrives on
-- standard input; STYLE ("raw" or "steps"), BUDGET (VM instructions) and
-- REPEATS (times a line is shown) are set by the caller. The report is
-- "TRACE <bytes>\n" and the trace, "OUTPUT <bytes>\n" and what the program
-- printed, then "END ok" or "END <reason>".

local source = io.read("*a")
local lines = {}
for l in (source .. "\n"):gmatch("([^\n]*)\n") do lines[#lines + 1] = l end

local out, outbuf = {}, {}
local function emit(s) out[#out + 1] = s end

-- Values ----------------------------------------------------------------------

local function quote(s)
  local cut = #s > 40
  if cut then s = s:sub(1, 40) end
  s = s:gsub("\\", "\\\\"):gsub("\"", "\\\""):gsub("\n", "\\n"):gsub("\t", "\\t")
  return "\"" .. s .. (cut and "..." or "") .. "\""
end

local render
local function key_text(k)
  if type(k) == "string" and k:match("^[%a_][%w_]*$") then return k end
  return "[" .. render(k, 1) .. "]"
end

render = function(v, depth)
  local t = type(v)
  if t == "string" then return quote(v) end
  if t == "number" or t == "boolean" or t == "nil" then return tostring(v) end
  if t ~= "table" then return t end
  if depth >= 1 then return "{...}" end
  local n, count = #v, 0
  for _ in pairs(v) do count = count + 1 end
  local parts = {}
  if count == n then
    for i = 1, math.min(n, 8) do parts[#parts + 1] = render(v[i], depth + 1) end
    if n > 8 then parts[#parts + 1] = "...(" .. n .. ")" end
  else
    local keys = {}
    for k in pairs(v) do keys[#keys + 1] = k end
    table.sort(keys, function(a, b)
      local ta, tb = type(a), type(b)
      if ta ~= tb then return ta < tb end
      if ta == "number" or ta == "string" then return a < b end
      return false
    end)
    for i = 1, math.min(#keys, 6) do parts[#parts + 1] = key_text(keys[i]) .. "=" .. render(v[keys[i]], depth + 1) end
    if #keys > 6 then parts[#parts + 1] = "...(" .. #keys .. " keys)" end
  end
  return "{" .. table.concat(parts, ",") .. "}"
end

-- Events ----------------------------------------------------------------------

local steps = STYLE == "steps"
local frames = {}          -- per program-call depth: { func=, name=, seen={name=text} }
local shown, hidden = {}, {}  -- times each line was shown / counted
local pending, pending_at = {}, {}  -- changes since the last shown line, one entry per name
local hidden_lines = {}

local function indent(depth) return string.rep("  ", depth - 1) end

local function flush_hidden(depth)
  if #hidden_lines == 0 then return end
  table.sort(hidden_lines)
  local parts = {}
  for _, l in ipairs(hidden_lines) do parts[#parts + 1] = "[" .. l .. "] x" .. hidden[l] .. " more" end
  if steps then
    local said = {}
    for i, l in ipairs(hidden_lines) do said[i] = "line " .. l .. " " .. hidden[l] .. " more times" end
    emit(indent(depth) .. "The loop goes on: " .. table.concat(said, ", ") .. ".")
  else
    emit(indent(depth) .. "    (... " .. table.concat(parts, ", ") .. ")")
  end
  for _, l in ipairs(hidden_lines) do hidden[l] = 0 end
  hidden_lines = {}
end

local function flush_changes(depth)
  if #pending == 0 then return end
  local parts = {}
  for i, change in ipairs(pending) do
    local name = change.name
    if steps and name:sub(1, 1) == "^" then name = "outer " .. name:sub(2) end
    parts[i] = steps and (name .. " is " .. change.text) or (name .. "=" .. change.text)
  end
  if steps and #out == 0 then
    emit("At the start " .. table.concat(parts, ", ") .. ".")
  elseif steps then
    out[#out] = out[#out] .. " Now " .. table.concat(parts, ", ") .. "."
  else
    emit(indent(depth) .. "    " .. table.concat(parts, " "))
  end
  pending, pending_at = {}, {}
end

-- Stack levels as seen from the functions below, which the hook calls through
-- on_line: 1 is the function itself, then on_line, then the hook, then the
-- program function that is running.
local RUNNING_FROM_HELPER = 4
local RUNNING_FROM_ON_LINE = 3

-- The program frames on the stack, innermost first.
local function program_frames()
  local list, level = {}, RUNNING_FROM_HELPER
  while true do
    local info = debug.getinfo(level, "Sf")
    if not info then break end
    if info.source == "=prog" then list[#list + 1] = info.func end
    level = level + 1
  end
  return list
end

local function note(name, text)
  local at = pending_at[name]
  if at then
    pending[at].text = text
  else
    pending[#pending + 1] = { name = name, text = text }
    pending_at[name] = #pending
  end
end

local function collect(frame)
  local i = 1
  while true do
    local name, value = debug.getlocal(RUNNING_FROM_HELPER, i)
    if not name then break end
    if name:sub(1, 1) ~= "(" then
      local text = render(value, 0)
      if frame.seen[name] ~= text then
        frame.seen[name] = text
        note(name, text)
      end
    end
    i = i + 1
  end
  local u = 1
  while frame.func do
    local name, value = debug.getupvalue(frame.func, u)
    if not name then break end
    local text = render(value, 0)
    if frame.seen["^" .. name] ~= text then
      frame.seen["^" .. name] = text
      note("^" .. name, text)
    end
    u = u + 1
  end
end

local depth_now = 0

-- Emits the returns from program functions deeper than `depth`.
local function return_to(depth)
  while depth_now > depth do
    flush_changes(depth_now)
    emit(indent(depth_now - 1) .. (steps and ("It returns from " .. frames[depth_now].name .. ".") or ("return from " .. frames[depth_now].name)))
    frames[depth_now] = nil
    depth_now = depth_now - 1
  end
end

local function on_line(line)
  local stack = program_frames()
  local depth = #stack
  return_to(depth)
  -- A frame at this depth running another function: the earlier call returned
  -- and a new one began between two line events.
  if depth == depth_now and depth > 1 and frames[depth].func ~= stack[1] then
    flush_changes(depth)
    emit(indent(depth - 1) .. (steps and ("It returns from " .. frames[depth].name .. ".") or ("return from " .. frames[depth].name)))
    depth_now = depth - 1
    frames[depth] = nil
  end
  for d = depth_now + 1, depth do
    local name = "function"
    if d == depth then
      local info = debug.getinfo(RUNNING_FROM_ON_LINE, "n")
      name = info and info.name or name
    end
    frames[d] = { func = stack[depth - d + 1], name = name, seen = {} }
    if d > 1 then
      flush_changes(d - 1)
      emit(indent(d - 1) .. (steps and ("It calls " .. name .. ".") or ("call " .. name)))
    end
  end
  depth_now = depth
  local frame = frames[depth]
  collect(frame)
  shown[line] = (shown[line] or 0) + 1
  if shown[line] > REPEATS then
    if (hidden[line] or 0) == 0 then hidden_lines[#hidden_lines + 1] = line end
    hidden[line] = (hidden[line] or 0) + 1
    return
  end
  if #hidden_lines > 0 then
    flush_hidden(depth)
  end
  flush_changes(depth)
  local text = (lines[line] or ""):gsub("%s+$", "")
  if steps then
    emit(indent(depth) .. "Line " .. line .. ": " .. text:gsub("^%s+", "") .. ".")
  else
    emit(indent(depth) .. "[" .. line .. "] " .. text)
  end
end

-- Running ----------------------------------------------------------------------

local env = setmetatable({}, { __index = _G })
-- Everything the program writes goes through here: the output it reports, and
-- one `out:` event per completed line (a partial line waits for its end).
local partial = ""
local function write_out(text)
  outbuf[#outbuf + 1] = text
  -- A program function that returned since the last line event has returned now.
  local frames_now, level = 0, 3
  while true do
    local info = debug.getinfo(level, "S")
    if not info then break end
    if info.source == "=prog" then frames_now = frames_now + 1 end
    level = level + 1
  end
  return_to(frames_now)
  local depth = math.max(depth_now, 1)
  flush_changes(depth)
  partial = partial .. text
  while true do
    local nl = partial:find("\n", 1, true)
    if not nl then break end
    local piece = partial:sub(1, nl - 1)
    partial = partial:sub(nl + 1)
    if steps then
      out[#out] = out[#out] .. " It prints " .. quote(piece) .. "."
    else
      emit(indent(depth) .. "    out: " .. piece)
    end
  end
end

local function flush_partial()
  if partial == "" then return end
  local depth = math.max(depth_now, 1)
  if steps then
    out[#out] = out[#out] .. " It prints " .. quote(partial) .. "."
  else
    emit(indent(depth) .. "    out: " .. partial)
  end
  partial = ""
end

local function writes(...)
  local parts = {}
  for i = 1, select("#", ...) do
    local v = select(i, ...)
    parts[i] = type(v) == "number" and string.format("%.14g", v) or tostring(v)
  end
  return table.concat(parts)
end

env.print = function(...)
  local parts = {}
  for i = 1, select("#", ...) do parts[i] = tostring((select(i, ...))) end
  write_out(table.concat(parts, "\t") .. "\n")
end
local stdout = setmetatable({}, { __index = io.stdout })
function stdout:write(...) write_out(writes(...)) return self end
env.io = setmetatable({ stdout = stdout, write = function(...) write_out(writes(...)) return stdout end }, { __index = io })

local chunk, err = loadstring(source, "=prog")
local status
if not chunk then
  status = "syntax: " .. tostring(err)
else
  setfenv(chunk, env)
  local instructions = 0
  debug.sethook(function(event, line)
    if event == "count" then
      instructions = instructions + 1000
      if instructions > BUDGET then error("instruction budget exhausted", 0) end
      return
    end
    local info = debug.getinfo(2, "S")
    if info and info.source == "=prog" then on_line(line) end
  end, "l", 1000)
  local ok, e = pcall(chunk)
  debug.sethook()
  if ok then
    flush_partial()
    flush_hidden(math.max(depth_now, 1))
    flush_changes(math.max(depth_now, 1))
    status = "ok"
  else
    status = "error: " .. tostring(e):gsub("\n", " ")
  end
end

local trace = table.concat(out, "\n") .. (#out > 0 and "\n" or "")
local printed = table.concat(outbuf)
io.write("TRACE ", #trace, "\n", trace, "OUTPUT ", #printed, "\n", printed, "END ", status, "\n")
