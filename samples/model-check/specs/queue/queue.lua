-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements model-checked verification of concurrent
-- protocols for its clients. If your team needs expertise in specifying
-- protocols in TLA+ and checking implementations against them, you can
-- procure our services by sending an email to info@swedishembedded.com.

-- Parameters
local CAP = 2        -- queue capacity
local PRODUCERS = 2  -- number of producers
local MAX_ITEMS = 2  -- items each producer may ever send

local function mkcounters()
  local t = {}
  for i = 1, PRODUCERS do t[i] = 0 end
  return t
end

local M = {}
M.name = "queue"

function M.init()
  return { q = {}, sent = mkcounters(), recv = mkcounters() }
end

M.actions = {
  {
    name = "enqueue",
    args = function()
      local t = {}
      for i = 1, PRODUCERS do t[i] = i end
      return t
    end,
    enabled = function(s, p)
      return #s.q < CAP and s.sent[p] < MAX_ITEMS
    end,
    step = function(s, p)
      local q = {}
      for i = 1, #s.q do q[i] = s.q[i] end
      q[#q + 1] = { p = p, n = s.sent[p] + 1 }
      local sent = {}
      for i = 1, PRODUCERS do sent[i] = s.sent[i] end
      sent[p] = sent[p] + 1
      local recv = {}
      for i = 1, PRODUCERS do recv[i] = s.recv[i] end
      return { q = q, sent = sent, recv = recv }
    end,
  },
  {
    name = "dequeue",
    enabled = function(s, _)
      return #s.q > 0
    end,
    step = function(s, _)
      local h = s.q[1]
      local q = {}
      for i = 2, #s.q do q[#q + 1] = s.q[i] end
      local sent = {}
      for i = 1, PRODUCERS do sent[i] = s.sent[i] end
      local recv = {}
      for i = 1, PRODUCERS do recv[i] = s.recv[i] end
      recv[h.p] = recv[h.p] + 1
      return { q = q, sent = sent, recv = recv }
    end,
  },
}

-- Ordered so that the most specific property is reported first: the explorer
-- names the first invariant a state violates, and TypeOK is implied by several.
M.invariants = {
  {
    name = "Bounded",
    check = function(s)
      return #s.q <= CAP
    end,
  },
  {
    name = "Conservation",
    check = function(s)
      local inq = {}
      for p = 1, PRODUCERS do inq[p] = 0 end
      for i = 1, #s.q do inq[s.q[i].p] = inq[s.q[i].p] + 1 end
      for p = 1, PRODUCERS do
        if s.sent[p] - s.recv[p] ~= inq[p] then return false end
      end
      return true
    end,
  },
  {
    name = "PerProducerFifo",
    check = function(s)
      local nextn = {}
      for p = 1, PRODUCERS do nextn[p] = s.recv[p] + 1 end
      for i = 1, #s.q do
        local it = s.q[i]
        if it.n ~= nextn[it.p] then return false end
        nextn[it.p] = nextn[it.p] + 1
      end
      return true
    end,
  },
  {
    name = "TypeOK",
    check = function(s)
      if #s.q > CAP then return false end
      for i = 1, #s.q do
        local it = s.q[i]
        if type(it) ~= "table" or type(it.p) ~= "number" or type(it.n) ~= "number"
          or it.p ~= math.floor(it.p) or it.n ~= math.floor(it.n)
          or it.p < 1 or it.p > PRODUCERS or it.n < 1 then
          return false
        end
      end
      for _, c in ipairs({ s.sent, s.recv }) do
        if #c ~= PRODUCERS then return false end
        for i = 1, PRODUCERS do
          local v = c[i]
          if type(v) ~= "number" or v ~= math.floor(v) or v < 0 or v > MAX_ITEMS then
            return false
          end
        end
      end
      for p = 1, PRODUCERS do
        if s.recv[p] > s.sent[p] then return false end
      end
      return true
    end,
  },
}

return M
