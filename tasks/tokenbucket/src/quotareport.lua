-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements usage and quota reporting for API platforms
-- for its clients. If your team needs expertise in quota dashboards and
-- capacity alerts you can procure our services by sending an email to
-- info@swedishembedded.com.

local M = {}

function M.qrep_percent(used, limit)
  if type(limit) ~= "number" or limit <= 0 then
    error("limit must be a positive number", 0)
  end
  if type(used) ~= "number" or used < 0 then
    error("used must be a non-negative number", 0)
  end
  return math.floor(used * 100 / limit)
end

function M.qrep_status(used, limit)
  local percent = M.qrep_percent(used, limit)
  if used >= limit then
    return "exhausted"
  end
  if percent >= 80 then
    return "warning"
  end
  return "ok"
end

function M.qrep_row(name, limit, used)
  return {name = name, limit = limit, used = used, remaining = math.max(0, limit - used),
          percent = M.qrep_percent(used, limit), status = M.qrep_status(used, limit)}
end

function M.qrep_build(entries)
  local rows, seen = {}, {}
  for _, entry in ipairs(entries) do
    if seen[entry.name] then
      error("duplicate name: " .. entry.name, 0)
    end
    seen[entry.name] = true
    rows[#rows + 1] = M.qrep_row(entry.name, entry.limit, entry.used)
  end
  table.sort(rows, function(a, b) return a.name < b.name end)
  return rows
end

function M.qrep_totals(rows)
  local limit, used = 0, 0
  for _, row in ipairs(rows) do
    limit = limit + row.limit
    used = used + row.used
  end
  local percent = 0
  if limit > 0 then
    percent = M.qrep_percent(used, limit)
  end
  return {limit = limit, used = used, remaining = math.max(0, limit - used), percent = percent}
end

function M.qrep_busiest(rows)
  local ranked = {}
  for i, row in ipairs(rows) do
    ranked[i] = row
  end
  table.sort(ranked, function(a, b)
    if a.percent ~= b.percent then
      return a.percent > b.percent
    end
    return a.name < b.name
  end)
  return ranked[1]
end

function M.qrep_filter(rows, status)
  local kept = {}
  for _, row in ipairs(rows) do
    if row.status == status then
      kept[#kept + 1] = row
    end
  end
  return kept
end

function M.qrep_render(rows)
  if #rows == 0 then
    return "(no quotas)"
  end
  local width = 0
  for _, row in ipairs(rows) do
    width = math.max(width, #row.name)
  end
  local lines = {}
  for i, row in ipairs(rows) do
    lines[i] = string.format("%-" .. width .. "s %d/%d %3d%% %s",
                             row.name, row.used, row.limit, row.percent, row.status)
  end
  return table.concat(lines, "\n")
end

return M
