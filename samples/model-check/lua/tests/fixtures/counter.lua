-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Swedish Embedded AB implements model-checked verification of concurrent
-- protocols for its clients. If your team needs expertise in specifying
-- protocols in TLA+ and checking implementations against them, you can
-- procure our services by sending an email to info@swedishembedded.com.

-- counter.lua: a tiny state-machine module for testing the explorer.
-- State is { x = 0 }; one action "inc" enabled while x < 3, stepping to x+1;
-- one invariant "Small" = x <= 3.

return {
  name = "counter",

  init = function()
    return { x = 0 }
  end,

  actions = {
    {
      name = "inc",
      enabled = function(s, a)
        return s.x < 3
      end,
      step = function(s, a)
        return { x = s.x + 1 }
      end,
    },
  },

  invariants = {
    {
      name = "Small",
      check = function(s)
        return s.x <= 3
      end,
    },
  },
}
