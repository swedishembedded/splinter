# 042. Mutation-seeded tasks read only `python3 -m unittest`, so other languages go through it

`WorkspaceGenerator::from_mutations` is refused as a whole unless the
profile's test command is `python3 -m unittest` (the only runner read per
test); another command can qualify a baseline by exit status and nothing
more. A family in Lua, shell or any other language therefore yields no
fault-seeded task unless its tests are Python unittest cases. The three Lua
families here run one `lua` process per test method from a unittest file and
assert on the printed text. The fault operators are textual and
language-agnostic, so they apply to the Lua sources unchanged.

Consequence read from the operators: a Lua comment line is not skipped by the
line-emptying operator (only `#` and `//` are), so each Lua comment line is an
equivalent candidate. The funnel of the Lua families was not complete when
this was written, so their refusals by reason are unmeasured.

How to avoid it: until another runner is read per test, give a non-Python
project a unittest driver, and keep Lua comments few.
