<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# graphkit

A small graph library in Lua 5.1: adjacency storage for directed and
undirected weighted graphs, traversal orders, shortest paths by priority-queue
relaxation over a binary heap, fewest-hop paths, topological ordering with
cycle reporting, strongly connected components, union-find with minimum
spanning trees, bipartite checks and matching, connected components with
bridges, and transitive closure with reachability queries.

The modules live in `src/` and are plain Lua 5.1 (`local M = {} ... return M`).
The tests are Python `unittest` files that start one `lua` process per test
method, print a canonical text rendering of the result and compare it with the
expected text. They need the `lua` interpreter (5.1) on the path and nothing
else.

## Running the tests

    python3 -m unittest

Run it in this directory. A missing `lua` makes the tests fail, never skip.

## The graph value

A graph is a plain table built by `adjacency.gkadj_new`:

| field      | meaning                                                          |
|------------|------------------------------------------------------------------|
| `directed` | boolean                                                          |
| `order`    | vertices in insertion order                                      |
| `out`      | vertex -> list of `{to = v, w = number}` arcs in insertion order |
| `indeg`    | vertex -> number of arcs ending at the vertex                    |
| `edges`    | number of edges (an undirected edge counts once)                 |

An undirected edge is stored as one arc in each direction (a self-loop as one
arc). Every algorithm module reads only `directed`, `order` and `out`, and
visits arcs in stored order, so all results are deterministic. Vertices are
strings or numbers. Errors are raised with `error(message, 0)`, so the message
has no position prefix. Every module that takes a vertex raises
`unknown vertex: <v>` for a vertex that is not in the graph unless noted.

## adjacency

- `gkadj_new(directed)` returns an empty graph; a missing flag means undirected.
- `gkadj_add_vertex(g, v)` adds `v`; true when it is new, false when present.
  Error: `vertex must not be nil`.
- `gkadj_add_edge(g, a, b, w)` adds the edge (weight default 1), creating both
  endpoints. Adding an existing edge replaces its weight and does not count a
  new edge. Errors: `vertex must not be nil` (graph unchanged),
  `weight must be a number`.
- `gkadj_remove_edge(g, a, b)` removes the edge (both arcs when undirected);
  true when something was removed.
- `gkadj_remove_vertex(g, v)` removes `v` with all its arcs and fixes the
  counts. Error: unknown vertex.
- `gkadj_has_vertex(g, v)`, `gkadj_has_edge(g, a, b)` return booleans;
  unknown vertices simply have no edge.
- `gkadj_weight(g, a, b)` returns the weight or nil.
- `gkadj_neighbors(g, v)` returns a fresh list of `{to, w}` entries.
  Error: unknown vertex.
- `gkadj_vertices(g)` returns a copy of the vertex list.
- `gkadj_degree_out(g, v)`, `gkadj_degree_in(g, v)` return arc counts.
  Error: unknown vertex.
- `gkadj_edge_count(g)` returns the number of edges.
- `gkadj_edges(g)` lists each edge once as `{from, to, w}`; undirected edges
  appear from the endpoint inserted first.

## traversal

All functions take a start vertex that must exist (unknown vertex otherwise).

- `gkwalk_breadth_order(g, start)` breadth-first visiting order.
- `gkwalk_depth_preorder(g, start)` depth-first order, vertices listed on entry.
- `gkwalk_depth_postorder(g, start)` depth-first order, vertices listed on exit.
- `gkwalk_breadth_levels(g, start)` list of levels, level 1 being `{start}`.
- `gkwalk_breadth_within(g, start, limit)` vertices at most `limit` steps away,
  nearest first. Error: `depth limit must be a non-negative number`.
- `gkwalk_breadth_order_all(g)`, `gkwalk_depth_preorder_all(g)` cover every
  vertex, starting a new walk at each unvisited vertex in vertex order.

## heap

A binary min-heap ordered by priority; equal priorities leave in arrival order.

- `gkheap_new()` empty heap.
- `gkheap_push(h, value, priority)` returns the new size. Errors:
  `value must not be nil`, `priority must be a number`.
- `gkheap_pop(h)` returns `value, priority` of the minimum.
  Error: `heap is empty`.
- `gkheap_peek(h)` the same without removing. Error: `heap is empty`.
- `gkheap_size(h)`, `gkheap_is_empty(h)`.
- `gkheap_pop_many(h, count)` list of up to `count` values.
  Error: `count must be a non-negative number`.
- `gkheap_drain(h)` all values in order, leaving the heap empty.
- `gkheap_from_pairs(list)` builds a heap from `{value, priority}` pairs.

## shortest

Single-source shortest paths by priority-queue relaxation (uses `heap`).
Weights must be non-negative on every arc that is reached.

- `gkpath_distances(g, source)` map vertex -> cost for reachable vertices.
- `gkpath_settle_order(g, source)` vertices nearest first; ties in queue order.
- `gkpath_parents(g, source)` map vertex -> predecessor (first one on ties).
- `gkpath_cost(g, source, target)` cost or nil when unreachable.
- `gkpath_route(g, source, target)` returns the vertex list and the cost, or nil.
- `gkpath_within(g, source, limit)` settled vertices with cost <= limit.
- `gkpath_nearest(g, source, count)` the first `count` settled vertices after the source.
- `gkpath_farthest(g, source)` last settled vertex and its cost.

Errors: unknown vertex (source, and target where one is taken);
`negative weight on edge a>b`.

## hops

Fewest-step queries; arc weights are ignored.

- `gkhops_distance(g, source, target)` steps or nil.
- `gkhops_all(g, source)` map vertex -> steps.
- `gkhops_path(g, source, target)` vertex list or nil (first-discovered parents).
- `gkhops_farthest(g, source)` first-discovered farthest vertex and its steps.
- `gkhops_ring(g, source, steps)` vertices exactly `steps` away, discovery order.
  Error: `steps must be a non-negative number`.
- `gkhops_reach_count(g, source, steps)` vertices other than the source within `steps`.
- `gkhops_count_paths(g, source, target)` number of distinct fewest-step paths (0 if none).
- `gkhops_diameter(g)` the largest fewest-step distance between joined vertices.

## topo

Needs a directed graph (`topological order needs a directed graph`). A cycle is
a list `v1..vk` with arcs `v1->v2 ... vk->v1`.

- `gktopo_order(g)` an order of all vertices, or `nil, cycle`.
- `gktopo_sort(g)` the order; error `cycle detected: a>b>a` otherwise.
- `gktopo_find_cycle(g)` a cycle or nil; `gktopo_is_acyclic(g)`.
- `gktopo_reverse_finish(g)` another valid order (reverse depth-first finish).
- `gktopo_layers(g)` groups that can run together; `gktopo_chain_length(g)`
  number of layers; `gktopo_rank_map(g)` vertex -> position. All raise the cycle error.
- `gktopo_sources(g)`, `gktopo_sinks(g)` vertices with no incoming or no outgoing arcs.

## strong

Needs a directed graph (`components need a directed graph`). Components are
ordered so that arcs between components point forward; members keep vertex order.

- `gkscc_finish_order(g)` vertices by depth-first finishing time.
- `gkscc_components(g)`, `gkscc_component_map(g)` (vertex -> index),
  `gkscc_component_count(g)`.
- `gkscc_is_strongly_connected(g)` true for one component (false when empty).
- `gkscc_same_component(g, a, b)`. Error: unknown vertex.
- `gkscc_condensation(g)` list of `{from, to}` component links, each once.
- `gkscc_largest(g)` first component with the most members.
- `gkscc_cyclic_vertices(g)` vertices on some cycle (including self-loops).

## unionfind

Disjoint sets with path halving and union by size; on equal sizes the first
argument's root stays root. Unknown elements raise `unknown element: <x>`.

- `gkdsu_new()`, `gkdsu_from_list(list)`, `gkdsu_add(d, x)` (true when new;
  error `element must not be nil`).
- `gkdsu_find(d, x)` the root; `gkdsu_union(d, a, b)` true when two sets merged.
- `gkdsu_connected(d, a, b)`, `gkdsu_size_of(d, x)`, `gkdsu_set_count(d)`,
  `gkdsu_element_count(d)`.
- `gkdsu_depth_of(d, x)` links from `x` to its root, without compressing.
- `gkdsu_groups(d)` sets ordered by first member; `gkdsu_members(d, x)`.

## spantree

Needs an undirected graph (`spanning tree needs an undirected graph`). Edges are
taken lightest first (heaviest first for maximum), ties in storage order.

- `gkmst_forest(g)` returns the chosen `{from, to, w}` edges and their total.
- `gkmst_maximum_forest(g)` the same for a maximum forest.
- `gkmst_tree(g)` like `gkmst_forest` but requires connectivity
  (`graph is not connected: N trees`).
- `gkmst_total_weight(g)`, `gkmst_tree_count(g)`, `gkmst_is_spanning_tree(g)`.
- `gkmst_contains_edge(g, a, b)` whether the minimum forest uses the edge.
- `gkmst_bottleneck(g, a, b)` smallest possible heaviest edge on a path from
  `a` to `b` (0 when equal, nil when disconnected). Unknown vertices raise
  `unknown element: <v>`.

## bipartite

Needs an undirected graph (`bipartite check needs an undirected graph`).

- `gkbip_colouring(g)` map vertex -> 1 or 2, or nil when not bipartite.
- `gkbip_is_bipartite(g)`; `gkbip_conflict(g)` the first same-coloured edge `{a, b}` or nil.
- `gkbip_sides(g)` returns the colour-1 and colour-2 vertex lists, or nil.
- `gkbip_matching(g)` a maximum matching as `{left, right}` pairs ordered by
  right vertex; error `graph is not bipartite`. `gkbip_matching_size(g)`.

## components

Needs an undirected graph (`components need an undirected graph`).

- `gkcomp_components(g)` groups in breadth-first order, by first vertex.
- `gkcomp_count(g)`, `gkcomp_sizes(g)`, `gkcomp_largest(g)`,
  `gkcomp_is_connected(g)` (an empty graph is connected).
- `gkcomp_component_of(g, v)` group index. Error: unknown vertex.
- `gkcomp_isolated(g)` vertices without any arc.
- `gkcomp_bridges(g)` edges whose removal splits a component, as `{a, b}`.

## reach

Reachability uses paths of one or more arcs, so a vertex reaches itself only
through a cycle.

- `gkreach_closure(g)` map vertex -> reachable vertices in vertex order.
- `gkreach_pairs(g)` the closure as `{from, to}` pairs; `gkreach_pair_count(g)`.
- `gkreach_reachable_from(g, v)`; `gkreach_ancestors(g, v)`.
- `gkreach_can_reach(g, from, to)`; `gkreach_mutual(g, a, b)`.
- `gkreach_reduction(g)` arcs not implied by a longer path;
  error `reduction needs an acyclic graph`.

## Example

    package.path = "src/?.lua;" .. package.path
    local adj = require("adjacency")
    local path = require("shortest")
    local g = adj.gkadj_new(true)
    adj.gkadj_add_edge(g, "a", "b", 7)
    adj.gkadj_add_edge(g, "a", "c", 2)
    adj.gkadj_add_edge(g, "c", "b", 3)
    local route, cost = path.gkpath_route(g, "a", "b")
    print(table.concat(route, ">"), cost)   --> a>c>b   5

Swedish Embedded AB implements graph algorithm libraries and routing,
scheduling and dependency analysis for its clients. If your team needs
expertise in graph algorithms and network analysis you can procure our
services by sending an email to info@swedishembedded.com.
