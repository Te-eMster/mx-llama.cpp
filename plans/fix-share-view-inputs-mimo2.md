# Fix: GGML_SCHED_SHARE_VIEW_INPUTS corrupts output (MiMo V2.6 Flash)

## Symptom

- `GGML_SCHED_SHARE_VIEW_INPUTS=1` (default): gibberish output, e.g. `295,,0., a。3 for0,58 ...`
- `GGML_SCHED_SHARE_VIEW_INPUTS=0`: correct output, but prompt processing ~50% slower
- Only MiMo V2.6 Flash (arch `mimo2`) on GPU+CPU (HIP gfx906 + CPU offload) is affected; other models pass

## Root cause

The view-sharing scheduler stages a full-size reshape view as a view of its root's
cross-backend copy instead of a second copy. This is only safe if the root copy is
pinned in the graph allocator, because a scheduler-shared input view is an
"uncounted view": it holds no `n_views` lifetime reference on its root
([`ggml-alloc.c:939`](../ggml/src/ggml-alloc.c)).

The pin is applied only when the root copy is created inside the share branch
([`ggml-backend.cpp:1914-1915`](../ggml/src/ggml-backend.cpp)). When the root
already crossed earlier as itself, its copy was created by the plain crossing
path which sets `GGML_TENSOR_FLAG_INPUT/OUTPUT` only when `sched->n_copies > 1`
([`ggml-backend.cpp:1960`](../ggml/src/ggml-backend.cpp)) - and MiMo's decode
runs with `sched copies = 1` (see `sched_reserve` in the logs). The share branch
then adopts this un-pinned copy as the backing of the shared view-copy without
pinning it.

Consequence in ggml-alloc: the root copy's `n_children` counts only its direct
consumers (nodes using the root itself), the uncounted shared view-copy adds
nothing. As soon as the last direct consumer is passed, the root copy is freed
([`ggml-alloc.c:951`](../ggml/src/ggml-alloc.c)) while the shared view-copy is
still consumed by later nodes. The freed slot is reused best-fit
([`ggml-alloc.c:202`](../ggml/src/ggml-alloc.c)) by those nodes' outputs and by
the per-split input fill that rewrites the root copy, so reads and writes alias.

### Trigger in MiMo V2.6 decode graph (confirmed in `sched_split_view_1.log`)

Layer 6, split 3 (ROCm0), single split where root and view cross together:

1. `ffn_moe_probs-6` (root, sigmoid output, computed on ROCm1) crosses 1 -> 0.
   Its first consumer is the probs-bias [`ggml_add()`](../src/llama-graph.cpp:2261)
   (or [`ggml_argsort_top_k()`](../src/llama-graph.cpp:2305)) - the plain crossing
   path creates the root copy, un-pinned.
2. `ffn_moe_probs-6 (reshaped)` = [`ggml_reshape_3d(probs, 1, n_expert, n_tokens)`](../src/llama-graph.cpp:2316)
   crosses in the same split. The share branch adopts the existing un-pinned copy
   (log line: `share view ffn_moe_probs-6 (reshaped) ... backends 1 -> 0` with no
   matching `share register` - the dedup found the root already registered).
3. Its consumer [`ggml_get_rows()`](../src/llama-graph.cpp:2319) reads the shared
   view (the root copy's memory) later in the same split to build
   `ffn_moe_weights-6`.

At allocation time the root copy is freed right after the add/argsort and its slot
is reused by the argsort output (`ffn_moe_topk-6`) or the get_rows output. At
runtime the get_rows then reads expert weights from memory that now holds
topk ids or its own output - corrupted MoE weights poison the residual stream of
every later layer, producing exactly the observed gibberish.

With `GGML_SCHED_SHARE_VIEW_INPUTS=0` each crossing gets its own directly
consumed copy whose lifetime is counted normally, so no aliasing occurs.

Note: the layer-13 pattern (root crosses in split 9, view in split 11) survives
only by accident - the root copy is re-registered as a graph node at split 11 and
therefore re-allocated. The fix below makes both patterns robust.

## Fix (revised: lifetime reference instead of pinning)

The first fix attempt pinned the root copy with `ggml_set_output()` so ggml-alloc
never frees it. Correctness was restored, but prefill performance collapsed to
the `GGML_SCHED_SHARE_VIEW_INPUTS=0` level: every pinned root copy
(e.g. `ffn_norm-27`, 32 MB at prefill, one per layer) lives to graph end, so the
compute buffer cannot recycle the slot across layers (~1 GB extra, see
`sched_reserve`: 3624 MiB with sharing vs 2560 MiB without). The `=0` path frees
each copy after its consumers and reuses one slot for all layers.

The root copy must live exactly until the last consumer of its shared views -
not forever. That is what `n_views` is for: ggml-alloc already frees a view's
`view_src` when the view's own consumers are done and its counted reference
drops ([`ggml-alloc.c`](../ggml/src/ggml-alloc.c) free path). Scheduler-shared
input views were deliberately uncounted, which is only safe when the root is
pinned - the broken combination was "uncounted + unpinned".

Changes:

1. [`ggml/src/ggml-alloc.c`](../ggml/src/ggml-alloc.c) count phase: count an
   `n_views` reference for view srcs that are not graph nodes (the scheduler's
   shared input views), deduplicated with the existing `counted_view` flag.
   Graph-node views keep their existing node-based counting.
2. [`ggml/src/ggml-backend.cpp`](../ggml/src/ggml-backend.cpp) `share_root`
   block: drop the `ggml_set_output()` pin (keep `ggml_set_input()`); the
   reference now provides the lifetime.
3. [`FEATURES.md`](../FEATURES.md): the invariant text updated accordingly.

Effects:

- The root copy is freed right after the last shared-view consumer, so the slot
  recycles across layers - memory profile back to the `=0` level.
- The use-after-free is gone: the reference keeps the copy alive while any
  shared view is consumed, including the MiMo decode layer 6 case where the
  root is also consumed directly earlier in the same split.
- Crossings stay single-copy, so the transfer saving remains.

## Steps

1. Rebuild (same HIP build as used for the logs).
2. Correctness check: run MiMo V2.6 Flash with `GGML_SCHED_SHARE_VIEW_INPUTS=1`,
   greedy (temp 0), same prompt as the logs; compare with the
   `GGML_SCHED_SHARE_VIEW_INPUTS=0` output - must be byte-identical.
3. Perplexity check: `llama-perplexity` on a small text sample with
   `GGML_SCHED_SHARE_VIEW_INPUTS=0` vs `=1` - results must match.
4. Performance check: prefill t/s with `=1` must be back to the fast level
   (no -50%) and `sched_reserve` compute buffer sizes must be near the
   `=0` run (2560 MiB-class), not the pinned 3624 MiB. Compare with a fixed
   config (`-fit off`, explicit `-ts`/`-ngl`): the fit picks different splits
   per mode (log 0: splits 172/75, log 1: 180/83) and small buffer deltas can
   flip its choice, which alone skews benchmarks.
5. Debug check: run with `GGML_SCHED_DEBUG=1` and confirm no
   `fill-order violation` lines and stable `rebind ... needs_realloc=0`.
6. Run `ctest` (test-backend-ops, test-scheduler-*) for regressions.

## Optional hardening (separate, discuss first)

- Debug-only assert in ggml-alloc: when freeing a tensor, warn if it is the
  `view_src` of any registered shared view copy (catches future variants of
  this bug early).
- Keep `GGML_SCHED_SHARE_VIEW_INPUTS=0` as the escape hatch (already exists).
