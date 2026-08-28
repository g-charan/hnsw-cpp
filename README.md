# hnsw-cpp

An approximate nearest-neighbour index in C++23, with no dependencies outside
the standard library.

It builds a Hierarchical Navigable Small World graph over float32 vectors,
serialises it to a binary format that opens with one `mmap` and no parsing, and
answers top-k queries with hand-written NEON distance kernels. Recall is
measured against ground truth the dataset ships, not against itself.

```
vectors ──► HnswIndex ──► binary index file
   .fvecs      build          │
                              ├──► mmap ──► GraphView ──► beam search ──► top-k
                              │                              │
                            no parse                   NEON L2 kernels
```

## Build, index, query

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure

./tools/get_dataset.sh siftsmall          # 10k x 128, ~5 MB
./build/cli/hnsw build data/siftsmall/siftsmall_base.fvecs small.idx
./build/cli/hnsw eval  small.idx \
    data/siftsmall/siftsmall_query.fvecs \
    data/siftsmall/siftsmall_groundtruth.ivecs
```

`./tools/get_dataset.sh sift` fetches the full SIFT1M set (~160 MB) used for the
numbers below.

## Benchmarks

Every figure comes from a benchmark target in this repository, on an Apple M4
Max with Apple clang 17. Re-run them; they will differ on your hardware.

### Distance kernels

`./build/bench/bench_distance`

| dimension | scalar | NEON | speedup |
|---|---|---|---|
| 128 | 31.10 ns/op | 6.08 ns/op | **5.11x** |
| 384 | 119.01 ns/op | 18.92 ns/op | **6.29x** |

More than NEON's four lanes alone would give. The rest comes from carrying four
independent accumulators, which keeps the pipeline fed instead of serialising on
the FMA latency chain, and from clang being unable to auto-vectorise the scalar
reduction on its own: float addition is not associative, so it cannot reorder
the sum without `-ffast-math`, which this project does not enable.

There is no AVX2 path. There is no x86 machine here to test one on, and an
untested SIMD kernel is a correctness risk rather than a feature.

### Search on SIFT1M

1,000,000 x 128 vectors, 10,000 queries, k=10, M=16, ef_construction=200,
scored against the ground truth the dataset ships.
`./build/bench/bench_search data/sift sift`

| ef | recall@10 | QPS | p50 | p99 | vs brute force |
|---|---|---|---|---|---|
| 10 | 0.7265 | 37,645 | 25 us | 45 us | 275x |
| 40 | 0.9384 | 14,685 | 69 us | 95 us | 107x |
| 100 | **0.9866** | **6,817** | 152 us | 196 us | **50x** |
| 200 | 0.9966 | 3,747 | 276 us | 362 us | 27x |
| 400 | 0.9988 | 2,060 | 503 us | 671 us | 15x |

Exhaustive search over the same data manages 137 QPS at 7.2 ms per query. The
index is 626 MiB including vectors, of which the graph is 138 MiB, and builds in
about 250 s at an average degree of 22.5.

### Against hnswlib

`./build/bench/bench_compare data/sift sift`

Comparing QPS between two approximate indexes is meaningless unless recall is
held equal, since any index can be made faster by returning worse answers. Both
are swept across ef and compared at the same recall.

| recall at least | this project | hnswlib 0.8.0 | ratio |
|---|---|---|---|
| 0.90 | 14,540 | 8,436 | **1.72x** |
| 0.95 | 10,310 | 6,127 | **1.68x** |
| 0.99 | 3,671 | 2,191 | **1.68x** |

Build 250 s against 357 s, and 626 MiB against 748 MiB.

**That comparison is real but it is not a claim about the graph.** hnswlib's
SIMD kernels are guarded on `__SSE__` and `__AVX__`, which are x86-only, so on
aarch64 it runs the scalar `L2Sqr` at `space_l2.h:215`. The table above is this
project's NEON kernels against hnswlib's scalar fallback -- which is genuinely
what you get on an Apple machine, and is genuinely not an algorithmic win.

So `bench_compare_scalar` builds this project with `VEC_FORCE_SCALAR=1` and runs
the same sweep, which isolates graph quality from kernel quality:

| recall at least | this project (scalar) | hnswlib 0.8.0 | ratio |
|---|---|---|---|
| 0.90 | 9,929 | 10,306 | 0.96x |
| 0.95 | 7,139 | 7,459 | 0.96x |
| 0.99 | 2,571 | 2,722 | 0.94x |

The honest reading: the graph is at parity with the reference implementation and
a few percent behind it, this project reaches slightly higher recall at every ef
(0.9866 against 0.9829 at ef=100), and the entire advantage on Apple Silicon
comes from having a NEON path that hnswlib does not.

### Index open time

`hnsw info` / `MappedIndex`

| | |
|---|---|
| open a 6.3 MiB index | **0.024 - 0.055 ms** |
| first query after open | 0.19 - 0.31 ms |

Open time is a syscall and a header check, and does not grow with the index.
The first query is more expensive than later ones because it is what actually
faults the pages in; that cost is paging, not parsing.

## Why it is built this way

### Layer 0 is one flat array

Layer 0 holds every node and absorbs essentially all the search traffic, so it
is stored as a single `uint32` array of `count * (max_m0 + 1)` entries: a degree
followed by its neighbour ids, inline. No per-node `vector`, no pointer chase to
reach a neighbour list, and neighbour lists of nearby ids share cache lines.

Upper layers are sparse -- a node reaches level *l* with probability about
`1/M^l` -- so those lists are packed end to end, with each node recording the
offset where its own run starts.

### Offsets, not pointers

Upper-layer lists are addressed by a per-node offset array rather than by
pointers. A pointer would have to be fixed up for every node when an index is
mapped from disk, which is exactly the O(n) load-time work the format exists to
avoid. With offsets, `GraphView` is built by pointing at five places in the
mapping and nothing is decoded.

### One search implementation

`GraphView` is a read-only handle of raw pointers with no ownership. Both a
freshly built index and one mapped off disk expose the same view, so a single
beam search serves both and the two cannot drift apart.

### The visited set is never cleared

Marking visited nodes uses a generation-stamped array: a node counts as visited
when its stamp equals the current generation, and a new query increments the
generation. Clearing a `count`-entry set per query would come to dominate the
query itself. The counter wrapping is handled explicitly, since a wrap would
otherwise alias stale marks as visited.

### Neighbour selection is the paper's heuristic, not top-M

Keeping the M nearest candidates outright clusters every edge in whichever
direction the data is densest, and the graph stops being navigable from the
sparse side. Algorithm 4 keeps a candidate only when it is closer to the query
than to any already-chosen neighbour, which spreads edges across directions.
The same heuristic runs when a node's neighbour list overflows, rather than
simply dropping the furthest.

## Correctness

```bash
ctest --test-dir build --output-on-failure
```

Correctness for an approximate index means recall, so most tests score the graph
against exhaustive search over the same data. That would keep passing if both
shared a misunderstanding of the input, so `test_dataset` scores it against
ground truth SIFT ships, computed by someone else's exhaustive search -- and
checks that this project's own flat index reproduces that ground truth exactly
before trusting any graph number. It skips rather than fails when `data/` has
not been fetched.

The distance tests compare NEON against the scalar reference across dimensions
chosen to exercise the 16-wide body, the 4-wide tail and the scalar remainder,
with a relative epsilon rather than bit equality, because the two sum in
different orders.

## Layout

```
include/core/    arena allocator, mmap wrapper, latency histogram, bench runner
include/vec/     distance kernels, vector store, flat index, HNSW, on-disk format
cli/             build / query / eval / info
bench/           distance, search, and the hnswlib comparison
tests/           one binary per unit, plus the real-dataset check
tools/           dataset fetcher
```

## References

Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search
using Hierarchical Navigable Small World graphs*, 2016.
