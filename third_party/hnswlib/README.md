# Vendored: hnswlib

Upstream: https://github.com/nmslib/hnswlib
Version:  v0.8.0
License:  Apache-2.0, see `LICENSE` in this directory
Authors:  Yury Malkov and the hnswlib contributors

Vendored rather than fetched so the comparison benchmark builds without a
network. These headers are **not** used by anything this project ships: only
`bench/bench_compare.cpp` and `bench/bench_compare_scalar.cpp` include them, and
they exist to measure this implementation against an established one.

Unmodified from upstream apart from `hnswlib.h`, which carries an upstream note
referencing https://github.com/nmslib/hnswlib/pull/508.
