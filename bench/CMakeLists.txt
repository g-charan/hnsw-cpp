foreach(b bench_distance bench_search)
  add_executable(${b} ${b}.cpp)
  target_link_libraries(${b} PRIVATE hnsw)
endforeach()

# hnswlib is third-party and not warning-clean under this project's flags, so it
# gets its own target with warnings off rather than lowering the bar everywhere.
add_executable(bench_compare bench_compare.cpp)
target_link_libraries(bench_compare PRIVATE hnsw)
target_include_directories(bench_compare SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
target_compile_options(bench_compare PRIVATE -Wno-everything)

# The same comparison with this project's SIMD disabled. hnswlib has no NEON
# path and falls back to scalar on aarch64, so the default comparison measures
# kernel quality; this one isolates graph quality.
add_executable(bench_compare_scalar bench_compare.cpp)
target_link_libraries(bench_compare_scalar PRIVATE hnsw)
target_include_directories(bench_compare_scalar SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
target_compile_options(bench_compare_scalar PRIVATE -Wno-everything -DVEC_FORCE_SCALAR=1)
