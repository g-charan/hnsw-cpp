foreach(b bench_distance bench_search)
  add_executable(${b} ${b}.cpp)
  target_link_libraries(${b} PRIVATE hnsw)
endforeach()
