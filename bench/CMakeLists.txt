foreach(b bench_distance)
  add_executable(${b} ${b}.cpp)
  target_link_libraries(${b} PRIVATE hnsw)
endforeach()
