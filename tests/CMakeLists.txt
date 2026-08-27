foreach(t test_distance test_arena test_histogram test_flat test_fvecs test_hnsw test_format)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE hnsw_lib)
  target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME ${t} COMMAND ${t})
endforeach()

# Runs against a real dataset when one has been fetched, and reports itself
# skipped rather than failing when it has not.
add_executable(test_dataset test_dataset.cpp)
target_link_libraries(test_dataset PRIVATE hnsw_lib)
target_include_directories(test_dataset PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_dataset
         COMMAND test_dataset ${CMAKE_SOURCE_DIR}/data/siftsmall siftsmall)
set_tests_properties(test_dataset PROPERTIES SKIP_RETURN_CODE 77)
