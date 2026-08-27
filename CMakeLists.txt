cmake_minimum_required(VERSION 3.20)
project(hnsw_cpp CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -g is kept in release builds: the profiles and perf numbers in the README are
# only worth anything if the binary they came from can be symbolised.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

add_library(hnsw INTERFACE)
target_include_directories(hnsw INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(hnsw INTERFACE -Wall -Wextra -Wpedantic)

enable_testing()
add_subdirectory(tests)
add_subdirectory(bench)
