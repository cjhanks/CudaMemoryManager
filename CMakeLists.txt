# vim: ts=2 sw=2 et ai tw=80
# ============================================================================ #
# Don't believe what you hear, the world is a wonderful place.
# ============================================================================ #

cmake_minimum_required(VERSION 3.16)
project(cmm LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------- #
# All C++ compilation options for all programs.
# ---------------------------------------------------------------------------- #

string(CONCAT CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++14")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  string(CONCAT CMAKE_CXX_FLAGS
         "${CMAKE_CXX_FLAGS} "
         "-O3 "
         "-Werror "
         "-march=native "
         "-mtune=native ")
  add_definitions(-DNDEBUG)
else()
  string(CONCAT CMAKE_CXX_FLAGS
         "${CMAKE_CXX_FLAGS} "
         "-g "
         "-O0 "
         )
endif()

# ---------------------------------------------------------------------------- #
# Build in this order:
#   - Library
#   - Application
#   - Tests
# ---------------------------------------------------------------------------- #

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_subdirectory(lib)
add_subdirectory(app)
add_subdirectory(test)
