cmake_minimum_required(VERSION 2.8.3)
project(multirotor-drag-sysid)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -fopenmp -mfma")

#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIRS})

#find_package(Boost REQUIRED)
#include_directories(${Boost_INCLUDE_DIRS})

include_directories(lib/logging-utils-lib/include)
add_subdirectory(lib/logging-utils-lib)
include_directories(lib/progress_bar/include)
add_subdirectory(lib/progress_bar)
add_subdirectory(lib/osqp-eigen)

add_executable(multirotor_mu_least_squares
  src/main.cpp
)
target_link_libraries(multirotor_mu_least_squares
  OsqpEigen
  yaml-cpp
  logging-utils-lib
  progress_bar
)
