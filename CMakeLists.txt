cmake_minimum_required(VERSION 2.8)
project(random  LANGUAGES CXX CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(rw starter.cu)
CUDA_curand_LIBRARY(rw -lcurand)
# target_link_libraries(rw cuda_vector_lib -lcublas -lcurand)
# target_link_libraries(rw ${CUDA_curand_LIBRARY})