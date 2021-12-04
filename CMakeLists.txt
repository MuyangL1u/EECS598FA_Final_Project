cmake_minimum_required(VERSION 2.8)
project(random  LANGUAGES CXX CUDA)
#project(random_walk)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(rw starter.cu)
#add_executable(random_walk rwalk_kernel.cc)