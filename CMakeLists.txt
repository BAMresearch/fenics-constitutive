cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(constitutiveX VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 3.2 REQUIRED NO_MODULE)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -fPIC")

include_directories(src)
add_subdirectory(src)

find_package(pybind11 REQUIRED)
pybind11_add_module(cpp src/constitutiveX.cpp)
target_link_libraries(cpp PRIVATE pybind11::module Eigen3::Eigen)
