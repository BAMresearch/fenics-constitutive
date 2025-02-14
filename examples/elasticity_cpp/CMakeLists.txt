cmake_minimum_required(VERSION 3.15...3.26)
project(elasticity_cpp LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

pybind11_add_module(elasticity_cpp MODULE src/main.cpp)
#install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
target_link_libraries(elasticity_cpp PRIVATE pybind11::module Eigen3::Eigen)