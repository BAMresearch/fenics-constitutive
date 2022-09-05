cmake_minimum_required(VERSION 3.18..3.20)

project(cofe)

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED CONFIG)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)

pybind11_add_module(cofe main.cpp)
target_link_libraries(cofe PUBLIC pybind11::module xtensor-python Python::NumPy)

target_compile_definitions(cofe PRIVATE VERSION_INFO=0.1.0)
