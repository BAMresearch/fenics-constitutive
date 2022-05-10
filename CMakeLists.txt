cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(constitutive VERSION 0.1 LANGUAGES CXX)
enable_language(Fortran)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 3.2 REQUIRED NO_MODULE)

#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -fPIC -L/opt/intel/compilers_and_libraries_2020.2.254/linux/compiler/lib/intel64 -lifcore -limf -lintlc -Wl,--no-as-needed -ldl -lirc")

include_directories(src)
add_subdirectory(src)

#add_library(labtoolsLib STATIC IMPORTED)
#set_target_properties(labtoolsLib PROPERTIES IMPORTED_LOCATION /home/zit.bam.de/vkindrac/labtools/lib/libumat.a)

find_package(pybind11 REQUIRED)
pybind11_add_module(cpp src/constitutive.cpp)
#target_link_libraries(cpp PRIVATE pybind11::module Eigen3::Eigen labtoolsLib)

target_link_libraries(cpp PRIVATE pybind11::module Eigen3::Eigen)

#add_library(umatLib STATIC IMPORTED)
#set_target_properties(umatLib PROPERTIES IMPORTED_LOCATION /home/zit.bam.de/vkindrac/fenics-constitutive/src/umat_linear_elastic.a)
#set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -fPIC -lgfortran")
#target_link_libraries(cpp PRIVATE umatLib)
#target_link_libraries(cpp PRIVATE pybind11::module Eigen3::Eigen umatLib)

# create static library
# gfortran -c -fPIC umat_linear_elastic.f
# ar -r umat_linear_elastic.a umat_linear_elastic.o


#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -fPIC -L/opt/intel/compilers_and_libraries_2020.2.254/linux/compiler/lib/intel64 -lifcore -limf -lintlc -Wl,--no-as-needed -ldl -lirc")
#target_link_libraries(cpp PRIVATE labtoolsLib)
