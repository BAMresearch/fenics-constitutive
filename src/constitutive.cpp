#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <strain.h>

namespace py = pybind11;

PYBIND11_MODULE(cpp, m)
{
    m.doc() = "constitutive";
    m.def("times_two", &constitutive::timesTwo, "doc");
}
