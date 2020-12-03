#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <strain.h>

namespace py = pybind11;
using namespace constitutive;

PYBIND11_MODULE(cpp, m)
{
    m.doc() = "constitutive";
    m.def("times_two", &timesTwo, "doc");

    using Mandel2 = Mandel<2>;
    using MandelX2 = MandelX<2>;

    py::class_<Mandel2, Eigen::Vector3d> mandel2(m, "Mandel2");
    mandel2.def(py::init<Eigen::Vector3d>());

    // py::class_<MandelX2, Eigen::Vector4d> mandelX2(m, "MandelX2");
    // mandelX2.def(py::init<Eigen::Vector4d>());
    // mandelX2.def(py::init<Mandel2>());
}
