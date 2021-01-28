#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <interfaces.h>
#include <laws.h>

namespace py = pybind11;

PYBIND11_MODULE(cpp, m)
{
    // This was created with the help of
    //     https://pybind11.readthedocs.io/en/stable/basics.html
    // and
    //     https://pybind11.readthedocs.io/en/stable/classes.html

    m.doc() = "Super awesome, super fast constitutive lib";

    pybind11::enum_<Constraint>(m, "Constraint")
            .value("UNIAXIAL_STRAIN", Constraint::UNIAXIAL_STRAIN)
            .value("UNIAXIAL_STRESS", Constraint::UNIAXIAL_STRESS)
            .value("PLANE_STRAIN", Constraint::PLANE_STRAIN)
            .value("PLANE_STRESS", Constraint::PLANE_STRESS)
            .value("FULL", Constraint::FULL)
            .value("3D", Constraint::FULL);

    pybind11::class_<IpBase> ipBase(m, "IpBase");
    ipBase.def("evaluate", &IpBase::evaluate);
    ipBase.def("update", &IpBase::update);

    pybind11::class_<LinearElastic, IpBase> linearElastic(m, "LinearElastic");
    linearElastic.def(pybind11::init<double, double, Constraint>());

    pybind11::class_<Base> base(m, "Base");
    base.def(pybind11::init<IpBase&>());
    base.def("evaluate", &Base::evaluate);
    base.def("update", &Base::update);
    base.def_readonly("stress", &Base::_stress);
    base.def_readonly("dstress", &Base::_dstress);
}
