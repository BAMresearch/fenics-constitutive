#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "interfaces.h"
#include "linear_elastic.h"

namespace py = pybind11;

void eigen_ref(Eigen::Ref<Eigen::VectorXd> myarray){
        myarray*=2.;
}
PYBIND11_MODULE(cpp, m)
{
    // This was created with the help of
    //     https://pybind11.readthedocs.io/en/stable/basics.html
    // and
    //     https://pybind11.readthedocs.io/en/stable/classes.html

    m.doc() = "Super awesome, super fast constitutive lib";

    /*************************************************************************
     **   ENUMS AND FREE METHODS
     *************************************************************************/
    pybind11::enum_<Constraint>(m, "Constraint")
            .value("UNIAXIAL_STRAIN", Constraint::UNIAXIAL_STRAIN)
            .value("UNIAXIAL_STRESS", Constraint::UNIAXIAL_STRESS)
            .value("PLANE_STRAIN", Constraint::PLANE_STRAIN)
            .value("PLANE_STRESS", Constraint::PLANE_STRESS)
            .value("FULL", Constraint::FULL)
            .value("3D", Constraint::FULL);

    pybind11::enum_<Q>(m, "Q")
            .value("SIGMA", Q::SIGMA)
            .value("DSIGMA_DEPS", Q::DSIGMA_DEPS)
            .value("EEQ", Q::EEQ)
            .value("EPS", Q::EPS)
            .value("E", Q::E)
            .value("KAPPA", Q::KAPPA)
            .value("DEEQ", Q::DEEQ)
            .value("DSIGMA_DE", Q::DSIGMA_DE)
            .value("LAST", Q::LAST);

    m.def("g_dim", &Dim::G);
    m.def("q_dim", &Dim::StressStrain);
    m.def("eigen_mul",&eigen_ref);
    /*************************************************************************
     **   IPLOOP AND MAIN INTERFACES
     *************************************************************************/




    /*************************************************************************
     **   "PURE" MECHANICS LAWS
     *************************************************************************/

    pybind11::class_<LinearElastic<FULL>, std::shared_ptr<LinearElastic<FULL>>> linearElastic(m, "LinearElastic3D");
    linearElastic.def(pybind11::init<double, double, int>(), py::arg("E"), py::arg("nu"), py::arg("number of quadrature points"));
    linearElastic.def("evaluate", &LinearElastic<FULL>::EvaluateAll);
    linearElastic.def("update", &LinearElastic<FULL>::UpdateAll);
    linearElastic.def("inputs", &LinearElastic<FULL>::DefineInputs);




    /*************************************************************************
     **   PLASTICITY
     *************************************************************************/
}
