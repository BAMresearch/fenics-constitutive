#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "interfaces.h"
#include "linear_elastic.h"
#include "stress_rates.h"
#include "hypoelasticity.h"
#include "jh.h"

namespace py = pybind11;

void eigen_ref(const Eigen::Ref<const Eigen::VectorXd> &myarray, Eigen::Ref<Eigen::VectorXd> myarray2){
        auto seg = myarray.segment<4>(0);
        auto mat = Eigen::Map<const Eigen::Matrix2d>(seg.data()); 
        myarray2 *= 42.;
        std::cout << mat<< "\n";
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
            .value("GRAD_V", Q::GRAD_V)
            .value("RHO", Q::RHO)
            .value("DAMAGE", Q::DAMAGE)
            .value("LAST", Q::LAST);

    m.def("g_dim", &Dim::G);
    m.def("q_dim", &Dim::StressStrain);
    m.def("eigen_mul",&eigen_ref);
    /*************************************************************************
     **   IPLOOP AND MAIN INTERFACES
     *************************************************************************/


    /*************************************************************************
     **   Stress Rates
     *************************************************************************/
    m.def("jaumann_rotate_3d", &JaumannRotate<FULL>);
    m.def("jaumann_rotate_fast_3d", &JaumannRotateFast<FULL>);
    m.def("tensor_to_mandel_3d", &TensorToMandel<FULL>);
    m.def("mandel_to_tensor_3d", &MandelToTensor<FULL>);

    /*************************************************************************
     **   "PURE" MECHANICS LAWS
     *************************************************************************/
    pybind11::class_<RefLawInterface, std::shared_ptr<RefLawInterface>> law(m, "RefLawInterface");

    pybind11::class_<LinearElastic<FULL>, std::shared_ptr<LinearElastic<FULL>>> linearElastic(m, "LinearElastic3D");
    linearElastic.def(pybind11::init<double, double, int>(), py::arg("E"), py::arg("nu"), py::arg("number of quadrature points"));
    linearElastic.def("evaluate", &LinearElastic<FULL>::EvaluateAll);
    linearElastic.def("update", &LinearElastic<FULL>::UpdateAll);
    linearElastic.def("inputs", &LinearElastic<FULL>::DefineInputs);


    /*************************************************************************
     **   Dynamics Laws
     *************************************************************************/

    pybind11::class_<HypoElastic<FULL>, std::shared_ptr<HypoElastic<FULL>>> hypo_elastic(m, "HypoElastic3D");
    hypo_elastic.def(pybind11::init<double, double, int>(), py::arg("E"), py::arg("nu"), py::arg("number of quadrature points"));
    hypo_elastic.def("evaluate", &HypoElastic<FULL>::EvaluateAll);
    hypo_elastic.def("update", &HypoElastic<FULL>::UpdateAll);
    hypo_elastic.def("inputs", &HypoElastic<FULL>::DefineInputs);
        
    pybind11::class_<JH2Parameters, std::shared_ptr<JH2Parameters>> jh2_parameters(m, "JH2Parameters");
    jh2_parameters.def(pybind11::init<>());
    jh2_parameters.def_readwrite("RHO", &JH2Parameters::RHO);
    jh2_parameters.def_readwrite("SHEAR_MODULUS", &JH2Parameters::SHEAR_MODULUS);
    jh2_parameters.def_readwrite("A", &JH2Parameters::A);
    jh2_parameters.def_readwrite("B", &JH2Parameters::B);
    jh2_parameters.def_readwrite("C", &JH2Parameters::C);
    jh2_parameters.def_readwrite("M", &JH2Parameters::M);
    jh2_parameters.def_readwrite("N", &JH2Parameters::N);
    jh2_parameters.def_readwrite("EPS0", &JH2Parameters::EPS0);
    jh2_parameters.def_readwrite("T", &JH2Parameters::T);
    jh2_parameters.def_readwrite("SIGMAHEL", &JH2Parameters::SIGMAHEL);
    jh2_parameters.def_readwrite("PHEL", &JH2Parameters::PHEL);
    jh2_parameters.def_readwrite("D1", &JH2Parameters::D1);
    jh2_parameters.def_readwrite("D2", &JH2Parameters::D2);
    jh2_parameters.def_readwrite("K1", &JH2Parameters::K1);
    jh2_parameters.def_readwrite("K2", &JH2Parameters::K2);
    jh2_parameters.def_readwrite("K3", &JH2Parameters::K3);
    jh2_parameters.def_readwrite("BETA", &JH2Parameters::BETA);
    jh2_parameters.def_readwrite("MOGEL", &JH2Parameters::MOGEL);
    jh2_parameters.def_readwrite("EFMIN", &JH2Parameters::EFMIN);

    pybind11::class_<JH2<FULL>, std::shared_ptr<JH2<FULL>>> jh2(m, "JH23D");
    jh2.def(pybind11::init<std::shared_ptr<JH2Parameters>, int>(), py::arg("JH2Parameters"), py::arg("number of quadrature points"));
    jh2.def("get_internal_var", &JH2<FULL>::GetInternalVar);
    jh2.def("evaluate", &JH2<FULL>::EvaluateAll);
    jh2.def("update", &JH2<FULL>::UpdateAll);
    jh2.def("inputs", &JH2<FULL>::DefineInputs);

    /*************************************************************************
     **   PLASTICITY
     *************************************************************************/
}
