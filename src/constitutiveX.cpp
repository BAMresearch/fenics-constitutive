#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "interfaces.h"
#include "linear_elastic.h"
#include "stress_rates.h"
#include "hypoelasticity.h"
#include "jh.h"

namespace py = pybind11;
//PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string, Eigen::Ref<Eigen::VectorXd>>);
void eigen_ref(std::vector<Eigen::Ref<const Eigen::VectorXd>> &myarrays){
        //auto seg = myarrays[Q::E].segment<4>(0);
        auto seg = myarrays[Q::E].segment<4>(0);
        auto mat = Eigen::Map<const Eigen::Matrix2d>(seg.data()); 
        std::cout << mat;
        //mat *= 42.;
}
// void eigen_reference_wrapper(std::vector<std::reference_wrapper<const Eigen::Ref<const Eigen::VectorXd>>> &myarrays){
//         //auto seg = myarrays[Q::E].segment<4>(0);
//         auto const seg = myarrays[0].get().segment<4>(0);
//         //auto mat = Eigen::Map<const Eigen::Matrix2d>(seg.data()); 
//         std::cout << seg;
//         //mat *= 42.;
// }
void eigen_dict_ref(std::map<std::string, Eigen::Ref<Eigen::VectorXd>> &myarrays){
        auto seg = myarrays.at("E").segment<4>(0);
        auto mat = Eigen::Map<Eigen::Matrix2d>(seg.data()); 
        mat *= 42.;
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
            .value("LAMBDA", Q::LAMBDA)
            .value("NONLOCAL", Q::NONLOCAL)
            .value("LAST", Q::LAST);

    m.def("g_dim", &Dim::G);
    m.def("q_dim", &Dim::StressStrain);
    m.def("eigen_mul",&eigen_ref);
    m.def("eigen_dict",&eigen_dict_ref);
    //m.def("eigen_const", &eigen_reference_wrapper);
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
     **   Strain Manipulations
     *************************************************************************/
    m.def("apply_b_bar_3d", &ApplyBBar<FULL>);

    /*************************************************************************
     **   "PURE" MECHANICS LAWS
     *************************************************************************/
    pybind11::class_<RefLawInterface, std::shared_ptr<RefLawInterface>> law(m, "RefLawInterface");

    
    pybind11::class_<LinearElastic<FULL>, std::shared_ptr<LinearElastic<FULL>>> linearElastic(m, "LinearElastic3D");
    linearElastic.def(pybind11::init<double, double, int>(), py::arg("E"), py::arg("nu"), py::arg("number of quadrature points"));
    //linearElastic.def("evaluate", &LinearElastic<FULL>::EvaluateAll);
    //linearElastic.def("update", &LinearElastic<FULL>::UpdateAll);
    linearElastic.def("inputs", &LinearElastic<FULL>::DefineInputs);
    linearElastic.def("evaluate", py::overload_cast<std::vector<Eigen::Ref<Eigen::VectorXd>>&, double>(&LinearElastic<FULL>::EvaluateAll),
               py::arg("input list"), py::arg("del t"));
//     linearElastic.def("evaluate", py::overload_cast<std::map<std::string,Eigen::Ref<Eigen::VectorXd>>&, double>(&LinearElastic<FULL>::EvaluateAll),
//                py::arg("input dictionary"), py::arg("del t"));
    
    
    pybind11::class_<VFLinearElastic<FULL>, std::shared_ptr<VFLinearElastic<FULL>>> vf_linear_elastic(m, "VFLinearElastic3D");
    vf_linear_elastic.def(pybind11::init<double, double, int>(), py::arg("E"), py::arg("nu"), py::arg("number of quadrature points"));
    vf_linear_elastic.def("evaluate", &VFLinearElastic<FULL>::EvaluateAll);


    /*************************************************************************
     **   Dynamics Laws
     *************************************************************************/

    pybind11::class_<HypoElastic<FULL>, std::shared_ptr<HypoElastic<FULL>>> hypo_elastic(m, "HypoElastic3D");
    hypo_elastic.def(pybind11::init<double, double, int>(), py::arg("E"), py::arg("nu"), py::arg("number of quadrature points"));
    hypo_elastic.def("evaluate", py::overload_cast<std::vector<Eigen::Ref<Eigen::VectorXd>>&, double>(&HypoElastic<FULL>::EvaluateAll),
               py::arg("input list"), py::arg("del t"));
    //hypo_elastic.def("evaluate", &HypoElastic<FULL>::EvaluateAll);
    //hypo_elastic.def("update", &HypoElastic<FULL>::UpdateAll);
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
    jh2_parameters.def_readwrite("TENSILE_EOS", &JH2Parameters::TENSILE_EOS);

    pybind11::class_<JH2<FULL>, std::shared_ptr<JH2<FULL>>> jh2(m, "JH23D");
    jh2.def(pybind11::init<std::shared_ptr<JH2Parameters>, int>(), py::arg("JH2Parameters"), py::arg("number of quadrature points"));
    jh2.def("get_internal_var", &JH2<FULL>::GetInternalVar);
    jh2.def("evaluate", py::overload_cast<std::vector<Eigen::Ref<Eigen::VectorXd>>&, double>(&JH2<FULL>::EvaluateAll),
               py::arg("input list"), py::arg("del t"));
    jh2.def("evaluate_some", py::overload_cast<std::vector<Eigen::Ref<Eigen::VectorXd>>&, Eigen::Ref<Eigen::VectorXi>, double>(&JH2<FULL>::EvaluateSome),
               py::arg("input list"), py::arg("indices"), py::arg("del t"));
    //jh2.def("evaluate", &JH2<FULL>::EvaluateAll);
    //jh2.def("update", &JH2<FULL>::UpdateAll);
    jh2.def("inputs", &JH2<FULL>::DefineInputs);
    
    pybind11::class_<JH2Nonlocal<FULL>, std::shared_ptr<JH2Nonlocal<FULL>>> jh2_nonlocal(m, "JH2Nonlocal3D");
    jh2_nonlocal.def(pybind11::init<std::shared_ptr<JH2Parameters>, int>(), py::arg("JH2Parameters"), py::arg("number of quadrature points"));
    jh2_nonlocal.def("get_internal_var", &JH2Nonlocal<FULL>::GetInternalVar);
    jh2_nonlocal.def("evaluate", py::overload_cast<std::vector<Eigen::Ref<Eigen::VectorXd>>&, double>(&JH2Nonlocal<FULL>::EvaluateAll),
               py::arg("input list"), py::arg("del t"));
    //jh2_nonlocal.def("evaluate", &JH2Nonlocal<FULL>::EvaluateAll);
    //jh2_nonlocal.def("update", &JH2Nonlocal<FULL>::UpdateAll);
    jh2_nonlocal.def("inputs", &JH2Nonlocal<FULL>::DefineInputs);

    /*************************************************************************
     **   PLASTICITY
     *************************************************************************/
}
