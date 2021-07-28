#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "interfaces.h"
#include "linear_elastic.h"
#include "local_damage.h"
//#include "plasticity.h"
#include "hypoelasticity.h"

namespace py = pybind11;

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
            .value("L", Q::L)
            .value("TIME_STEP", Q::TIME_STEP)
            .value("KAPPA", Q::KAPPA)
            .value("DEEQ", Q::DEEQ)
            .value("DSIGMA_DE", Q::DSIGMA_DE);

    m.def("g_dim", &Dim::G);
    m.def("q_dim", &Dim::Q);
    
    m.def("jaumann_rotate_W", &jaumann_rotate_W);
    m.def("jaumann_rotate_L", &jaumann_rotate_L);
    m.def("compute_W", &compute_W);
    m.def("strain_increment", &strain_increment);

    pybind11::class_<JaumannUpdater, std::shared_ptr<JaumannUpdater>> jaumann_updater(m, "JaumannUpdater");
    jaumann_updater.def(pybind11::init<int>(), py::arg("n"));
    jaumann_updater.def("set", &JaumannUpdater::Set, py::arg("L"));
    jaumann_updater.def("__call__", &JaumannUpdater::Rotate, py::arg("stress"), py::arg("stepsize"));
    
    /*************************************************************************
     **   IPLOOP AND MAIN INTERFACES
     *************************************************************************/

    pybind11::class_<IpLoop> ipLoop(m, "IpLoop");
    ipLoop.def(pybind11::init<>());
    ipLoop.def("add_law", py::overload_cast<std::shared_ptr<MechanicsLaw>, std::vector<int>>(&IpLoop::AddLaw),
               py::arg("law"), py::arg("ips") = std::vector<int>());
    ipLoop.def("add_law", py::overload_cast<std::shared_ptr<LawInterface>, std::vector<int>>(&IpLoop::AddLaw),
               py::arg("law"), py::arg("ips") = std::vector<int>());

    ipLoop.def("evaluate", py::overload_cast<const Eigen::VectorXd&, const Eigen::VectorXd&>(&IpLoop::Evaluate),
               py::arg("eps"), py::arg("e") = Eigen::VectorXd());
    ipLoop.def("evaluate", py::overload_cast<>(&IpLoop::Evaluate));
    //ipLoop.def("evaluate", &IpLoop::Evaluate, py::arg("eps"), py::arg("e") = Eigen::VectorXd());
    ipLoop.def("update", &IpLoop::Update, py::arg("eps"), py::arg("e") = Eigen::VectorXd());
    ipLoop.def("resize", &IpLoop::Resize);
    ipLoop.def("get", &IpLoop::Get);
    ipLoop.def("get_ips", &IpLoop::GetIPs);
    ipLoop.def("set", &IpLoop::Set);
    ipLoop.def("required_inputs", &IpLoop::RequiredInputs);

    pybind11::class_<LawInterface, std::shared_ptr<LawInterface>> law(m, "LawInterface");

    pybind11::class_<MechanicsLaw, std::shared_ptr<MechanicsLaw>> mechanicsLaw(m, "MechanicsLaw");
    mechanicsLaw.def("evaluate", &MechanicsLaw::Evaluate, py::arg("strain"), py::arg("i") = 0);
    mechanicsLaw.def("update", &MechanicsLaw::Update, py::arg("strain"), py::arg("i") = 0);
    mechanicsLaw.def("resize", &MechanicsLaw::Resize, py::arg("n"));

    /*************************************************************************
     **   DAMAGE LAWS
     *************************************************************************/

    pybind11::class_<DamageLawInterface, std::shared_ptr<DamageLawInterface>> damageLaw(m, "DamageLawInterface");
    damageLaw.def("evaluate", &DamageLawInterface::Evaluate);

    pybind11::class_<DamageLawExponential, std::shared_ptr<DamageLawExponential>, DamageLawInterface> damageExponential(
            m, "DamageLawExponential");
    damageExponential.def(pybind11::init<double, double, double>(), py::arg("k0"), py::arg("alpha"), py::arg("beta"));


    /*************************************************************************
     **   STRAIN NORMS
     *************************************************************************/

    pybind11::class_<StrainNormInterface, std::shared_ptr<StrainNormInterface>> strainNorm(m, "StrainNormInterface");
    strainNorm.def("evaluate", &StrainNormInterface::Evaluate);

    pybind11::class_<ModMisesEeq, std::shared_ptr<ModMisesEeq>, StrainNormInterface> modMises(m, "ModMisesEeq");
    modMises.def(pybind11::init<double, double, Constraint>(), py::arg("k"), py::arg("nu"), py::arg("constraint"));

    /*************************************************************************
     **   "PURE" MECHANICS LAWS
     *************************************************************************/

    pybind11::class_<LinearElastic, std::shared_ptr<LinearElastic>, MechanicsLaw> linearElastic(m, "LinearElastic");
    linearElastic.def(pybind11::init<double, double, Constraint>(), py::arg("E"), py::arg("nu"), py::arg("constraint"));

    pybind11::class_<Hypoelasticity, std::shared_ptr<Hypoelasticity>, LawInterface> hypoelasticity(m, "Hypoelasticity");
    hypoelasticity.def(pybind11::init<double, double>(), py::arg("E"), py::arg("nu"));
    hypoelasticity.def_readonly("C", &Hypoelasticity::C);

    pybind11::class_<HookesLaw, std::shared_ptr<HookesLaw>, LawInterface> hookes_law(m, "HookesLaw");
    hookes_law.def(pybind11::init<double, double, bool, bool>(), py::arg("E"), py::arg("nu"), py::arg("total_strains") = true, py::arg("tangent") = true);
    hookes_law.def_readonly("C", &HookesLaw::_C);
    
    pybind11::class_<LocalDamage, std::shared_ptr<LocalDamage>, MechanicsLaw> local(m, "LocalDamage");
    local.def(pybind11::init<double, double, Constraint, std::shared_ptr<DamageLawInterface>,
                             std ::shared_ptr<StrainNormInterface>>());
    local.def("kappa", &LocalDamage::Kappa);

    /*************************************************************************
     **   GRADIENT DAMAGE LAW
     *************************************************************************/

    pybind11::class_<GradientDamage, std::shared_ptr<GradientDamage>, LawInterface> gdm(m, "GradientDamage");
    gdm.def(pybind11::init<double, double, Constraint, std::shared_ptr<DamageLawInterface>,
                           std ::shared_ptr<StrainNormInterface>>());
    gdm.def("kappa", &GradientDamage::Kappa);

    /*************************************************************************
     **   PLASTICITY
     *************************************************************************/
    //pybind11::class_<NormVM> normVM(m, "NormVM");
    //normVM.def(pybind11::init<Constraint>());
    //normVM.def("__call__", &NormVM::Call);
    //normVM.def_readonly("P", &NormVM::_P);
    
    //pybind11::class_<RateIndependentHistory> RateIndependentHistory(m, "RateIndependentHistory");
    //RateIndependentHistory.def(pybind11::init<>());
    //RateIndependentHistory.def("__call__", &RateIndependentHistory::Call);
//     RateIndependentHistory.def_readonly("P", &RateIndependentHistory::_p);
}
