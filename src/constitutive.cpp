#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "interfaces.h"
#include "linear_elastic.h"
#include "local_damage.h"
#include "radial_plasticity.h"
#include "hypoelasticity.h"
#include "mises_eos.h"
#include "jh.h"
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
            .value("EPS_P", Q::EPS_P)
            .value("EEQ", Q::EEQ)
            .value("EPS", Q::EPS)
            .value("E", Q::E)
            .value("L", Q::L)
            .value("TIME_STEP", Q::TIME_STEP)
            .value("KAPPA", Q::KAPPA)
            .value("LAMBDA", Q::LAMBDA)
            .value("DEEQ", Q::DEEQ)
            .value("DSIGMA_DE", Q::DSIGMA_DE)
            .value("RHO", Q::RHO)
            .value("DAMAGE", Q::DAMAGE);


    m.def("g_dim", &Dim::G);
    m.def("q_dim", &Dim::Q);
    
    m.def("strain_increment", &strain_increment);

    
    /*************************************************************************
     **   IPLOOP AND MAIN INTERFACES
     *************************************************************************/
    pybind11::class_<QValues, std::shared_ptr<QValues>> q_values(m, "QValues");
    q_values.def(pybind11::init<int,int>(), py::arg("rows"), py::arg("cols")=1);
    q_values.def("resize", &QValues::Resize);
    q_values.def("set", py::overload_cast<double,int>(&QValues::Set));
    q_values.def("set", py::overload_cast<Eigen::MatrixXd,int>(&QValues::Set));
    q_values.def("get_scalar", &QValues::GetScalar);
    q_values.def("get", &QValues::Get);
    q_values.def_readonly("data", &QValues::data);

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
    ipLoop.def("update", py::overload_cast<const Eigen::VectorXd&, const Eigen::VectorXd&>(&IpLoop::Update),
                py::arg("eps"), py::arg("e") = Eigen::VectorXd());
    ipLoop.def("update", py::overload_cast<>(&IpLoop::Update));
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
    
    /************************************************************************
     **   OBJECTIVE STRESS RATES
     ************************************************************************/
    pybind11::class_<ObjectiveStressRate, std::shared_ptr<ObjectiveStressRate>> objective_stress_rate(m, "ObjectiveStressRate");
    pybind11::class_<JaumannStressRate, std::shared_ptr<JaumannStressRate>, ObjectiveStressRate> jaumann_updater(m, "JaumannStressRate");
    jaumann_updater.def(pybind11::init<>());
    jaumann_updater.def("set", &JaumannStressRate::Set, py::arg("L"));
    jaumann_updater.def("resize", &JaumannStressRate::Resize, py::arg("n"));
    jaumann_updater.def("__call__", &JaumannStressRate::Rotate, py::arg("stress"), py::arg("stepsize"));
    
    
    /*************************************************************************
     **   RADIAL PLASTICITY
     *************************************************************************/
    pybind11::class_<AnalyticMisesPlasticity, std::shared_ptr<AnalyticMisesPlasticity>, LawInterface> analytic_mises_plasticity(m, "AnalyticMisesPlasticity");
    analytic_mises_plasticity.def(pybind11::init<double, double, double, double, bool, bool>(), py::arg("E"), py::arg("nu"), py::arg("sig0"), py::arg("H"),py::arg("total_strains") = true, py::arg("tangent") = true);
    analytic_mises_plasticity.def("get_internal_var", &AnalyticMisesPlasticity::GetInternalVar);
    analytic_mises_plasticity.def_readonly("C", &AnalyticMisesPlasticity::_C);

    //pybind11::class_<RadialReturnMisesPlasticity, std::shared_ptr<RadialReturnMisesPlasticity>, LawInterface> radial_return_mises_plasticity(m, "RadialReturnMisesPlasticity");
    //radial_return_mises_plasticity.def(pybind11::init<double, double, double, double>(), py::arg("E"), py::arg("nu"), py::arg("sig0"), py::arg("H"));
    //radial_return_mises_plasticity.def("get_internal_var", &RadialReturnMisesPlasticity::GetInternalVar);
    //radial_return_mises_plasticity.def_readonly("C", &RadialReturnMisesPlasticity::_C);

    pybind11::class_<RadialReturnPlasticity, std::shared_ptr<RadialReturnPlasticity>, LawInterface> radial_return_plasticity(m, "RadialReturnPlasticity");
    radial_return_plasticity.def(pybind11::init<double, double, std::shared_ptr<RadialYieldSurface>>(), py::arg("E"), py::arg("nu"), py::arg("Y"));
    radial_return_plasticity.def("get_internal_var", &RadialReturnPlasticity::GetInternalVar);
    radial_return_plasticity.def_readonly("C", &RadialReturnPlasticity::_C);

    pybind11::class_<RadialYieldSurface, std::shared_ptr<RadialYieldSurface>> radial_yield_surface(m, "RadialYieldSurface");

    pybind11::class_<RadialMisesYieldSurface, std::shared_ptr<RadialMisesYieldSurface>, RadialYieldSurface> radial_mises_yield_surface(m, "RadialMisesYieldSurface");
    radial_mises_yield_surface.def(pybind11::init<double, double>(), py::arg("sig0"), py::arg("H"));
    
    /*************************************************************************
     ** Hydrocode models
     *************************************************************************/
    pybind11::class_<MisesEOS, std::shared_ptr<MisesEOS>, LawInterface> mises_eos(m, "MisesEOS");
    mises_eos.def(pybind11::init<double, double, double, double, std::shared_ptr<EOSInterface>>(), py::arg("sig0"), py::arg("mu"), py::arg("rho"), py::arg("H"), py::arg("EOS"));
    mises_eos.def("get_internal_var", &MisesEOS::GetInternalVar);

    pybind11::class_<EOSInterface, std::shared_ptr<EOSInterface>> eos_interface(m, "EOSInterface");

    pybind11::class_<PolynomialEOS, std::shared_ptr<PolynomialEOS>, EOSInterface> polynomial_eos(m, "PolynomialEOS");
    polynomial_eos.def(pybind11::init<Eigen::VectorXd, int, int>(), py::arg("coeff"), py::arg("deg A"), py::arg("deg B"));


    /*************************************************************************
     ** Johnson Holmquist material model for ceramics
     *************************************************************************/

    pybind11::class_<JH2Parameters, std::shared_ptr<JH2Parameters>> jh2_parameters(m, "JH2Parameters");
    jh2_parameters.def(pybind11::init<>());
    jh2_parameters.def_readwrite("RHO", &JH2Parameters::RHO);
    jh2_parameters.def_readwrite("SHEAR_MODULUS", &JH2Parameters::SHEAR_MODULUS);

    pybind11::class_<JH2, std::shared_ptr<JH2>, LawInterface> jh2(m, "JH2");
    jh2.def(pybind11::init<std::shared_ptr<JH2Parameters>>(), py::arg("JH2Parameters"));
    jh2.def("get_internal_var", &JH2::GetInternalVar);

}
