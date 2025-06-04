#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <eigen3/Eigen/Core>
#include <iostream>

inline Eigen::Vector<double, 6> strain_from_grad_u(const Eigen::Matrix3d &grad_u)
{
    constexpr double FACTOR = 0.707106781186547524400844362104849039;
    Eigen::Vector<double, 6> strain;
    strain(0) = grad_u(0, 0);
    strain(1) = grad_u(1, 1);
    strain(2) = grad_u(2, 2);
    strain(3) = FACTOR * (grad_u(0, 1) + grad_u(1, 0));
    strain(4) = FACTOR * (grad_u(0, 2) + grad_u(2, 0));
    strain(5) = FACTOR * (grad_u(1, 2) + grad_u(2, 1));
    return strain;
}

struct Elasticity3D
{
    Eigen::Matrix<double, 6, 6> D_;
    Elasticity3D(double E, double nu)
    {
        double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        double mu = E / (2.0 * (1.0 + nu));
        D_ << lambda + 2.0 * mu, lambda, lambda, 0.0, 0.0, 0.0,
            lambda, lambda + 2.0 * mu, lambda, 0.0, 0.0, 0.0,
            lambda, lambda, lambda + 2.0 * mu, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 2.0 * mu, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 2.0 * mu, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * mu;
    }

    void evaluate(
        double t,
        double del_t,
        const Eigen::Ref<const Eigen::VectorXd> &del_grad_u,
        Eigen::Ref<Eigen::VectorXd> &stress,
        Eigen::Ref<Eigen::VectorXd> &tangent,
        std::map<std::string, Eigen::Ref<Eigen::VectorXd>> &history)
    {
        int n_ip = del_grad_u.size() / 9;
        for (int ip = 0; ip < n_ip; ip++)
        {
            Eigen::Vector<double, 9> grad_u_flat = del_grad_u.segment<9>(9 * ip);
            stress.segment<6>(6 * ip) += D_ * strain_from_grad_u(Eigen::Map<Eigen::Matrix3d>(grad_u_flat.data()));
            tangent.segment<36>(36 * ip) = Eigen::Map<Eigen::Vector<double, 36>>(D_.data());
        }
    }
    
    std::map<std::string, int> history_dim() { 
        return {};
    }
    
};

namespace py = pybind11;

PYBIND11_MODULE(elasticity_cpp, m)
{
    m.doc() = "";
    py::class_<Elasticity3D> elasticity_3d(m, "Elasticity3D");
    elasticity_3d.def(py::init<double,double>(), py::arg("E"), py::arg("nu"));
    elasticity_3d.def("evaluate", &Elasticity3D::evaluate);
    elasticity_3d.def("history_dim", &Elasticity3D::history_dim);
}
