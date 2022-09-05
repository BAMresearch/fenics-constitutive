#pragma once
#include "interfaces.h"
#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor-blas/xlinalg.hpp"

class LinearElastic : public MechanicsLawInterface
{
public:
    LinearElastic(double E, double nu, bool tangents, int n)
        : MechanicsLawInterface(tangents, n)
    {
        double m = E /(2*(1+nu));
        double l = E*nu/((1+nu)*(1-2*nu));

        _C = xt::pytensor<double,2>(
            {{2*m+l, l, l, 0., 0., 0.},
            {l, 2*m+l, l, 0., 0., 0.},
            {l, l, 2*m+l, 0., 0., 0.},
            {0., 0., 0., 2*m, 0., 0.},
            {0., 0., 0., 0., 2*m, 0.},
            {0., 0., 0., 0., 0., 2*m}});
    }
    inline void EvaluateIP(
            int i,
            xt::pytensor<double,1>& eps_vector,
            xt::pytensor<double,1>& sigma_vector,
            xt::pytensor<double,1>& tangents_vector,
            double del_t
            ) override
    {
        int stress_strain_dim = 6;
        auto eps = GetVectorView(eps_vector, i, stress_strain_dim);
        auto sigma = GetVectorView(sigma_vector, i, stress_strain_dim);
        auto Ct = GetQuadratureView(tangents_vector, i, stress_strain_dim, stress_strain_dim);
        sigma = xt::linalg::dot(_C,eps);
        Ct = _C;
    }
    xt::pytensor<double, 2> _C;
};

class EigenLinearElastic : public EigenMechanicsLawInterface
{
public:
    EigenLinearElastic(double E, double nu, bool tangents, int n)
        : EigenMechanicsLawInterface(tangents, n)
    {
        double m = E /(2*(1+nu));
        double l = E*nu/((1+nu)*(1-2*nu));
        _C.setZero(6,6);
        
        _C << 2*m+l, l, l, 0., 0., 42.,
             l, 2*m+l, l, 0., 0., 0.,
             l, l, 2*m+l, 0., 0., 0.,
             0., 0., 0., 2*m, 0., 0.,
             0., 0., 0., 0., 2*m, 0.,
             24., 0., 0., 0., 0., 2*m;
    }
    inline void EvaluateIP(
            int i,
            Eigen::Ref<Eigen::VectorXd> eps_vector,
            Eigen::Ref<Eigen::VectorXd> sigma_vector,
            Eigen::Ref<Eigen::VectorXd> tangents_vector,
            double del_t
            ) override
    {
        int stress_strain_dim = 6;
        auto eps = eps_vector.segment<6>(i*6);
        auto sigma = sigma_vector.segment<6>(i*6);
        auto Ct_flat = tangents_vector.segment<36>(i*36);
        //Eigen::MatrixXd C_copy = _C;
        //C_copy.transposeInPlace();
        //Ct_flat = Eigen::Map<Eigen::VectorXd>(C_copy.data(), C_copy.size());
        Ct_flat = Eigen::Map<Eigen::VectorXd>(_C.data(), _C.size());
        
        sigma = _C * eps;
        //Ct = _C;
    }
    RowMatrixXd _C;
};
