#pragma once
#include "interfaces.h"

template <Constraint TC>
MandelMatrix<TC> C(double E, double nu);

template <>
MandelMatrix<UNIAXIAL_STRAIN> C<UNIAXIAL_STRAIN>(double E, double nu)
{
    return MandelMatrix<UNIAXIAL_STRAIN>::Constant(E);
}

template <>
MandelMatrix<UNIAXIAL_STRESS> C<UNIAXIAL_STRESS>(double E, double nu)
{
    return MandelMatrix<UNIAXIAL_STRAIN>::Constant(E);
}
/*
template <>
MandelMatrix<PLANE_STRESS> C<PLANE_STRESS>(double E, double nu)
{
    const double C11 = E / (1 - nu * nu);
    const double C12 = nu * C11;
    const double C33 = (1 - nu) * 0.5 * C11;
    MandelMatrix<PLANE_STRAIN> c;
    c << C11, C12, 0, C12, C11, 0, 0, 0, C33;
    return c;
}

template <>
MandelMatrix<PLANE_STRAIN> C<PLANE_STRAIN>(double E, double nu)
{
    const double l = E * nu / (1 + nu) / (1 - 2 * nu);
    const double m = E / (2.0 * (1 + nu));
    MandelMatrix<PLANE_STRAIN> c = MandelVector<PLANE_STRAIN>({2 * m, 2 * m, m}).asDiagonal();
    c.block<2, 2>(0, 0) += Eigen::Matrix2d::Constant(l);
    return c;
}*/

template <>
MandelMatrix<FULL> C<FULL>(double E, double nu)
{
    const double l = E * nu / (1 + nu) / (1 - 2 * nu);
    const double m = E / (2.0 * (1 + nu));

    MandelVector<FULL> diagonal;
    diagonal.segment<6>(0) = Eigen::Matrix<double, 6, 1>::Constant(2 * m);
    //diagonal.segment<3>(3) = Eigen::Vector3d::Constant(m);
    MandelMatrix<FULL> c = diagonal.asDiagonal();
    c.block<3, 3>(0, 0) += Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Constant(l);
    return c;
}

Eigen::MatrixXd C(double E, double nu, Constraint c)
{
    if (c == UNIAXIAL_STRAIN)
        return C<UNIAXIAL_STRAIN>(E, nu);
    if (c == UNIAXIAL_STRESS)
        return C<UNIAXIAL_STRESS>(E, nu);
    if (c == PLANE_STRAIN)
        return C<PLANE_STRAIN>(E, nu);
    if (c == PLANE_STRESS)
        return C<PLANE_STRESS>(E, nu);
    if (c == FULL)
        return C<FULL>(E, nu);
    throw std::runtime_error("Stuffy");
}

template <Constraint TC> class LinearElastic : public RefLawInterface
{
public:
    MandelMatrix<TC> _C;
    
    LinearElastic(double E, double nu, int n)
        : RefLawInterface(n)
    {
        _C = C<TC>(E, nu);
    }
    std::vector<Q> DefineInputs() const override
    {
        std::vector<Q> inputs = {Q::EPS, Q::SIGMA, Q::DSIGMA_DEPS};
        return inputs;
    }
    void EvaluateIP(int i, std::vector<Eigen::Ref<Eigen::VectorXd>>& inputs, double del_t) override
    {
        const int dim = Dim::StressStrain(TC);
        const auto eps = inputs[Q::EPS].segment<dim>(i*dim);
        auto sigma = inputs[Q::SIGMA].segment<dim>(i*dim);
        auto Ct_flat = inputs[Q::DSIGMA_DEPS].segment<dim*dim>(i*dim*dim);

        sigma = _C * eps;
        Ct_flat = Eigen::Map<Eigen::VectorXd>(_C.data(), _C.size());
    }
};
template <Constraint TC> class MapLinearElastic : public MapLawInterface
{
public:
    MandelMatrix<TC> _C;
    
    MapLinearElastic(std::map<std::string, double> &parameters, int n)
        : MapLawInterface(parameters, n)
    {
        _C = C<TC>(_parameters.at("E"), _parameters.at("nu"));
    }
    std::map<std::string, std::pair<int,int>> DefineInput() const override
    {
        std::map<std::string, std::pair<int,int>> inputs = {{"eps",{Dim::StressStrain(TC), 1}},
                                                            {"sigma",{Dim::StressStrain(TC), 1}},
                                                            {"dsigma_deps",{Dim::StressStrain(TC),Dim::StressStrain(TC)}}};
        return inputs;
    }
    std::map<std::string, std::pair<int,int>> DefineInternal() const override
    {
        std::map<std::string, std::pair<int,int>> internal;
        return internal;
    }

    void EvaluateIP(int i, std::map<std::string, Eigen::Ref<Eigen::VectorXd>>& input, std::map<std::string, Eigen::Ref<Eigen::VectorXd>>& internal, double del_t) override
    {
        const int dim = Dim::StressStrain(TC);
        const auto eps = input.at("eps").segment<dim>(i*dim);
        auto sigma = input.at("sigma").segment<dim>(i*dim);
        auto Ct_flat = input.at("dsigma_deps").segment<dim*dim>(i*dim*dim);

        sigma = _C * eps;
        Ct_flat = Eigen::Map<Eigen::VectorXd>(_C.data(), _C.size());
    }
};

template <Constraint TC> class VFLinearElastic
{
//This classs is prob. as fast as it gets with my knowledge of c++
//Good for comparison with other interfaced classes and working out
//possible bottlenecks and costs of other abstractions
public:
    MandelMatrix<TC> _C;
    int _n;
    VFLinearElastic(double E, double nu, int n)
    {
        _n = n;
        _C = C<TC>(E, nu);
    }
    void EvaluateAll(Eigen::Ref<Eigen::VectorXd> eps,Eigen::Ref<Eigen::VectorXd> sigma, Eigen::Ref<Eigen::VectorXd> tangent, double del_t)
    {
        const int dim = Dim::StressStrain(TC);
        MandelVector<TC> eps_i;

        //const int dim_Ct_flat = dim*dim;
        for(int i = 0; i<_n;i++){
            eps_i = eps(Eigen::seqN(i*dim,Eigen::fix<dim>));
            auto sigma_i = sigma(Eigen::seqN(i*dim,Eigen::fix<dim>));
            auto Ct_flat = tangent(Eigen::seqN(i*dim*dim, Eigen::fix<dim*dim>));

            sigma_i = _C * eps_i;
            Ct_flat =Eigen::Map<Eigen::VectorXd>(_C.data(), _C.size());
        }
    }
};


