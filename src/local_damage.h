#pragma once
#include "interfaces.h"
#include <cmath>

class DamageLawExponential
{
public:
    DamageLawExponential(double k0, double beta, double alpha)
        : _k0(k0)
        , _b(beta)
        , _a(alpha)
    {
    }

    std::pair<double, double> evaluate(double k) const
    {
        if (k < _k0)
            return {0., 0.};
        const double omega = 1 - _k0 / k * (1 - _a + _a * std::exp(_b * (_k0 - k)));
        const double domega = _k0 / k * ((1 / k + _b) * _a * std::exp(_b * (_k0 - k)) + (1 - _a) / k);
        return {omega, domega};
    }

private:
    const double _k0;
    const double _b;
    const double _a;
};


std::pair<double, Eigen::VectorXd> I1(Eigen::VectorXd v, double nu, Constraint c)
{
    switch (c)
    {
    case Constraint::UNIAXIAL_STRAIN:
    {
        return {v[0], Eigen::Matrix<double, 1, 1>::Constant(1)};
    }
    case Constraint::UNIAXIAL_STRESS:
    {
        return {(1. - 2 * nu) * v[0], Eigen::Matrix<double, 1, 1>::Constant(1. - 2 * nu)};
    }
    case Constraint::PLANE_STRAIN:
    {
        return {v[0] + v[1], {1, 1, 0}};
    }
    case Constraint::PLANE_STRESS:
    {
        const double f = 1 + nu / (nu - 1);
        return {f * v[0] + f * v[1], {f, f, 0}};
    }
    case Constraint::FULL:
    {
        V<FULL> d;
        d.segment<3>(0) = Eigen::Vector3d::Constant(1);
        d.segment<3>(3) = Eigen::Vector3d::Constant(0);
        return {v[0] + v[1] + v[2], d};
    }
    }
}

class ModMisesEeq
{
public:
    ModMisesEeq(double k, double nu, Constraint c)
        : _K1((k - 1.0) / (2.0 * k * (1.0 - 2.0 * nu)))
        , _K2(3.0 / (k * (1.0 + nu) * (1.0 + nu)))
        , _c(c)
    {
    }

    std::pair<double, Eigen::VectorXd> evaluate(Eigen::VectorXd strain) const
    {
    }

private:
    const double _K1;
    const double _K2;
    const Constraint _c;
};


class LocalDamage : public IpBase
{
public:
    LocalDamage(double E, double nu, Constraint TC, double ft, double alpha, double beta)
        : _C(E, nu, TC)
        , _omega(ft / E, alpha, beta)
    {
    }

    void allocate_history_data(int n) override
    {
        _kappa.resize(n);
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate(const Eigen::VectorXd& strain, int i = 0) override
    {
        return {_C * strain, _C};
    }

    virtual int qdim() const override
    {
        return _C.rows();
    }

    std::pair<double, double> kappa(double eeq, double kappa) const
    {
        if (eeq > kappa)
            return {eeq, 1.};
        else
            return {kappa, 0};
    }

private:
    Eigen::MatrixXd _C;
    DamageLawExponential _omega;
    Eigen::VectorXd _kappa;
};

