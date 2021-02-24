#pragma once
#include "interfaces.h"
#include <cmath>
#include <tuple>
#include <iostream>

class DamageLawExponential
{
public:
    DamageLawExponential(double k0, double alpha, double beta)
        : _k0(k0)
        , _a(alpha)
        , _b(beta)
    {
    }

    std::pair<double, double> evaluate(double k) const
    {
        if (k <= _k0)
            return {0., 0.};
        const double omega = 1 - _k0 / k * (1 - _a + _a * std::exp(_b * (_k0 - k)));
        const double domega = _k0 / k * ((1 / k + _b) * _a * std::exp(_b * (_k0 - k)) + (1 - _a) / k);
        return {omega, domega};
    }

private:
    const double _k0;
    const double _a;
    const double _b;
};

std::pair<double, V<FULL>> InvariantI1(V<FULL> v)
{
    const double I1 = v[0] + v[1] + v[2];
    V<FULL> dI1 = V<FULL>::Zero();
    dI1.segment<3>(0) = Eigen::Vector3d::Ones();
    return {I1, dI1};
}

std::pair<double, V<FULL>> InvariantJ2(V<FULL> v)
{
    const double J2 =
            ((v[0] - v[1]) * (v[0] - v[1]) + (v[1] - v[2]) * (v[1] - v[2]) + (v[2] - v[0]) * (v[2] - v[0])) / 6. +
            0.25 * (v[3] * v[3] + v[4] * v[4] + v[5] * v[5]);
    V<FULL> dJ2;
    dJ2[0] = (2. * v[0] - v[1] - v[2]) / 3.;
    dJ2[1] = (2. * v[1] - v[2] - v[0]) / 3.;
    dJ2[2] = (2. * v[2] - v[0] - v[1]) / 3.;
    dJ2[3] = 0.5 * v[3];
    dJ2[4] = 0.5 * v[4];
    dJ2[5] = 0.5 * v[5];
    return {J2, dJ2};
}

Eigen::Matrix<double, 6, Eigen::Dynamic> T3D(double nu, Constraint c)
{
    Eigen::Matrix<double, 6, Eigen::Dynamic> T(6, Dim::Q(c));
    T.setZero();
    switch (c)
    {
    case Constraint::UNIAXIAL_STRAIN:
    {
        T(0, 0) = 1;
        break;
    }
    case Constraint::UNIAXIAL_STRESS:
    {
        T(0, 0) = 1;
        T(1, 0) = -nu;
        T(2, 0) = -nu;
        break;
    }
    case Constraint::PLANE_STRAIN:
    {
        T(0, 0) = 1;
        T(1, 1) = 1;
        T(5, 2) = 1;
        break;
    }
    case Constraint::PLANE_STRESS:
    {
        T(0, 0) = 1;
        T(1, 1) = 1;
        T(2, 0) = nu / (nu - 1.);
        T(2, 1) = nu / (nu - 1.);
        T(5, 2) = 1;
        break;
    }
    case Constraint::FULL:
    {
        T = M<FULL>::Identity();
        break;
    }
    }
    return T;
}

class ModMisesEeq
{
public:
    ModMisesEeq(double k, double nu, Constraint c)
        : _K1((k - 1.0) / (2.0 * k * (1.0 - 2.0 * nu)))
        , _K2(3.0 / (k * (1.0 + nu) * (1.0 + nu)))
        , _nu(nu)
        , _c(c)
        , _T3D(T3D(_nu, _c))
    {
    }

    std::pair<double, Eigen::VectorXd> evaluate(Eigen::VectorXd strain) const
    {
        // transformation to 3D and invariants
        const V<FULL> strain3D = _T3D * strain;
        double I1, J2;
        V<FULL> dI1, dJ2;
        std::tie(I1, dI1) = InvariantI1(strain3D);
        std::tie(J2, dJ2) = InvariantJ2(strain3D);

        // actual modified mises norm
        const double A = std::sqrt(_K1 * _K1 * I1 * I1 + _K2 * J2) + 1.e-14;
        const double eeq = _K1 * I1 + A;
        const double deeq_dI1 = _K1 + _K1 * _K1 * I1 / A;
        const double deeq_dJ2 = _K2 / (2 * A);
        //
        //// derivative in 3D and transformation back
        V<FULL> deeq = deeq_dI1 * dI1 + deeq_dJ2 * dJ2;
        return {eeq, _T3D.transpose() * deeq};
    }

private:
    const double _K1;
    const double _K2;
    const double _nu;
    const Constraint _c;
    const Eigen::Matrix<double, 6, Eigen::Dynamic> _T3D;
};


class LocalDamage : public IpBase
{
public:
    LocalDamage(double E, double nu, Constraint c, double ft, double alpha, double gf, double k)
        : _C(C(E, nu, c))
        , _omega(ft / E, alpha, ft / gf)
        , _eeq(k, nu, c)
    {
    }

    void resize(int n) override
    {
        _kappa.resize(n);
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate(const Eigen::VectorXd& strain, int i) override
    {
        double kappa, dkappa, omega, domega, eeq;
        Eigen::VectorXd deeq;

        std::tie(eeq, deeq) = _eeq.evaluate(strain);
        std::tie(kappa, dkappa) = evaluate_kappa(eeq, _kappa[i]);
        std::tie(omega, domega) = _omega.evaluate(kappa);

        return {(1. - omega) * _C * strain, (1. - omega) * _C - _C * strain * domega * dkappa * deeq.transpose()};
    }

    virtual int qdim() const override
    {
        return _C.rows();
    }

    std::pair<double, double> evaluate_kappa(double eeq, double kappa) const
    {
        if (eeq >= kappa)
            return {eeq, 1.};
        else
            return {kappa, 0};
    }

    virtual void update(const Eigen::VectorXd& strain, int i) override
    {
        const double eeq = _eeq.evaluate(strain).first;
        const double kappa = evaluate_kappa(eeq, _kappa[i]).first;
        _kappa[i] = kappa;
    }


private:
    Eigen::MatrixXd _C;
    DamageLawExponential _omega;
    ModMisesEeq _eeq;
    Eigen::VectorXd _kappa;
};

class GradientDamage
{
public:
    GradientDamage(double E, double nu, Constraint c, double ft, double alpha, double beta, double k)
        : _C(C(E, nu, c))
        , _omega(ft / E, alpha, beta)
        , _eeq(k, nu, c)
    {
    }

    void resize(int n)
    {
        _kappa.resize(n);
    }

    std::tuple<double, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd>
    evaluate(const Eigen::VectorXd& strain, const double neeq, int i)
    {
        double kappa, dkappa, omega, domega, eeq;
        Eigen::VectorXd deeq;

        std::tie(kappa, dkappa) = evaluate_kappa(neeq, _kappa[i]);
        std::tie(omega, domega) = _omega.evaluate(kappa);
        std::tie(eeq, deeq) = _eeq.evaluate(strain);

        // kappa = 0.;
        // omega = 0.;
        // dkappa = 0.;
        // domega = 0.;
        // eeq = 0.;
        // deeq.setZero(strain.rows());
        //
        return {eeq, (1. - omega) * _C * strain, deeq, -_C * strain * domega * dkappa, (1. - omega) * _C};
    }

    virtual int qdim() const
    {
        return _C.rows();
    }

    std::pair<double, double> evaluate_kappa(double eeq, double kappa) const
    {
        if (eeq >= kappa)
            return {eeq, 1.};
        else
            return {kappa, 0};
    }

    virtual void update(const Eigen::VectorXd& strain, const double neeq, int i)
    {
        _kappa[i] = evaluate_kappa(neeq, _kappa[i]).first;
    }


private:
    Eigen::MatrixXd _C;
    DamageLawExponential _omega;
    ModMisesEeq _eeq;
    Eigen::VectorXd _kappa;
};

class BaseGDM
{
public:
    BaseGDM(GradientDamage& law)
        : _law(law)
    {
    }
    virtual void resize(int n)
    {
        _n = n;
        int q = _law.qdim();
        _law.resize(n);
        // scalar variables
        _eeq.resize(n);
        // vector variables
        _stress.resize(_n * q);
        _deeq.resize(_n * q);
        _dstress_deeq.resize(_n * q);
        // tensor variables
        _dstress_deps.resize(_n * q * q);
    }


    virtual void evaluate(const Eigen::VectorXd& all_strains, const Eigen::VectorXd& all_neeq)
    {
        int q = _law.qdim();
        const int n = all_strains.rows() / q;
        if (_stress.rows() == 0)
            resize(n);

        assert(n == _n);
        for (int i = 0; i < _n; ++i)
        {
            Eigen::VectorXd strain = all_strains.segment(i * q, q);
            const double neeq = all_neeq[i];
            auto eval = _law.evaluate(strain, neeq, i);
            // scalar variables
            _eeq[i] = std::get<0>(eval);

            // vector variables
            _stress.segment(i * q, q) = std::get<1>(eval);
            _deeq.segment(i * q, q) = std::get<2>(eval);
            _dstress_deeq.segment(i * q, q) = std::get<3>(eval);

            // tensor variables
            auto t = std::get<4>(eval);
            _dstress_deps.segment(q * q * i, q * q) = Eigen::Map<Eigen::VectorXd>(t.data(), t.size());
        }
    }

    virtual void update(const Eigen::VectorXd& all_strains, const Eigen::VectorXd& all_neeq)
    {
        int q = _law.qdim();
        for (int i = 0; i < _n; ++i)
        {
            Eigen::VectorXd strain = all_strains.segment(i * q, q);
            const double neeq = all_neeq[i];
            _law.update(strain, neeq, i);
        }
    }

    GradientDamage& _law;
    int _n = 0;
    Eigen::VectorXd _stress;
    Eigen::VectorXd _dstress_deps;
    Eigen::VectorXd _dstress_deeq;
    Eigen::VectorXd _eeq;
    Eigen::VectorXd _deeq;
};

