#pragma once
#include "interfaces.h"
#include <cmath>
#include <tuple>
#include <iostream>
#include <vector>

class DamageLawExponential
{
public:
    DamageLawExponential(double k0, double alpha, double beta)
        : _k0(k0)
        , _a(alpha)
        , _b(beta)
    {
    }

    std::pair<double, double> Evaluate(double k) const
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

    std::pair<double, Eigen::VectorXd> Evaluate(Eigen::VectorXd strain) const
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

        std::tie(eeq, deeq) = _eeq.Evaluate(strain);
        std::tie(kappa, dkappa) = EvaluateKappa(eeq, _kappa[i]);
        std::tie(omega, domega) = _omega.Evaluate(kappa);

        return {(1. - omega) * _C * strain, (1. - omega) * _C - _C * strain * domega * dkappa * deeq.transpose()};
    }

    virtual int qdim() const override
    {
        return _C.rows();
    }

    std::pair<double, double> EvaluateKappa(double eeq, double kappa) const
    {
        if (eeq >= kappa)
            return {eeq, 1.};
        else
            return {kappa, 0};
    }

    virtual void update(const Eigen::VectorXd& strain, int i) override
    {
        const double eeq = _eeq.Evaluate(strain).first;
        const double kappa = EvaluateKappa(eeq, _kappa[i]).first;
        _kappa[i] = kappa;
    }


private:
    Eigen::MatrixXd _C;
    DamageLawExponential _omega;
    ModMisesEeq _eeq;
    Eigen::VectorXd _kappa;
};

class QValues
{
public:
    QValues() = default;

    //! @brief stores n x rows x cols values where n is the number of IPs
    QValues(int rows, int cols = 1)
        : _rows(rows)
        , _cols(cols)
    {
    }

    void Resize(int n)
    {
        data.setZero(n * _rows * _cols);
    }

    void Set(double value, int i)
    {
        assert(_rows == 1);
        assert(_cols == 1);
        data[i] = value;
    }

    void Set(Eigen::MatrixXd value, int i)
    {
        assert(value.rows() == _rows);
        assert(value.cols() == _cols);
        data.segment(_rows * _cols * i, _rows * _cols) = Eigen::Map<Eigen::VectorXd>(value.data(), value.size());
    }

    double GetScalar(int i) const
    {
        assert(_rows == 1);
        assert(_cols == 1);
        return data[i];
    }

    Eigen::MatrixXd Get(int i) const
    {
        Eigen::VectorXd ip_values = data.segment(_rows * _cols * i, _rows * _cols);
        return Eigen::Map<Eigen::MatrixXd>(ip_values.data(), _rows, _cols);
    }

    bool IsUsed() const
    {
        return _rows != 0;
    }


    // private:
    int _rows = 0;
    int _cols = 0;
    Eigen::VectorXd data;
};

enum Q
{
    SIGMA,
    DSIGMA_DEPS,
    EEQ,
    DEEQ,
    DSIGMA_DE,
    EPS,
    E,
    LAST
};

class GradientDamage
{
public:
    GradientDamage(double E, double nu, Constraint c, double ft, double alpha, double beta, double k)
        : _C(C(E, nu, c))
        , _omega(ft / E, alpha, beta)
        , _strain_norm(k, nu, c)
        , _kappa(1)
    {
    }

    void DefineOutputs(std::vector<QValues>& out) const
    {
        const int q = _C.rows();
        out[EEQ] = QValues(1);
        out[DEEQ] = QValues(q);
        out[SIGMA] = QValues(q);
        out[DSIGMA_DE] = QValues(q);
        out[DSIGMA_DEPS] = QValues(q, q);
    }

    void DefineInputs(std::vector<QValues>& input) const
    {
        const int q = _C.rows();
        input[E] = QValues(1);
        input[EPS] = QValues(q);
    }

    void Resize(int n)
    {
        _kappa.Resize(n);
    }

    void Evaluate(const std::vector<QValues>& input, std::vector<QValues>& out, int i)
    {
        double kappa, dkappa, omega, domega, eeq;
        Eigen::VectorXd deeq;
        auto strain = input[EPS].Get(i);

        std::tie(kappa, dkappa) = EvaluateKappa(input[E].GetScalar(i), _kappa.GetScalar(i));
        std::tie(omega, domega) = _omega.Evaluate(kappa);
        std::tie(eeq, deeq) = _strain_norm.Evaluate(strain);

        out[EEQ].Set(eeq, i);
        out[SIGMA].Set((1. - omega) * _C * strain, i);
        out[DEEQ].Set(deeq, i);
        out[DSIGMA_DE].Set(-_C * strain * domega * dkappa, i);
        out[DSIGMA_DEPS].Set((1. - omega) * _C, i);
    }

    std::pair<double, double> EvaluateKappa(double eeq, double kappa) const
    {
        if (eeq >= kappa)
            return {eeq, 1.};
        else
            return {kappa, 0};
    }

    void Update(const std::vector<QValues>& input, int i)
    {
        _kappa.Set(EvaluateKappa(input[E].GetScalar(i), _kappa.GetScalar(i)).first, i);
    }


private:
    Eigen::MatrixXd _C;
    DamageLawExponential _omega;
    ModMisesEeq _strain_norm;

    // history values
    QValues _kappa;
};


class IpLoop
{
public:
    IpLoop(GradientDamage& law)
        : _law(law)
    {
        _outputs.resize(Q::LAST);
        _inputs.resize(Q::LAST);
        _law.DefineOutputs(_outputs);
        _law.DefineInputs(_inputs);
    }

    virtual void Resize(int n)
    {
        _n = n;
        for (auto& qvalues : _outputs)
            qvalues.Resize(n);
        _law.Resize(n);
    }

    Eigen::VectorXd Get(Q what)
    {
        return _outputs.at(what).data;
    }

    std::vector<Q> RequiredInputs() const
    {
        std::vector<Q> required;
        for (int iQ = 0; iQ < _inputs.size(); ++iQ)
        {
            Q q = static_cast<Q>(iQ);
            if (_inputs[q].IsUsed())
                required.push_back(q);
        }
        return required;
    }

    virtual void Evaluate(const Eigen::VectorXd& all_strains, const Eigen::VectorXd& all_neeq)
    {
        _inputs[E].data = all_neeq;
        _inputs[EPS].data = all_strains;
        for (int i = 0; i < _n; ++i)
            _law.Evaluate(_inputs, _outputs, i);
    }

    virtual void Update(const Eigen::VectorXd& all_strains, const Eigen::VectorXd& all_neeq)
    {
        _inputs[E].data = all_neeq;
        _inputs[EPS].data = all_strains;
        for (int i = 0; i < _n; ++i)
            _law.Update(_inputs, i);
    }

    GradientDamage& _law;
    std::vector<QValues> _outputs;
    std::vector<QValues> _inputs;
    int _n = 0;
};

