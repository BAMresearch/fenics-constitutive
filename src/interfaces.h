#pragma once
#include <Eigen/Core>
#include <exception>

enum Constraint
{
    UNIAXIAL_STRAIN,
    UNIAXIAL_STRESS,
    PLANE_STRAIN,
    PLANE_STRESS,
    FULL
};

struct Dim
{
    static constexpr int G(Constraint c)
    {
        if (c == UNIAXIAL_STRAIN)
            return 1;
        if (c == UNIAXIAL_STRESS)
            return 1;
        if (c == PLANE_STRAIN)
            return 2;
        if (c == PLANE_STRESS)
            return 2;
        if (c == FULL)
            return 3;
        static_assert(true, "Constraint type not supported.");
        return -1;
    }

    static constexpr int Q(Constraint c)
    {
        if (c == UNIAXIAL_STRAIN)
            return 1;
        if (c == UNIAXIAL_STRESS)
            return 1;
        if (c == PLANE_STRAIN)
            return 3;
        if (c == PLANE_STRESS)
            return 3;
        if (c == FULL)
            return 6;
        static_assert(true, "Constraint type not supported.");
        return -1;
    }
};


template <Constraint TC>
using V = Eigen::Matrix<double, Dim::Q(TC), 1>;

template <Constraint TC>
using M = Eigen::Matrix<double, Dim::Q(TC), Dim::Q(TC)>;

template <Constraint TC>
M<TC> C(double E, double nu);

template <>
M<UNIAXIAL_STRAIN> C<UNIAXIAL_STRAIN>(double E, double nu)
{
    return M<UNIAXIAL_STRAIN>::Constant(E);
}

template <>
M<UNIAXIAL_STRESS> C<UNIAXIAL_STRESS>(double E, double nu)
{
    return M<UNIAXIAL_STRAIN>::Constant(E);
}

template <>
M<PLANE_STRESS> C<PLANE_STRESS>(double E, double nu)
{
    const double C11 = E / (1 - nu * nu);
    const double C12 = nu * C11;
    const double C33 = (1 - nu) * 0.5 * C11;
    M<PLANE_STRAIN> c;
    c << C11, C12, 0, C12, C11, 0, 0, 0, C33;
    return c;
}

template <>
M<PLANE_STRAIN> C<PLANE_STRAIN>(double E, double nu)
{
    const double l = E * nu / (1 + nu) / (1 - 2 * nu);
    const double m = E / (2.0 * (1 + nu));
    M<PLANE_STRAIN> c = V<PLANE_STRAIN>({2 * m, 2 * m, m}).asDiagonal();
    c.block<2, 2>(0, 0) += Eigen::Matrix2d::Constant(l);
    return c;
}

template <>
M<FULL> C<FULL>(double E, double nu)
{
    const double l = E * nu / (1 + nu) / (1 - 2 * nu);
    const double m = E / (2.0 * (1 + nu));

    V<FULL> diagonal;
    diagonal.segment<3>(0) = Eigen::Vector3d::Constant(2 * m);
    diagonal.segment<3>(3) = Eigen::Vector3d::Constant(m);
    M<FULL> c = diagonal.asDiagonal();
    c.block<3, 3>(0, 0) += Eigen::Matrix3d::Constant(l);
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


class IpBase
{
public:
    virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate(const Eigen::VectorXd& strain, int i = 0) = 0;

    virtual void update(const Eigen::VectorXd& strain, int i = 0)
    {
    }

    virtual int qdim() const = 0;
};

class Base
{
public:
    Base(IpBase& law)
        : _law(law)
    {
    }

    virtual void resize(int n)
    {
        _n = n;
        int q = _law.qdim();
        _stress.resize(_n * q);
        _dstress.resize(_n * q * q);
    }


    virtual void evaluate(const Eigen::VectorXd& all_strains)
    {
        int q = _law.qdim();
        const int n = all_strains.rows() / q;
        if (_stress.rows() == 0)
            resize(n);

        assert(n == _n);
        for (int i = 0; i < _n; ++i)
        {
            auto eval = _law.evaluate(all_strains.segment(i * q, q), i);
            _stress.segment(i * q, q) = eval.first;
            _dstress.segment(q * q * i, q * q) = Eigen::Map<Eigen::VectorXd>(eval.second.data(), eval.second.size());
        }
    }

    virtual void update(const Eigen::VectorXd& all_strains)
    {
        int q = _law.qdim();
        for (int i = 0; i < _n; ++i)
            _law.update(all_strains.segment(i * q, q));
    }

    IpBase& _law;
    int _n = 0;
    Eigen::VectorXd _stress;
    Eigen::VectorXd _dstress;
};

