#pragma once

#include <eigen3/Eigen/Core>

namespace constitutive
{

struct Dim
{
    static constexpr int Mandel(int d)
    {
        return d == 1 ? 1 : (d == 2 ? 3 : 6);
    }

    static constexpr int MandelX(int d)
    {
        return d == 1 ? 3 : (d == 2 ? 4 : 6);
    }
};


template <int TDim>
class Mandel : public Eigen::Matrix<double, Dim::Mandel(TDim), 1>
{
    using Parent = Eigen::Matrix<double, Dim::Mandel(TDim), 1>;

public:
    Mandel(const Parent& other)
        : Parent(other)
    {
    }
};

template <int TDim>
class MandelX : public Eigen::Matrix<double, Dim::MandelX(TDim), 1>
{
    using Parent = Eigen::Matrix<double, Dim::MandelX(TDim), 1>;

public:
    MandelX(const Parent& other)
        : Parent(other)
    {
    }

    MandelX(const Mandel<TDim>& other);
};

template <>
inline MandelX<1>::MandelX(const Mandel<1>& other)
{
    (*this)[0] = other[0];
    (*this)[1] = 0;
    (*this)[2] = 0;
}

template <>
inline MandelX<2>::MandelX(const Mandel<2>& other)
{
    (*this)[0] = other[0];
    (*this)[1] = other[1];
    (*this)[2] = 0;
    (*this)[3] = other[2];
}

template <>
inline MandelX<3>::MandelX(const Mandel<3>& other)
{
    (*this) = other;
}

Eigen::Vector3d timesTwo(Eigen::Vector3d a)
{
    return a * 2.;
}
} // namespace constitutive
