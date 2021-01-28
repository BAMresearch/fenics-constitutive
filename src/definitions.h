#include <eigen3/Eigen/Core>

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

