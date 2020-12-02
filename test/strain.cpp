#include <eigen3/Eigen/Core>
#include <iostream>
#include <tuple>

class ModifiedMisesStrainNorm
{
public:
    ModifiedMisesStrainNorm(double nu, double k)
        : K1((k - 1.) / (2. * k * (1 - 2 * nu)))
        , K2(3. / (k * (1. + nu) * (1. + nu)))
    {
    }

    double Value(double I1, double J2)
    {
        const double A = std::sqrt(K1 * K1 * I1 * I1 + K2 * J2);
        return K1 * I1 + A;
    }

    std::tuple<double, double, double> Eval(double I1, double J2)
    {
        const double A = std::sqrt(K1 * K1 * I1 * I1 + K2 * J2) + 1.e-14;
        const double value = K1 * I1 + A;
        const double dI1 = K1 + K1 * K1 * I1 / A;
        const double dJ2 = K2 / (2 * A);
        return {A, dI1, dJ2};
    }

private:
    double K1, K2;
};


class EngineeringStrain3D : public Eigen::Matrix<double, 6, 1>
{
    using Parent = Eigen::Matrix<double, 6, 1>;

public:
    EngineeringStrain3D(Parent other)
        : Parent(other)
    {
    }
};


class EngineeringStrain2D : public Eigen::Vector3d
{
public:
    EngineeringStrain2D(Eigen::Vector3d other)
        : Eigen::Vector3d(other)
    {
    }
};

struct Transform3D
{
    static Eigen::Matrix<double, 6, 3> From2D()
    {
        Eigen::Matrix<double, 6, 3> T;
        T(0, 0) = 1;
        T(1, 1) = 1;
        T(5, 2) = 1;
        return T;
    }
};

int main(int argc, char* argv[])
{
    EngineeringStrain2D v({1, 2, 3});
    auto T = Transform3D::From2D();

    Eigen::Matrix<double, 6, 1> vec3D = T * v;
    EngineeringStrain3D v3D = vec3D;

    return 0;
}
