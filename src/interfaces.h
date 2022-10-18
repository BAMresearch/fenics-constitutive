#pragma once
#include <eigen3/Eigen/Core>
#include <exception>
#include <vector>
#include <numeric>
#include <memory>
#include <iostream>

enum Constraint
{
    UNIAXIAL_STRAIN,
    UNIAXIAL_STRESS,
    PLANE_STRAIN,
    PLANE_STRESS,
    FULL
};

enum Q
{
    EPS,
    E,
    SIGMA,
    DSIGMA_DEPS,
    DSIGMA_DE,
    EEQ,
    DEEQ,
    KAPPA,
    LAST
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

    static constexpr int StressStrain(Constraint c)
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
using MandelVector = Eigen::Matrix<double, Dim::StressStrain(TC), 1>;

template <Constraint TC>
using MandelMatrix = Eigen::Matrix<double, Dim::StressStrain(TC), Dim::StressStrain(TC), Eigen::RowMajor>;


class RefLawInterface
{
public:
    int _n;
    RefLawInterface(int n):_n(n){}
    virtual std::vector<Q> DefineInputs() const = 0;

    virtual void EvaluateIP(int i, std::vector<Eigen::Ref<Eigen::VectorXd>>& inputs, double del_t) = 0;
    virtual void UpdateIP(int i)
    {
    }
    void EvaluateAll(
            std::vector<Eigen::Ref<Eigen::VectorXd>>& input,
            double del_t)
    {
        for(int i=0;i<_n;i++){
            EvaluateIP(i, input, del_t);
        }
    }
    void UpdateAll()
    {
        for(int i=0;i<_n;i++){
            UpdateIP(i);
        }
    }
};

