#pragma once
#include <eigen3/Eigen/Dense>
#include <exception>
#include <vector>
#include <numeric>
#include <memory>
#include <iostream>
#include <map>

enum Constraint
{
    UNIAXIAL_STRAIN,
    UNIAXIAL_STRESS,
    PLANE_STRAIN,
    PLANE_STRESS,
    FULL
};
enum UpdatePolicy
{
    OVERRIDE,
    KEEP
};
enum Q
{
    EPS,
    E,
    SIGMA,
    DSIGMA_DEPS,
    DSIGMA_DE,
    GRAD_V,
    LAMBDA,
    EEQ,
    DEEQ,
    KAPPA,
    NONLOCAL,
    RHO,
    DAMAGE,
    PRESSURE,
    LAST
};

std::map<std::string, Q> string_to_Q = {{"eps", Q::EPS},{"sigma", Q::SIGMA}, {"dsigma_deps",Q::DSIGMA_DEPS}};


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
            return 4;
        if (c == PLANE_STRESS)
            return 4;
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

template <Constraint TC>
using FullTensor = Eigen::Matrix<double, Dim::G(TC),Dim::G(TC), Eigen::RowMajor>;

class RefLawInterface
{
public:
    int _n;
    RefLawInterface(int n):_n(n){}
    virtual std::vector<Q> DefineInputs() const = 0;

    virtual void EvaluateIP(int i, std::vector<Eigen::Ref<Eigen::VectorXd>>& inputs, double del_t) = 0;
    void EvaluateAll(
            std::vector<Eigen::Ref<Eigen::VectorXd>>& input,
            double del_t)
    {
        for(int i=0;i<_n;i++){
            EvaluateIP(i, input, del_t);
        }
    }
    void EvaluateSome(
            std::vector<Eigen::Ref<Eigen::VectorXd>>& input,
            Eigen::Ref<Eigen::VectorXi> indices,
            double del_t)
    {
        for(int i=0;i<indices.size();i++){
            EvaluateIP(indices[i], input, del_t);
        }
    }
};


class IPLawInterface
{
public:
    int _n;
    IPLawInterface(std::map<std::string, double> &parameters, int n)
    :_n(n)
    {
        SetParameters(parameters);
    }

    virtual bool EvaluationWithTangent() = 0; 

    virtual void SetParameters(std::map<std::string, double> &parameters) = 0;

    virtual std::map<Q, std::pair<int,int>> DefineInputs() const = 0;
    virtual std::map<Q, std::pair<int,int>> DefineInternalVariables() const = 0;
    virtual std::map<Q, std::pair<int,int>> DefineFormVariables() const = 0;


    virtual void EvaluateIP(
        int i,
        std::vector<Eigen::Ref<const Eigen::VectorXd>>& constant_inputs,
        std::vector<Eigen::Ref<Eigen::VectorXd>>& form_variables,
        std::vector<Eigen::Ref<const Eigen::VectorXd>>& internal_variables_0,
        std::vector<Eigen::Ref<Eigen::VectorXd>>& internal_variables_1,
        double del_t) = 0;
    
    void Evaluate(
        std::vector<Eigen::Ref<const Eigen::VectorXd>>& constant_inputs,
        std::vector<Eigen::Ref<Eigen::VectorXd>>& weak_form_variables,
        std::vector<Eigen::Ref<const Eigen::VectorXd>>& internal_variables_0,
        std::vector<Eigen::Ref<Eigen::VectorXd>>& internal_variables_1,
        double del_t)
    {
        for(int i=0;i<_n;i++){
            EvaluateIP(i, constant_inputs, weak_form_variables, internal_variables_0, internal_variables_1, del_t);
        }
    }
    // void EvaluateWithTangent(
    //     std::vector<Eigen::Ref<Eigen::VectorXd>>& constant_inputs,
    //     std::vector<Eigen::Ref<Eigen::VectorXd>>& weak_form_variables,
    //     std::vector<Eigen::Ref<Eigen::VectorXd>>& internal_variables_0,
    //     std::vector<Eigen::Ref<Eigen::VectorXd>>& internal_variables_1,
    //     double del_t)
    // {
    //     if (!EvaluationWithTangent())
    //         throw std::runtime_error("Evaluation with tangent is not implemented for this law.");
    //     if (weak_form_variables[Q::DSIGMA_DEPS].size() == 0)
    //         throw std::runtime_error("Please supply a vector for the tangent");


    //     for(int i=0;i<_n;i++){
    //         EvaluateIP(i, constant_inputs, weak_form_variables, internal_variables_0, internal_variables_1, del_t, true);
    //     }
    // }
};



struct ScalarVariableUpdaterInterface
{
    virtual inline void Set(int i, double value, Eigen::Ref<Eigen::VectorXd> internal_0, Eigen::Ref<Eigen::VectorXd> internal_1) = 0;
    virtual inline void Add(int i, double value, Eigen::Ref<Eigen::VectorXd> internal_0, Eigen::Ref<Eigen::VectorXd> internal_1) = 0;
    virtual inline void Multiply(int i, double value, Eigen::Ref<Eigen::VectorXd> internal_0, Eigen::Ref<Eigen::VectorXd> internal_1) = 0;
};

template <UpdatePolicy P> class ScalarVariableUpdater : public ScalarVariableUpdaterInterface
{

};

template<> 
class ScalarVariableUpdater<OVERRIDE> : public ScalarVariableUpdaterInterface
{
    inline void Set(int i, double value, Eigen::Ref<Eigen::VectorXd> internal_0, Eigen::Ref<Eigen::VectorXd> internal_1) override
    {
        internal_0(i) = value;
    }
    inline void Add(int i, double value, Eigen::Ref<Eigen::VectorXd> internal_0, Eigen::Ref<Eigen::VectorXd> internal_1) override
    {
        internal_0(i) += value;
    }
    inline void Multiply(int i, double value, Eigen::Ref<Eigen::VectorXd> internal_0, Eigen::Ref<Eigen::VectorXd> internal_1) override
    {
        internal_0(i) *= value;
    }

};
template<> 
class ScalarVariableUpdater<KEEP> : public ScalarVariableUpdaterInterface
{
    inline void Set(int i, double value, Eigen::Ref<Eigen::VectorXd> internal_0, Eigen::Ref<Eigen::VectorXd> internal_1) override
    {
        internal_1(i) = value;
    }
    inline void Add(int i, double value, Eigen::Ref<Eigen::VectorXd> internal_0, Eigen::Ref<Eigen::VectorXd> internal_1) override
    {
        internal_1(i) += value;
    }
    inline void Multiply(int i, double value, Eigen::Ref<Eigen::VectorXd> internal_0, Eigen::Ref<Eigen::VectorXd> internal_1) override
    {
        internal_1(i) *= value;
    }

};