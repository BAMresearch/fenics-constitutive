#pragma once
#include "interfaces.h"
#include <tuple>

Eigen::MatrixXd T_dev(6,6);
T_dev << 
        2/3, -1/3, -1/3, 0, 0, 0,
        -1/3, 2/3, -1/3, 0, 0, 0,
        -1/3, -1/3, 2/3, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 
        0, 0, 0, 0, 1, 0, 
        0, 0, 0, 0, 0, 1;
Eigen::VectorXd T_tr(6);
T_vol << 1,1,1,0,0,0;
//Plasticity with isotropic hardening
class IsotropicHardeningPlasticity : public LawInterface
{
public:
    std::shared_ptr<YieldFunction> _f;
    std::shared_ptr<IsotropicHardeningLaw> _h;
    std::shared_ptr<FlowRule> _g;
    std::vector<QValues> _kappa_0;
    std::vector<QValues> _kappa_1;
    Eigen::MatrixXd _C;
    bool _total_strains;
    bool _tangent;
    
    Plasticity(EigenMatrixXd& C, std::shared_ptr<YieldFunction> f, std::shared_ptr<FlowRule> g, std::shared_ptr<IsotropicHardeningLaw> h, bool total_strains = true, bool tangent = true)
    : _f(f),
    _h(h),
    _g(g),
    _total_strains(total_strains),
    _tangent(tangent),
    {

        _kappa_0[LAMBDA] = QValues(1);
        _kappa_1[LAMBDA] = QValues(1);
        if (_total_strains)
        {
            _kappa_0[EPS_P] = QValues(6);
            _kappa_1[EPS_P] = QValues(6);
        }
    }
    
    void DefineOutputs(std::vector<QValues>& output) const override
    {
        output[SIGMA] = QValues(6);
        if (tangent)
            output[DSIGMA_DEPS] = QValues(6,6);
    }
    
    void DefineInputs(std::vector<QValues>& input) const override
    {
        //depending on total_strains, EPS will either be interpreted as a strain increment or the total strains
        input[EPS] = QValues(6);
        if (!total_strains)
            input[SIGMA] = QValues(6)
    }
    
    void Evaluate(const std::vector<QValues>& input, std::vector<QValues>& out, int i) override
    {
        auto strain = input[EPS].Get(i);
        auto plastic_strain = _kappa_0[EPS_P].Get(i);
        if (_total_strains) {
            auto sigma_tr = _C * (strain-plastic_strain);
        } else {
            auto sigma_tr = input[SIGMA].Get(i) + _C * strain;
        }
        Eigen::MatrixXd newton_matrix();
        Eigen::VectorXd res();


        
        if (_tangent)
            out[DSIGMA_DEPS].Set(_C, i);
    }
    
    void Update(const std::vector<QValues>& input, int i) override
    {
        //_kappa_0.Set(_kappa_1.Get(i), i);
    }
    
    virtual void Resize(int n)
    {
        _kappa_0.Resize(n);
        _kappa_1.Resize(n);
    }

};

struct NewtonFunction
{
    
}

struct YieldFunction
{
    virtual void Evaluate(Eigen::VectorXd& sigma, Eigen::VectorXd& kappa);
    virtual void Evaluate(Eigen::VectorXd& sigma);
    
    //Get will return the i-th derivative 
    //of the yield function with respect to 
    //the stress. i=0 means the pure function
    //evaluation. In associative plasticity
    //we need up to i=2.
    Eigen::MatrixXd Get(int i) = 0;
};

class MisesYieldFunction
{

    int _n;
    MisesYieldFunction(n_derivatives = 3)
    : _n(n_derivatives)
    {
    }
    virtual void Evaluate(Eigen::VectorXd& sigma, Eigen::VectorXd& kappa)
    {
        auto sigma_dev = T_dev * sigma;
        auto f = sigma_dev.dot(sigma_dev);
    }
    virtual void Evaluate(Eigen::VectorXd& sigma);
    
    //Get will return the i-th derivative 
    //of the yield function with respect to 
    //the stress. i=0 means the pure function
    //evaluation. In associative plasticity
    //we need up to i=2.
    Eigen::MatrixXd Get(int i) = 0;
};
struct FlowRule
{
    bool _associative;
    virtual void Evaluate(Eigen::VectorXd& sigma, Eigen::VectorXd& kappa);
    virtual void Evaluate(Eigen::VectorXd& sigma);
    
    //Get will return the i-th derivative
    //of the flow rule with respect to 
    //the stress. i=0 means the pure function
    //evaluation. In associative plasticity
    //we need up to i=2.
    Eigen::MatrixXd Get(int i) = 0;
};

class AssociativeFlow : public FlowRule
{
//This class does nothing except to tell the plasticity class to onöly use the derivative of
//the yield function as a flow rule
public:
    AssociativeFlow()
    {
        _associative = true;
    }
};

struct IsotropicHardeningLaw
{
    double Evaluate(Eigen::VectorXd kappa) = 0;
};

