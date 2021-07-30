#pragma once
#include "interfaces.h"
#include <tuple>


//Plasticity with isotropic hardening
class PerfectPlasticity : public LawInterface
{
public:
    std::shared_ptr<YieldFunction> _f;
    //std::shared_ptr<HardeningLaw> _h;
    std::shared_ptr<FlowRule> _g;
    Eigen _kappa;
    Eigen::MatrixXd _C;
    bool _total_strains;
    bool _tangent;
    
    Plasticity(EigenMatrixXd& C, std::shared_ptr<YieldFunction> f, std::shared_ptr<FlowRule> g, std::shared_ptr<IsotropicHardeningLaw> h, bool total_strains = true, bool tangent = true)
    : _f(f), _h(h), _g(g), _total_strains(total_strains), kappa(kappa), _tangent(tangent)
    {
    }
    
    void DefineOutputs(std::vector<QValues>& output) const override
    {
        output[SIGMA] = QValues(6);
        if tangent
            output[DSIGMA_DEPS] = QValues(6,6);
    }
    
    void DefineInputs(std::vector<QValues>& input) const override
    {
        //depending on total_strains, EPS will either be interpreted as a strain increment or the total strains
        input[EPS] = QValues(6);
    }
    
    void Evaluate(const std::vector<QValues>& input, std::vector<QValues>& out, int i) override
    {
        // Stress return algo
    }
    
    void Update(const std::vector<QValues>& input, int i) override
    {
    
    }
    
    virtual void Resize(int n)
    {
    
    }

};



struct YieldFunction
{
    bool _associative;
    virtual void Evaluate(Eigen::VectorXd& sigma, Eigen::VectorXd& kappa);
    virtual void Evaluate(Eigen::VectorXd& sigma);
    
    //Get will return the i-th derivative 
    //of the yield function with respect to 
    //the stress. i=0 means the pure function
    //evaluation. In associative plasticity
    //we need up to i=2.
    Eigen::MatrixXd Get(int i) = 0;
};



