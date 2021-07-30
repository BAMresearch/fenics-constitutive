#pragma once
#include "interfaces.h"
#include <tuple>



class Plasticity : public LawInterface
{
public:
    std::shared_ptr<YieldFunction> f;
    std::shared_ptr<HardeningLaw> h;
    std::shared_ptr<FlowRule> g;
    std::shared_ptr<HistoryVariable> kappa;
    bool total_strains;
    bool tangent;
    
    Plasticity(std::shared_ptr<YieldFunction> f, std::shared_ptr<HistoryVariable> kappa, std::shared_ptr<HardeningLaw> h, std::shared_ptr<FlowRule> g, bool total_strains = true, bool tangent = true)
    : f(f), h(h), g(g), total_strains(total_strains), kappa(kappa), tangent(tangent)
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

class YieldFunction
{
    virtual void Evaluate(Eigen::VectorXd)
}

