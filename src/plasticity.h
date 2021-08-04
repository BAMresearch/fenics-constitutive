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
    std::shared_ptr<IsotropicHardeningLaw> _p;
    std::shared_ptr<FlowRule> _g;
    std::vector<QValues> _internal_vars_0;
    std::vector<QValues> _internal_vars_1;
    Eigen::MatrixXd _C;
    bool _total_strains;
    bool _tangent;
    
    Plasticity(EigenMatrixXd& C, std::shared_ptr<YieldFunction> f, std::shared_ptr<FlowRule> g, std::shared_ptr<IsotropicHardeningLaw> p, bool total_strains = true, bool tangent = true)
    : _f(f),
    _p(p),
    _g(g),
    _total_strains(total_strains),
    _tangent(tangent),
    {

        _internal_vars_0[KAPPA] = QValues(1);
        _internal_vars_1[KAPPA] = QValues(1);
        if (_total_strains)
        {
            _internal_vars_0[EPS_P] = QValues(6);
            _internal_vars_1[EPS_P] = QValues(6);
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
        auto plastic_strain = _internal_vars_0[EPS_P].Get(i);
        if (_total_strains) {
            auto sigma_tr = _C * (strain - plastic_strain);
        } else {
            auto sigma_tr = input[SIGMA].Get(i) + _C * strain;
        }
        Eigen::MatrixXd newton_matrix();
        Eigen::VectorXd res();


        
        if (_tangent)
            out[DSIGMA_DEPS].Set(_C, i);
    }
    
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> NewtonSystem(Eigen::VectorXd sigma, double kappa, double del_lam)
    {
        size = _C.rows();
        auto r_sig = sigma - sig_tr + del_lam * _C *  
        auto D11 = Eigen::MatrixXd::Identity(size,size) + lambda * _C *
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
    virtual void Evaluate(Eigen::VectorXd sigma, double kappa);
    virtual void Evaluate(Eigen::VectorXd sigma);
    
    //Get will return the i-th derivative 
    //of the yield function with respect to 
    //the stress. i=0 means the pure function
    //evaluation. In associative plasticity
    //we need up to i=2.
    Eigen::MatrixXd Get(int i) = 0;
};

class MisesYieldFunction
{
public:
    int _n;
    double _sig_0;
    double _H;
    double _f;
    Eigen::VectorXd _df_dsig;
    Eigen::MatrixXd _ddf_dsig;

    MisesYieldFunction(n_derivatives = 3, double sig_0, double H)
    : _n(n_derivatives),
      _sig_0(sig_0),
      _H(H)
    {
    }

    void Evaluate(Eigen::VectorXd sigma, double kappa) override
    {
        auto sig_dev = T_dev * sigma;
        auto sig_eq = std::sqrt(1.5 * sigma_dev.dot(sigma_dev));
        _f = sig_eq - _sig_0 - _H*kappa;
        _df_dsig = (1.5 / sig_eq) * sig_dev; //actually a row vector. Use .transpose()
        _ddf_dsig =  1.5 * (T_dev /sig_eq - sig_dev * _df_dsig.transpose()/ (sig_eq*sig_eq));
    }

    //virtual void Evaluate(Eigen::VectorXd sigma);
    
    //Get will return the i-th derivative 
    //of the yield function with respect to 
    //the stress. i=0 means the pure function
    //evaluation. In associative plasticity
    //we need up to i=2.
    //std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd> Get() override
    //{
    //    return {_f, _df_dsig, _ddf_dsig};
    //}
};

struct FlowRule
{
    bool _associative;
    virtual void Evaluate(Eigen::VectorXd sigma, double kappa);
    virtual void Evaluate(Eigen::VectorXd sigma);
    
    //Get will return the i-th derivative
    //of the flow rule with respect to 
    //the stress. i=0 means the pure function
    //evaluation. 
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
    virtual std::tuple<double, Eigen::VectorXd, double> Evaluate(Eigen::VectorXd sigma, double kappa) = 0;
};

class StrainHardening : public IsotropicHardeningLaw
{
public:
    StrainHardening()
    {
    
    }

    std::tuple<double, Eigen::VectorXd, double> Evaluate(Eigen::VectorXd sigma, double kappa) override
    {
        /*returns
         * double p(sig, kappa)
         * vector dp_dsig(sig, kappa)
         * double dp_dkappa(sig, kappa)
         * */
        return {1., MatrixXd::Zero(6,1), 0.};
    }

}
