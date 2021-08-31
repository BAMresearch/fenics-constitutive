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
    std::vector<QValues> _internal_vars_0;
    std::vector<QValues> _internal_vars_1;
    Eigen::MatrixXd _C;
    bool _total_strains;
    bool _tangent;
    
    IsotropicHardeningAssociativePlasticity(EigenMatrixXd& C, std::shared_ptr<YieldFunction> f,  std::shared_ptr<IsotropicHardeningLaw> p, bool total_strains = true, bool tangent = true)
    : _f(f),
    _p(p),
    _total_strains(total_strains),
    _tangent(tangent),
    {

        _internal_vars_0[KAPPA] = QValues(1);
        _internal_vars_0[LAMBDA] = QValues(1);

        _internal_vars_1[KAPPA] = QValues(1);
        _internal_vars_1[LAMBDA] = QValues(1);
        
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
        if (_total_strains) {
            auto plastic_strain = _internal_vars_0[EPS_P].Get(i);
            auto sigma_tr = _C * (strain - plastic_strain);
        } else {
            auto sigma_tr = input[SIGMA].Get(i) + _C * strain;
        }
        
        auto [res,jacobian] = NewtonSystem(sigma_tr, _internal_vars_0[KAPPA].GetScalar(i), 0, 0);
        Eigen::VectorXd x(size+2);
        x << sigma_tr, kappa, 0;

        while (res.norm() > 1e-10) {
            x = x - jacobian.partialPivLu().solve(res);
            [res, jacobian] = NewtonSystem(x.segment<6>(0), x[6], x[7], 0);
        }
        out[SIGMA].Set(x.segment<6>(0),i);
        _internal_vars_1[KAPPA].Set(x[6],i);
        _internal_vars_1[LAMBDA].Set(_internal_vars_0[LAMBDA].GetScalar(i)+x[7], i);
        
        if (_tangent) {
            //TODO
            out[DSIGMA_DEPS].Set(_C, i);
        }
    }
    
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> NewtonSystem(Eigen::VectorXd sigma, Eigen::VectorXd sigma_tr, double kappa, double del_lam, double kappa_0)
    {
        //get yield function, flow rule and their derivatives from the yieldfuncion
        //get hardening rule and derivatives
        auto [f, g, df_dsig, dg_dsig, df_dkappa, dg_dkappa] = _f.Evaluate(sigma, kappa);
        auto [p, dp_dsig, dp_dkappa] = _p.Evaluate(sigma, kappa);
        auto size = sigma.size();
        
        //set elements of residual vector
        auto r_sig = sigma - sigma_tr + del_lam * _C * g;
        auto r_kappa = kappa - kappa_0 - del_lam * p;
        auto r_f = f;
        
        //set blocks of jacobian matrix
        auto J11 = Eigen::MatrixXd::Identity(size,size) + lambda * _C * dg_dsig; // \in \R^{6,6}
        auto J12 = del_lam * _C * dg_dkappa; // \in \R^{6,1}
        auto J13 = _C * g; // \in \R^{6,1}

        auto J21 = -del_lam * dp_dsig.transpose(); // \in \R^{1,6}
        auto J22 = 1 - del_lam * dp_dkappa; // \in \R^{1}
        auto J23 = -p; // \in \R^{1}

        auto J31 = df_dsig.transpose(); // \in \R^{1,6}
        auto J32 = df_dkappa; // \in \R^{1}
        auto J33 = 0; // \in \R^{1}
        
        Eigen::VectorXd res(size+2)
        Eigen::MatrixXd jacobian(size+2, size+2);
        res << r_sig, r_kappa, r_f;
        jacobian << J11, J12, J13,
                    J21, J22, J23,
                    J31, J32, J33;
        return {res, jacobian};

    }

    void Update(const std::vector<QValues>& input, int i) override
    {
        //_kappa_0.Set(_kappa_1.Get(i), i);
    }
    
    virtual void Resize(int n)
    {
        _internal_vars_0.Resize(n);
        _internal_vars_1.Resize(n);
    }

};


struct YieldFunction
{
    virtual void Evaluate(Eigen::VectorXd sigma, double kappa) = 0;
};

class MisesYieldFunction : public YieldFunction
{
public:
    int _n;
    double _sig_0;
    double _H;
    double _f;
    Eigen::VectorXd _df_dsig;
    Eigen::VectorXd _g;
    Eigen::MatrixXd _dg_dsig;
    double _df_dkappa;
    Eigen::VectorXd _dg_dkappa;

    MisesYieldFunction(double sig_0, double H)
    : _sig_0(sig_0),
      _H(H)
    {
    }

    std::tuple<double, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, double, Eigen::VectorXd> Evaluate(Eigen::VectorXd sigma, double kappa) override
    {
        auto sig_dev = T_dev * sigma;
        auto sig_eq = std::sqrt(1.5 * sigma_dev.dot(sigma_dev));
        _f = sig_eq - _sig_0 - _H*kappa;
        _df_dsig = (1.5 / sig_eq) * sig_dev; //actually a row vector. Use .transpose()
        _g = _df_dsig;
        _dg_dsig =  1.5 * (T_dev /sig_eq - sig_dev * _g.transpose()/ (sig_eq*sig_eq));

        _df_dkappa = - _H;
        _dg_dkappa = _g * 0;
        return {_f, _g, _df_dsig, _dg_dsig, _df_dkappa, _dg_dkappa};//return {_f, _df_dsig, _ddf_dsig, _df_dkappa};
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


