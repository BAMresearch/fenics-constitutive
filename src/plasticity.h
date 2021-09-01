#pragma once
#include "interfaces.h"
#include <tuple>
#include <eigen3/Eigen/Dense>

//Plasticity with isotropic hardening

struct YieldFunction
{
    virtual std::tuple<double, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, double, Eigen::VectorXd> Evaluate(Eigen::VectorXd& sigma, double kappa) = 0;
};
struct IsotropicHardeningLaw
{
    virtual std::tuple<double, Eigen::VectorXd, double> Evaluate(Eigen::VectorXd& sigma, double kappa) = 0;
};


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
    int _stress_dim;
    
    IsotropicHardeningPlasticity(Eigen::MatrixXd& C, std::shared_ptr<YieldFunction>& f, std::shared_ptr<IsotropicHardeningLaw>& p, bool total_strains = true, bool tangent = true)
    : _f(f),
    _p(p),
    _total_strains(total_strains),
    _tangent(tangent)
    {
        _stress_dim = _C.rows();

        _internal_vars_0[KAPPA] = QValues(1);
        _internal_vars_0[LAMBDA] = QValues(1);

        _internal_vars_1[KAPPA] = QValues(1);
        _internal_vars_1[LAMBDA] = QValues(1);
        
        if (_total_strains)
        {
            //For the total strain formulation, we need the full plastic strain tensor to calculate the 
            //trial stress.
            _internal_vars_0[EPS_P] = QValues(_stress_dim);
            _internal_vars_1[EPS_P] = QValues(_stress_dim);
        }
    }
    
    void DefineOutputs(std::vector<QValues>& output) const override
    {
        output[SIGMA] = QValues(_stress_dim);
        if (_tangent)
            //return the consistent tangent if required by the user
            output[DSIGMA_DEPS] = QValues(_stress_dim,_stress_dim);
    }
    
    void DefineInputs(std::vector<QValues>& input) const override
    {
        //depending on total_strains, EPS will either be interpreted as a strain increment or the total strains
        input[EPS] = QValues(_stress_dim);
        if (!_total_strains)
            //If EPS is a strain increment, it is assumed that the stress may be transformed outside
            //the constitutive law (e.g. rotation with an objective stress rate). Therefore it is used
            //as an input instead of an iunternal variable
            input[SIGMA] = QValues(_stress_dim);
    }
    
    void Evaluate(const std::vector<QValues>& input, std::vector<QValues>& output, int i) override
    {
        auto maxit = 100;
        auto strain = input[EPS].Get(i);
        auto kappa = _internal_vars_0[KAPPA].Get(i);

        Eigen::VectorXd sigma_tr(_stress_dim,1);
        if (_total_strains) {
            auto plastic_strain = _internal_vars_0[EPS_P].Get(i);
            sigma_tr = _C * (strain - plastic_strain);
        } else {
            sigma_tr = input[SIGMA].Get(i) + _C * strain;
        }
        //Eigen::VectorXd res(8);
        //Eigen::MatrixXd jacobian(8,8);´
        auto [res,jacobian] = NewtonSystem(sigma_tr, sigma_tr, _internal_vars_0[KAPPA].GetScalar(i), 0., 0.);
        Eigen::VectorXd x(_stress_dim+2);
        x << sigma_tr, kappa, 0;
        
        int j = 0;
        while (res.norm() > 1e-10 && j < maxit) {
            x = x - jacobian.lu().solve(res);
            std::tie(res, jacobian) = NewtonSystem(x.segment(0,_stress_dim), sigma_tr, x[_stress_dim], x[_stress_dim+1], 0);
            j++;
        }

        output[SIGMA].Set(x.segment(0,_stress_dim),i);
        _internal_vars_1[KAPPA].Set(x[_stress_dim],i);
        _internal_vars_1[LAMBDA].Set(_internal_vars_0[LAMBDA].GetScalar(i)+x[_stress_dim+1], i);
        
        if (_tangent) {
            //TODO
            output[DSIGMA_DEPS].Set(_C, i);
        }
    }
    
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> NewtonSystem(Eigen::VectorXd sigma, Eigen::VectorXd sigma_tr, double kappa, double del_lam, double kappa_0)
    {
        //get yield function, flow rule and their derivatives from the yieldfuncion
        //get hardening rule and derivatives
        auto [f, g, df_dsig, dg_dsig, df_dkappa, dg_dkappa] = _f->Evaluate(sigma, kappa);
        auto [p, dp_dsig, dp_dkappa] = _p->Evaluate(sigma, kappa);
        
        //set elements of residual vector
        auto r_sig = sigma - sigma_tr + del_lam * _C * g;
        auto r_kappa = kappa - kappa_0 - del_lam * p;
        auto r_f = f;
        
        //set blocks of jacobian matrix
        auto J11 = Eigen::MatrixXd::Identity(_stress_dim,_stress_dim) + del_lam * _C * dg_dsig; // \in \R^{6,6}
        auto J12 = del_lam * _C * dg_dkappa; // \in \R^{6,1}
        auto J13 = _C * g; // \in \R^{6,1}

        auto J21 = -del_lam * dp_dsig.transpose(); // \in \R^{1,6}
        auto J22 = 1 - del_lam * dp_dkappa; // \in \R^{1}
        auto J23 = -p; // \in \R^{1}

        auto J31 = df_dsig.transpose(); // \in \R^{1,6}
        auto J32 = df_dkappa; // \in \R^{1}
        auto J33 = 0; // \in \R^{1}
        
        Eigen::VectorXd res(_stress_dim+2);
        Eigen::MatrixXd jacobian(_stress_dim+2, _stress_dim+2);
        res << r_sig, r_kappa, r_f;
        jacobian << J11, J12, J13,
                    J21, J22, J23,
                    J31, J32, J33;
        return {res, jacobian};

    }

    void Update(const std::vector<QValues>& input, int i) override
    {
        _internal_vars_0[KAPPA].Set(_internal_vars_1[KAPPA].Get(i), i);
        _internal_vars_0[LAMBDA].Set(_internal_vars_1[LAMBDA].Get(i), i);

        if (_total_strains)
            _internal_vars_0[EPS_P].Set(_internal_vars_1[EPS_P].Get(i), i);
        
    }
    
    virtual void Resize(int n)
    {
        for (auto& qvalues : _internal_vars_0)
            qvalues.Resize(n);

        for (auto& qvalues : _internal_vars_1)
            qvalues.Resize(n);
    }

};



class MisesYieldFunction : public YieldFunction
{
public:
    double _sig_0;
    double _H;
    double _f;
    Eigen::VectorXd _df_dsig;
    Eigen::VectorXd _g;
    Eigen::MatrixXd _dg_dsig;
    double _df_dkappa;
    Eigen::VectorXd _dg_dkappa;
    
    Eigen::MatrixXd T_dev;
    Eigen::VectorXd T_vol;
    
    MisesYieldFunction(double sig_0, double H)
    : _sig_0(sig_0),
      _H(H)
    {
        T_dev.resize(6,6);
        T_vol.resize(6);
        T_dev << 
                2./3., -1./3., -1./3., 0., 0., 0.,
                -1./3., 2./3., -1./3., 0., 0., 0.,
                -1./3., -1./3., 2./3., 0., 0., 0.,
                0., 0., 0., 1., 0., 0., 
                0., 0., 0., 0., 1., 0., 
                0., 0., 0., 0., 0., 1.;
        T_vol << 1,1,1,0,0,0;
    }

    std::tuple<double, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, double, Eigen::VectorXd> Evaluate(Eigen::VectorXd& sigma, double kappa) override
    {
        auto sig_dev = T_dev * sigma;
        auto sig_eq = std::sqrt(1.5 * sig_dev.dot(sig_dev));
        _f = sig_eq - _sig_0 - _H*kappa;
        _df_dsig = (1.5 / sig_eq) * sig_dev; //actually a row vector. Use .transpose()
        _g = _df_dsig;
        _dg_dsig =  1.5 * (T_dev /sig_eq - sig_dev * _g.transpose()/ (sig_eq*sig_eq));

        _df_dkappa = - _H;
        _dg_dkappa = _g * 0;
        return {_f, _g, _df_dsig, _dg_dsig, _df_dkappa, _dg_dkappa};//return {_f, _df_dsig, _ddf_dsig, _df_dkappa};
    }

};


class StrainHardening : public IsotropicHardeningLaw
{
public:
    StrainHardening()
    {
    
    }

    std::tuple<double, Eigen::VectorXd, double> Evaluate(Eigen::VectorXd& sigma, double kappa) override
    {
        /*returns
         * double p(sig, kappa)
         * vector dp_dsig(sig, kappa)
         * double dp_dkappa(sig, kappa)
         * */
        return {1., Eigen::MatrixXd::Zero(6,1), 0.};
    }
};



