#pragma once
#include "interfaces.h"
#include "plasticity.h"
#include <tuple>
#include <eigen3/Eigen/Dense>

class AnalyticMisesPlasticity : public LawInterface
{
public:
    std::vector<QValues> _internal_vars_0;
    std::vector<QValues> _internal_vars_1;

    double _mu;
    Eigen::MatrixXd _C;
    double _sig0;
    double _H;
    bool _total_strains;
    bool _tangent;
    int _stress_dim;
    Eigen::MatrixXd T_dev;
    Eigen::VectorXd T_vol;

    AnalyticMisesPlasticity(double E_, double nu, double sig0, double H, bool total_strains = true, bool tangent = true)
    : _sig0(sig0),
    _H(H),
    _total_strains(total_strains),
    _tangent(tangent)
    {
        const double l = E_ * nu / (1 + nu) / (1 - 2 * nu);
        _mu = E_ / (2.0 * (1 + nu));
        _C.setZero(6,6);

        _C << 2*_mu+l, l, l, 0., 0., 0.,
             l, 2*_mu+l, l, 0., 0., 0.,
             l, l, 2*_mu+l, 0., 0., 0.,
             0., 0., 0., 2*_mu, 0., 0.,
             0., 0., 0., 0., 2*_mu, 0.,
             0., 0., 0., 0., 0., 2*_mu;
        _internal_vars_0.resize(Q::LAST);
        _internal_vars_1.resize(Q::LAST);

        _stress_dim = 6;

        _internal_vars_0[LAMBDA] = QValues(1);

        _internal_vars_1[LAMBDA] = QValues(1);

        if (_total_strains)
        {
            //For the total strain formulation, we need the full plastic strain tensor to calculate the
            //trial stress.
            _internal_vars_0[SIGMA] = QValues(_stress_dim);
            _internal_vars_1[SIGMA] = QValues(_stress_dim);
            _internal_vars_0[EPS] = QValues(_stress_dim);
            _internal_vars_1[EPS] = QValues(_stress_dim);
        }
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
    Eigen::VectorXd GetInternalVar(Q which)
    {
        return _internal_vars_0.at(which).data;
    }
    void Evaluate(const std::vector<QValues>& input, std::vector<QValues>& output, int i) override
    {
        auto strain = input[EPS].Get(i);
        auto p = _internal_vars_0[LAMBDA].GetScalar(i);
        Eigen::VectorXd sigma_tr(_stress_dim);
        if (_total_strains) {
            auto sig_old = _internal_vars_0[SIGMA].Get(i);
            sigma_tr = sig_old + _C * (strain - _internal_vars_0[EPS].Get(i));
        } else {
            sigma_tr = input[SIGMA].Get(i) + _C * strain;
        }
        auto sigma_dev = T_dev * sigma_tr;

        double sigma_eq = std::sqrt(3. / 2.) * sigma_dev.norm();

        double f = sigma_eq - _sig0 - _H * p;

        if (f <= 0)
        {
            output[SIGMA].Set(sigma_tr,i);
            _internal_vars_1[SIGMA].Set(sigma_tr, i);
            if(_tangent)
              output[DSIGMA_DEPS].Set(_C,i);
        }
        else
        {
            double dp = f / (3 * _mu + _H);
            _internal_vars_1[LAMBDA].Set(p+dp, i);
            auto n_elas = sigma_dev / sigma_eq;
            double beta = (3 * _mu * dp) / sigma_eq;
            auto stress = sigma_tr - beta * sigma_dev;

            output[SIGMA].Set(stress,i);
            _internal_vars_1[SIGMA].Set(stress, i);
            if(_tangent){
                auto tangent = _C -
                          3 * _mu * (3 * _mu / (3 * _mu + _H) - beta) * n_elas *
                                  n_elas.transpose() -
                          2 * _mu * beta * T_dev;

                output[DSIGMA_DEPS].Set(tangent,i);
            }
        }

    }
    void Update(const std::vector<QValues>& input, int i) override
    {
        _internal_vars_0[EPS].Set(input[EPS].Get(i),i);
        _internal_vars_0[LAMBDA].Set(_internal_vars_1[LAMBDA].Get(i), i);

        if (_total_strains)
            _internal_vars_0[SIGMA].Set(_internal_vars_1[SIGMA].Get(i), i);

    }

    void Resize(int n) override
    {
        for (auto& qvalues : _internal_vars_0)
            qvalues.Resize(n);

        for (auto& qvalues : _internal_vars_1)
            qvalues.Resize(n);
    }
};


class RadialReturnMisesPlasticity : public LawInterface
{
public:

    std::vector<QValues> _internal_vars_0;
    std::vector<QValues> _internal_vars_1;
    Eigen::MatrixXd _C;
    Eigen::VectorXd T_vol;
    Eigen::MatrixXd T_dev;
    double _mu;
    double _H;
    double _sig0;

    RadialReturnMisesPlasticity(double E_, double nu, double sig0, double H)
    {
        _internal_vars_0.resize(Q::LAST);
        _internal_vars_1.resize(Q::LAST);

        _internal_vars_0[LAMBDA] = QValues(1);
        _internal_vars_1[LAMBDA] = QValues(1);

        _internal_vars_0[SIGMA] = QValues(6);
        _internal_vars_1[SIGMA] = QValues(6);
        _internal_vars_0[EPS] = QValues(6);

        const double l = E_ * nu / (1 + nu) / (1 - 2 * nu);
        _mu = E_ / (2.0 * (1 + nu));
        _H = H;
        _sig0 = sig0;
        _C.setZero(6,6);

        _C << 2*_mu+l, l, l, 0., 0., 0.,
             l, 2*_mu+l, l, 0., 0., 0.,
             l, l, 2*_mu+l, 0., 0., 0.,
             0., 0., 0., 2*_mu, 0., 0.,
             0., 0., 0., 0., 2*_mu, 0.,
             0., 0., 0., 0., 0., 2*_mu;

        T_dev.resize(6,6);
        T_vol.resize(6);
        T_dev <<
                2./3., -1./3., -1./3., 0., 0., 0.,
                -1./3., 2./3., -1./3., 0., 0., 0.,
                -1./3., -1./3., 2./3., 0., 0., 0.,
                0., 0., 0., 1., 0., 0.,
                0., 0., 0., 0., 1., 0.,
                0., 0., 0., 0., 0., 1.;
        T_vol << 1.,1.,1.,0.,0.,0.;


    }

    void DefineOutputs(std::vector<QValues>& output) const override
    {
        output[SIGMA] = QValues(6);
    }

    void DefineInputs(std::vector<QValues>& input) const override
    {
        input[EPS] = QValues(6);
    }
    Eigen::VectorXd GetInternalVar(Q which)
    {
        return _internal_vars_0.at(which).data;
    }

    void Evaluate(const std::vector<QValues>& input, std::vector<QValues>& output, int i) override
    {
        auto maxit = 100;

        auto strain = input[EPS].Get(i) ;
        auto lambda = _internal_vars_0[LAMBDA].GetScalar(i);
        Eigen::VectorXd sigma_tr(6);
        sigma_tr = _internal_vars_0[SIGMA].Get(i) + _C * (strain - _internal_vars_0[EPS].Get(i));
        auto s_tr = T_dev * sigma_tr;
        auto p = (1./3.) * T_vol.dot(sigma_tr);

        double sig0_0 = _sig0 + _H * lambda;
        double s_tr_eq = std::sqrt(3. / 2.) * s_tr.norm();

        if (s_tr_eq - sig0_0 <= 0)
        {
            //Elastic case. Use trial stress.
            output[SIGMA].Set(sigma_tr, i);
            _internal_vars_1[SIGMA].Set(sigma_tr,i);
        } else {
            //inelastic case. Use stress return algorithm.
            double del_lambda = 0.0;
            double sig0_1 = sig0_0 + _H * del_lambda;
            int j = 0;
            while ((s_tr_eq - 3*_mu*del_lambda - sig0_1)> 1e-10 && j < maxit) {
                del_lambda += del_lambda -(s_tr_eq - 3*_mu*del_lambda - sig0_1)/(- 3 * _mu - _H);
                sig0_1 = sig0_0 + _H * del_lambda;
                j++;
            }
            auto alpha = (1 - 3*_mu *del_lambda / s_tr_eq);
            auto s = alpha * s_tr;
            auto sigma = s + T_vol * p;
            output[SIGMA].Set(sigma,i);
            _internal_vars_1[SIGMA].Set(sigma,i);
            _internal_vars_1[LAMBDA].Set(_internal_vars_0[LAMBDA].GetScalar(i)+del_lambda, i);
        }
    }


    void Update(const std::vector<QValues>& input, int i) override
    {
        _internal_vars_0[EPS].Set(input[EPS].Get(i),i);

        _internal_vars_0[LAMBDA].Set(_internal_vars_1[LAMBDA].GetScalar(i), i);

        _internal_vars_0[SIGMA].Set(_internal_vars_1[SIGMA].Get(i), i);
    }

    void Resize(int n) override
    {
        for (auto& qvalues : _internal_vars_0)
            qvalues.Resize(n);

        for (auto& qvalues : _internal_vars_1)
            qvalues.Resize(n);
    }

};

struct RadialYieldSurface
{
    // returns the yield SURFACE (not the yield function) and its derivative with respect ro del_lambda
    virtual std::tuple<double, double> Evaluate(Eigen::VectorXd sigma, double lambda, double del_lambda) = 0;
};
class RadialMisesYieldSurface : public RadialYieldSurface
{
public:
  double _sig0;
  double _H;

  RadialMisesYieldSurface(double sig0, double H)
  : _sig0(sig0),
  _H(H)
  {

  }
  std::tuple<double,double> Evaluate(Eigen::VectorXd sigma, double lambda, double del_lambda) override
  {
    return {_sig0 + _H*(lambda+del_lambda), _H};
  }
};

class RadialReturnPlasticity : public LawInterface
{
public:
    std::shared_ptr<RadialYieldSurface> _Y;
    std::vector<QValues> _internal_vars_0;
    std::vector<QValues> _internal_vars_1;
    Eigen::MatrixXd _C;
    Eigen::VectorXd T_vol;
    Eigen::MatrixXd T_dev;
    double _mu;

    RadialReturnPlasticity(double E_, double nu, std::shared_ptr<RadialYieldSurface> Y)
    {
        _internal_vars_0.resize(Q::LAST);
        _internal_vars_1.resize(Q::LAST);

        _internal_vars_0[LAMBDA] = QValues(1);
        _internal_vars_1[LAMBDA] = QValues(1);

        _internal_vars_0[SIGMA] = QValues(6);
        _internal_vars_1[SIGMA] = QValues(6);
        _internal_vars_0[EPS] = QValues(6);

        const double l = E_ * nu / (1 + nu) / (1 - 2 * nu);
        _mu = E_ / (2.0 * (1 + nu));
        _Y = Y;
        _C.setZero(6,6);

        _C << 2*_mu+l, l, l, 0., 0., 0.,
             l, 2*_mu+l, l, 0., 0., 0.,
             l, l, 2*_mu+l, 0., 0., 0.,
             0., 0., 0., 2*_mu, 0., 0.,
             0., 0., 0., 0., 2*_mu, 0.,
             0., 0., 0., 0., 0., 2*_mu;

        T_dev.resize(6,6);
        T_vol.resize(6);
        T_dev <<
                2./3., -1./3., -1./3., 0., 0., 0.,
                -1./3., 2./3., -1./3., 0., 0., 0.,
                -1./3., -1./3., 2./3., 0., 0., 0.,
                0., 0., 0., 1., 0., 0.,
                0., 0., 0., 0., 1., 0.,
                0., 0., 0., 0., 0., 1.;
        T_vol << 1.,1.,1.,0.,0.,0.;


    }

    void DefineOutputs(std::vector<QValues>& output) const override
    {
        output[SIGMA] = QValues(6);
    }

    void DefineInputs(std::vector<QValues>& input) const override
    {
        input[EPS] = QValues(6);
    }
    Eigen::VectorXd GetInternalVar(Q which)
    {
        return _internal_vars_0.at(which).data;
    }

    void Evaluate(const std::vector<QValues>& input, std::vector<QValues>& output, int i) override
    {
        auto maxit = 100;

        auto strain = input[EPS].Get(i);
        auto lambda = _internal_vars_0[LAMBDA].GetScalar(i);
        Eigen::VectorXd sigma_tr(6);
        sigma_tr = _internal_vars_0[SIGMA].Get(i) + _C * (strain - _internal_vars_0[EPS].Get(i));
        auto s_tr = T_dev * sigma_tr;
        const double p = (1./3.) * T_vol.dot(sigma_tr);

        auto [Y, dY] = _Y->Evaluate(sigma_tr, lambda, 0.0);
        double s_tr_eq = std::sqrt(3. / 2.) * s_tr.norm();

        if (s_tr_eq - Y <= 0)
        {
            //Elastic case. Use trial stress.
            output[SIGMA].Set(sigma_tr, i);
            _internal_vars_1[SIGMA].Set(sigma_tr,i);
        } else {
            //inelastic case. Use stress return algorithm.
            double del_lambda = 0.0;
            int j = 0;
            do  {
                del_lambda += del_lambda - (s_tr_eq - 3*_mu*del_lambda - Y)/(-3*_mu-dY);
                std::tie(Y, dY) = _Y->Evaluate(sigma_tr, lambda, del_lambda);
                j++;
            } while (std::abs(s_tr_eq - 3 * _mu * del_lambda - Y)> 1e-10 && j < maxit);

            auto alpha = (1 - 3*_mu *del_lambda / s_tr_eq);
            auto s = alpha * s_tr;
            auto sigma = s + T_vol * p;
            output[SIGMA].Set(sigma,i);
            _internal_vars_1[SIGMA].Set(sigma,i);
            _internal_vars_1[LAMBDA].Set(_internal_vars_0[LAMBDA].GetScalar(i)+del_lambda, i);
        }
    }


    void Update(const std::vector<QValues>& input, int i) override
    {
        _internal_vars_0[EPS].Set(input[EPS].Get(i),i);

        _internal_vars_0[LAMBDA].Set(_internal_vars_1[LAMBDA].GetScalar(i), i);

        _internal_vars_0[SIGMA].Set(_internal_vars_1[SIGMA].Get(i), i);
    }

    void Resize(int n) override
    {
        for (auto& qvalues : _internal_vars_0)
            qvalues.Resize(n);

        for (auto& qvalues : _internal_vars_1)
            qvalues.Resize(n);
    }

};
