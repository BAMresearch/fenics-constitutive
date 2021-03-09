#pragma once
#include "interfaces.h"
#include <tuple>

class NormVM
{
public:
    NormVM(Constraint c)
    {
        _q = Dim::Q(c);
        _P.setZero(_q, _q);

        const double f = 6.;

        if (_q == 1)
        {
            _P(0, 0) = 2. / 3.;
        }
        else if (_q == 3)
        {
            _P << 2, -1, 0, -1, 2, 0, 0, 0, f;
            _P *= 1. / 3.;
        }
        else if (_q == 4)
        {
            _P << 2, -1, -1, 0, -1, 2, -1, 0, -1, -1, 2, 0, 0, 0, 0, f;
            _P *= 1. / 3.;
        }
        else if (_q == 6)
        {
            // kein bock.
        }
    }

    std::pair<double, Eigen::VectorXd> Call(Eigen::VectorXd ss) const
    {
        if (_q == 1)
        {
            Eigen::VectorXd m;
            m.resize(_q);
            m[0] = 1;
            if (ss[0] < 0.)
                m[0] = -1;
            return {std::abs(ss[0]), m};
        }
        else
        {
            assert(ss.rows() == _q);
            double se = std::sqrt(1.5 * ss.transpose() * _P * ss);
            if (se == 0)
            {
                return {se, Eigen::MatrixXd::Zero(_q, _q)};
            }
            else
            {
                return {se, 1.5 / se * _P * ss};
            }
        }
    }

    int _q;
    Eigen::MatrixXd _P;
};

class RateIndependentHistory
{
public:
    // Attributes
    Eigen::MatrixXd _p;
    Eigen::MatrixXd _dp_dsig;
    Eigen::MatrixXd _dp_dk;
    
    // Constructor
    RateIndependentHistory()
    {
    _p.setOnes(1, 1);
    _dp_dsig.setZero(1, 1);
    _dp_dk.setZero(1, 1);
    }
    
    // Call
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> Call(Eigen::VectorXd sigma, double kappa)
    {
    return {_p, _dp_dsig, _dp_dk};
    }
};


std::pair<double,double> constitutive_coeffs(double E=1000., double noo=0.0, Constraint c = PLANE_STRESS){
    double lamda = (E*noo/(1.+noo))/(1.-2.*noo);
    double mu = E/(2.*(1.+noo));
    if (c == PLANE_STRESS)
        lamda = 2.*mu*lamda/(lamda+2.*mu);
    return {mu, lamda};
}

class ElasticConstitutive
{
    public:
        double _E;
        double _noo;
        Constraint _c;
        int _ss_dim;
        Eigen::MatrixXd _D;

        ElasticConstitutive(double E, double noo, Constraint c){
        _E = E;
        _noo = noo;
        _c = c;
        _ss_dim = Dim::Q(c);
        
        if (_ss_dim == 1){
            _D << _E;
        }else{
            double mu, lamda;
            std::tie(mu, lamda) = constitutive_coeffs(_E, _noo, _c);
            double fact = 1.0;
            if (_ss_dim == 3){
                _D.resize(3,3);
                _D << 1.0 , _noo, 0.0, _noo, 1.0, 0.0, 0.0, 0.0, fact * 0.5 * (1 - _noo);
                _D *= _E / (1.0 - _noo * _noo);
            } else if(_ss_dim == 4){
                _D.resize(4,4);
                _D <<   2 * mu + lamda, lamda, lamda, 0,
                        lamda, 2 * mu + lamda, lamda, 0,
                        lamda, lamda, 2 * mu + lamda, 0,
                        0, 0, 0, fact * mu;
            }else if(_ss_dim == 6){
                _D.resize(6,6);
                _D <<
                        2 * mu + lamda, lamda, lamda, 0, 0, 0,
                        lamda, 2 * mu + lamda, lamda, 0, 0, 0,
                        lamda, lamda, 2 * mu + lamda, 0, 0, 0,
                        0, 0, 0, fact * mu, 0, 0,
                        0, 0, 0, 0, fact * mu, 0,
                        0, 0, 0, 0, 0, fact * mu;
                    
            }
        }
        }
};
class YieldVM
{
public:
    double _y0;
    Constraint _c;
    int _ss_dim;
    double _H;
    NormVM _vm_norm;


    YieldVM(double y0, Constraint c, double H=0):
        _y0(y0),
        _c(c),
        _ss_dim(Dim::Q(c)),
        _H(H),
        _vm_norm(NormVM(c))
    {
    }
        
    std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd, double, Eigen::VectorXd> Call(Eigen::VectorXd stress, double kappa=0.0)
    {
        //Evaluate the yield function quantities at a specific stress level (as a vector):
            //f: yield function itself
            //m: flow vector; derivative of "f" w.r.t. stress (associated flow rule)
            //dm: derivative of flow vector w.r.t. stress; second derivative of "f" w.r.t. stress
            //fk: derivative of "f" w.r.t. kappa
            //mk: derivative of "m" w.r.t. kappa
        //The given stress vector must be consistent with self.ss_dim
        //kappa: history variable(s), here related to isotropic hardening
        double se;
        Eigen::VectorXd m;
        Eigen::MatrixXd dm;
        Eigen::VectorXd mk;
        double f;

        std::tie(se, m) = _vm_norm.Call(stress);
        f = se - (_y0 + _H * kappa);
        //fk = - _H

        if (_ss_dim==1){
            dm.setZero(1,1);
            mk.setZero(1);
        } else {
                // Actually None, but None is too Python specific
                dm.setZero(_ss_dim,_ss_dim);
            if (se != 0.0){
                dm = 6.0 * se * _vm_norm._P -6.0 * (_vm_norm._P * stress) * m.transpose();
                dm /= 4 * se * se;
            }
            mk.setZero(_ss_dim);
        }
        return {f, m, dm, -_H, mk};
    }
};
