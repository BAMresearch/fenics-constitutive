#pragma once
#include "interfaces.h"
//#include "plasticity.h"
#include <tuple>
#include <eigen3/Eigen/Dense>
#include <math.h>
#include <complex>

struct JHConcreteParameters
{
    double RHO = 2.440e-6; //kg/mm^3
    double SHEAR_MODULUS = 14.86; //GPa
    double A = 0.79;
    double B = 1.6;
    double N = 0.61;
    double C = 0.007;
    double FC = 0.048; //GPa
    double SMAX = 7.0; //GPa

    double D1 = 0.04;
    double D2 = 1.;
    double EFMIN = 0.01;

    double EPS0 = 1e-3;
    
    double P_CRUSH = 0.016;
    double MU_CRUSH = 0.001;
    double K1 = 85;
    double K2 = -171;
    double K3 = 208;
    double P_LOCK = 0.8;
    double MU_LOCK = 0.1;
    double T = 0.004

};


class JHConcrete: public LawInterface
{
public:
    std::vector<QValues> _internal_vars;
    //std::vector<QValues> _internal_vars_1;
    Eigen::VectorXd T_vol;
    Eigen::MatrixXd T_dev;
    std::shared_ptr<JHConcreteParameters> _param;
    
    JHConcrete(std::shared_ptr<JHConcreteParameters> parameters)
        :  _param(parameters)
    {
        _internal_vars.resize(Q::LAST);
        //Accumulated plastic strain saved in LAMBDA
        _internal_vars[LAMBDA] = QValues(1);
        //internal energy saved in E
        //_internal_vars[E] = QValues(1);
        //current density saved in rho
        _internal_vars[RHO] = QValues(1);

        _internal_vars[DAMAGE] = QValues(1);
        _internal_vars[PRESSURE] = QValues(1);

        T_dev.resize(6,6);
        T_vol.resize(6);
        T_dev <<
                2./3., -1./3., -1./3., 0., 0., 0.,
                -1./3., 2./3., -1./3., 0., 0., 0.,
                -1./3., -1./3., 2./3., 0., 0., 0.,
                0., 0., 0., 1., 0., 0.,
                0., 0., 0., 0., 1., 0.,
                0., 0., 0., 0., 0., 1.;
        T_vol << 1./3.,1./3.,1./3.,0.,0.,0.;


    }

    void DefineOutputs(std::vector<QValues>& output) const override
    {
        output[SIGMA] = QValues(6);
    }

    void DefineInputs(std::vector<QValues>& input) const override
    {
        input[L] = QValues(3,3);
        input[SIGMA] = QValues(6);
        input[TIME_STEP] = QValues(1);
    }
    Eigen::VectorXd GetInternalVar(Q which)
    {
        return _internal_vars.at(which).data;
    }
    void Evaluate(const std::vector<QValues>& input, std::vector<QValues>& output, int i) override
    {
        int maxit = 1;
        const Eigen::Matrix3d L_ = input[L].Get(i);
        const Eigen::VectorXd sigma_n = input[SIGMA].Get(i);
        const auto h = input[TIME_STEP].GetScalar(i);
        const auto lambda_n = _internal_vars[LAMBDA].GetScalar(i);
        //const auto e_n = _internal_vars[E].GetScalar(i);
        const auto D_n = _internal_vars[DAMAGE].GetScalar(i);

        const auto D_ = 0.5 * (L_ + L_.transpose());
        const auto W_ = 0.5 * (L_ - L_.transpose());
        const auto d_eps = matrix_to_mandel(D_);
        const auto d_eps_vol = T_vol.dot(d_eps);
        
        auto stress = mandel_to_matrix(sigma_n);
        stress += h * (stress * W_.transpose() + W_ * stress);

        /***********************************************************************
         * START CONSTITUTIVE MODEL HERE
         * 1) Calculate failure surface Y_failure
         * 2) Calculate Yield surface Y_yield = f(Y_failure)
         **********************************************************************/
        double p_n = - T_vol.dot(sigma_n);// + _internal_vars[PRESSURE].GetScalar(i);
        auto s_n = T_dev * matrix_to_mandel(stress);
        auto s_tr = s_n + 2. * _param->SHEAR_MODULUS * T_dev * d_eps * h;
        double s_tr_eq = sqrt(1.5 * s_tr.transpose() * s_tr);
        double d_eps_eq = sqrt((2./3.) * d_eps.transpose() * d_eps);
        double alpha = 0.0;

        double p_s = p_n / _param->FC;
        double t_s = _param->T / _param->FC;
        double del_lambda = 0.0;
        double Y_yield = 0.0;
        double rate_factor;


        //Y_f = fmax(_param->A * pow(p_s + t_s, _param->N) * _param->SIGMAHEL, 0.0);
        //Y_r = fmax(_param->B * pow(p_s,_param->M) * _param->SIGMAHEL,0.0);

        if (d_eps_eq >= _param->EPS0){
            rate_factor = 1. + _param->C * log(d_eps_eq/_param->EPS0);
        } else {
            rate_factor = 1.;
        }
        
        Y_yield = (_param->A*(1.-D_n) + _param->B * pow(p_s,_param->N))*rate_factor;
        
        if (s_tr_eq > Y_yield){
            const double e_p_f = fmax(_param->D1 * pow(p_s + t_s, _param->D2), _param->EFMIN);
            
            del_lambda = (s_tr_eq-Y_yield) / (3.*_param->SHEAR_MODULUS);// + (Y_r-Y_f)/e_p_f);
            //Y_yield += del_lambda * (Y_r-Y_f)/e_p_f;
            alpha = Y_yield/s_tr_eq;

            _internal_vars[LAMBDA].Add(del_lambda, i);
        
        } else {
            //elastic
            alpha = 1.0;
        }

        //Update deviatoric stress s
        auto s = alpha * s_tr;
        /***********************************************************************
         * END CONSTITUTIVE MODEL HERE
         **********************************************************************/

        /***********************************************************************
         * UPDATE DENSITY
         * The density is updated using the explicit midpoint rule for the
         * deformation gradient.
         **********************************************************************/
        auto factor_1 = Eigen::MatrixXd::Identity(3,3)-0.5*h*L_;
        auto factor_2 = Eigen::MatrixXd::Identity(3,3)+0.5*h*L_;
        if (_internal_vars[RHO].GetScalar(i) == 0.0){
            _internal_vars[RHO].Set(_param->RHO * factor_1.determinant() / factor_2.determinant(), i);
        } else {
            _internal_vars[RHO].Set(_internal_vars[RHO].GetScalar(i) * factor_1.determinant() / factor_2.determinant(), i);
        }
        /***********************************************************************
         * UPDATE ENERGY AND EOS
         **********************************************************************/
        
        const auto mu = _internal_vars[RHO].GetScalar(i)/_param->RHO -1.;
        const auto mu_bar = (mu-_param->MU_LOCK)/(1.+_param->MU_LOCK);  
        
        const auto p = (mu > 0) ? _param->K1 * mu + _param->K2 * mu * mu + _param->K3 * mu * mu * mu : _param->K1 * mu;
        

        // Update damage variable or set to 1.
        const double eps_f_p =fmax(_param->D1*pow(p_s+t_s,_param->D2), _param->EFMIN);

        _internal_vars[DAMAGE].Set(fmin(D_n+del_lambda/e_p_f,1.0), i);
        /***********************************************************************
         * Combine deviatoric and volumetric stresses and apply stress rate
         **********************************************************************/

        //stress = mandel_to_matrix(s - 3. * T_vol * p);
        
        
        //output[SIGMA].Set(matrix_to_mandel(stress),i);
        output[SIGMA].Set(s - 3. * T_vol * p, i);
    }


    void Update(const std::vector<QValues>& input, int i) override
    {
    }

    void Resize(int n) override
    {
        for (auto& qvalues : _internal_vars)
            qvalues.Resize(n);


    }

};

