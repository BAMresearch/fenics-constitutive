#pragma once
#include "interfaces.h"
//#include "plasticity.h"
#include <tuple>
#include <eigen3/Eigen/Dense>
#include <math.h>
#include <complex>

struct JH2Parameters
{
    double RHO = 2.440e-6; //kg/mm^3
    double SHEAR_MODULUS = 12.5; //GPa
    double A = 0.6304;
    double B = 0.2101;
    double C = 0.006;
    double M = 0.8437;
    double N = 0.8437;
    double EPS0 = 1.0;
    double T = 0.0034;

    double SIGMAHEL = 1.005;
    double PHEL = 0.811;

    double D1 = 0.6;
    double D2 = 0.1;


    double K1 = 16.667;
    double K2 = 73.19;
    double K3 = -236.2;
    double BETA = 1.0;
    double MOGEL = 1.0;

};


class JH2Simple: public LawInterface
{
public:
    std::vector<QValues> _internal_vars;
    //std::vector<QValues> _internal_vars_1;
    Eigen::VectorXd T_vol;
    Eigen::MatrixXd T_dev;
    std::shared_ptr<JH2Parameters> _param;
    
    JH2Simple(std::shared_ptr<JH2Parameters> parameters)
        :  _param(parameters)
    {
        _internal_vars.resize(Q::LAST);
        //Accumulated plastic strain saved in LAMBDA
        _internal_vars[LAMBDA] = QValues(1);
        //internal energy saved in E
        _internal_vars[E] = QValues(1);
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
        const auto e_n = _internal_vars[E].GetScalar(i);
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

        double p_s = p_n / _param->PHEL;
        double t_s = _param->T / _param->PHEL;
        double del_lambda = 0.0;
        double Y_yield = 0.0;
        double dY_yield = 0.0;
        double Y_f;
        double Y_r;
        double rate_factor;


        Y_f = fmax(_param->A * pow(p_s + t_s, _param->N) * _param->SIGMAHEL, 0.0);
        Y_r = fmax(_param->B * pow(p_s,_param->M) * _param->SIGMAHEL,0.0);
        if (d_eps_eq >= _param->EPS0){
            rate_factor = 1. + _param->C * log(d_eps_eq/_param->EPS0);
        } else {
            rate_factor = 1.;
        }
        if (D_n == 0.0){
            Y_yield = Y_f*rate_factor;
        } else {
            Y_yield = (Y_f*(1.-D_n) + D_n * Y_r)*rate_factor;
        }
        if (s_tr_eq > Y_yield){
            const double e_p_f = fmax(_param->D1 * pow(p_s + t_s, _param->D2), 1e-200);
            
            del_lambda =_param->MOGEL * (s_tr_eq-Y_yield) / (3.*_param->SHEAR_MODULUS);// + (Y_r-Y_f)/e_p_f);
            //Y_yield += del_lambda * (Y_r-Y_f)/e_p_f;
            alpha = Y_yield/s_tr_eq;

            _internal_vars[LAMBDA].Add(del_lambda, i);
            // Update damage variable or set to 1.
            _internal_vars[DAMAGE].Set(fmin(D_n+del_lambda/e_p_f,1.0), i);
        
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
        
        const auto p = (mu > 0) ? _param->K1 * mu + _param->K2 * mu * mu + _param->K3 * mu * mu * mu + _internal_vars[PRESSURE].GetScalar(i): _param->K1 * mu;
        
        const double D_new = _internal_vars[DAMAGE].GetScalar(i);
        if (D_new > D_n){
            const double Y_old = (D_n * Y_r + (1-D_n) * Y_f);// * _param->SIGMAHEL;
            const double Y_new = (D_new * Y_r + (1-D_new) * Y_f);// * _param->SIGMAHEL;
            const double U_old = (Y_old * Y_old) / (6. * _param->SHEAR_MODULUS);
            const double U_new = (Y_new * Y_new) / (6. * _param->SHEAR_MODULUS);
            const double del_U = U_old - U_new;
            if (del_U < 0){
                cout << "help, this is wrong\n";
            } else {
            const double del_P_n = _internal_vars[PRESSURE].GetScalar(i);
            double K1 = _param->K1;
            double del_P = -K1 * mu + sqrt(pow(K1 * mu + del_P_n,2)+2.*_param->BETA * K1 * del_U);
            _internal_vars[PRESSURE].Set(del_P,i);
            }
        }

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

class JH2Nonlocal: public LawInterface
{
public:
    std::vector<QValues> _internal_vars;
    //std::vector<QValues> _internal_vars_1;
    Eigen::VectorXd T_vol;
    Eigen::MatrixXd T_dev;
    std::shared_ptr<JH2Parameters> _param;
    
    JH2Nonlocal(std::shared_ptr<JH2Parameters> parameters)
        :  _param(parameters)
    {
        _internal_vars.resize(Q::LAST);
        //Accumulated plastic strain saved in LAMBDA
        _internal_vars[LAMBDA] = QValues(1);
        //internal energy saved in E
        _internal_vars[E] = QValues(1);
        //current density saved in rho
        _internal_vars[RHO] = QValues(1);

        _internal_vars[NONLOCAL] = QValues(1);
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
        input[NONLOCAL] = QValues(1);
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
        const auto e_n = _internal_vars[E].GetScalar(i);
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

        double p_s = p_n / _param->PHEL;
        double t_s = _param->T / _param->PHEL;
        double del_lambda = 0.0;
        double Y_yield = 0.0;
        double dY_yield = 0.0;
        double Y_f;
        double Y_r;
        double rate_factor;

        const double e_p_f = fmax(_param->D1 * pow(p_s + t_s, _param->D2), 1e-200);
        const auto p_nl = _internal_vars[NONLOCAL].GetScalar(i);
        const auto del_p_nl = fmax(input[NONLOCAL].GetScalar(i) - p_nl, 0.);
        _internal_vars[NONLOCAL].Set(p_nl+del_p_nl,i);
        // Update damage variable or set to 1.
        //_internal_vars[DAMAGE].Set(fmin(D_n+input[NONLOCAL].GetScalar(i)/e_p_f,1.0), i);
        _internal_vars[DAMAGE].Set(fmin(D_n+del_p_nl/e_p_f,1.0), i);

        const auto D_n1 = _internal_vars[DAMAGE].GetScalar(i);
        
        Y_f = fmax(_param->A * pow(p_s + t_s, _param->N) * _param->SIGMAHEL, 0.0);
        Y_r = fmax(_param->B * pow(p_s,_param->M) * _param->SIGMAHEL,0.0);
        if (d_eps_eq >= _param->EPS0){
            rate_factor = 1. + _param->C * log(d_eps_eq/_param->EPS0);
        } else {
            rate_factor = 1.;
        }
        if (D_n1 == 0.0){
            Y_yield = Y_f*rate_factor;
        } else {
            Y_yield = (Y_f*(1.-D_n1) + D_n1 * Y_r)*rate_factor;
        }
        if (s_tr_eq > Y_yield){
            
            del_lambda =_param->MOGEL * (s_tr_eq-Y_yield) / (3.*_param->SHEAR_MODULUS);// + (Y_r-Y_f)/e_p_f);
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
        
        const auto p = (mu > 0) ? _param->K1 * mu + _param->K2 * mu * mu + _param->K3 * mu * mu * mu + _internal_vars[PRESSURE].GetScalar(i): _param->K1 * mu;
        
        const double D_new = D_n1;
        if (D_new > D_n){
            const double Y_old = (D_n * Y_r + (1-D_n) * Y_f);// * _param->SIGMAHEL;
            const double Y_new = (D_new * Y_r + (1-D_new) * Y_f);// * _param->SIGMAHEL;
            const double U_old = (Y_old * Y_old) / (6. * _param->SHEAR_MODULUS);
            const double U_new = (Y_new * Y_new) / (6. * _param->SHEAR_MODULUS);
            const double del_U = U_old - U_new;
            if (del_U < 0){
                cout << "help, this is wrong\n";
            } else {
            const double del_P_n = _internal_vars[PRESSURE].GetScalar(i);
            double K1 = _param->K1;
            double del_P = -K1 * mu + sqrt(pow(K1 * mu + del_P_n,2)+2.*_param->BETA * K1 * del_U);
            _internal_vars[PRESSURE].Set(del_P,i);
            }
        }

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
