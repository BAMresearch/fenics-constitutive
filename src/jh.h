#pragma once
#include "interfaces.h"
#include "plasticity.h"
#include <tuple>
#include <eigen3/Eigen/Dense>
#include <math.h>
#include <complex>

struct JHParameters
{
    double A;
    double B;
    double C;
    double M;
    double N;
    double EPS0;
    double T;
    double FC;

}

class JH: public LawInterface
{
public:
    std::vector<QValues> _internal_vars;
    //std::vector<QValues> _internal_vars_1;
    Eigen::VectorXd T_vol;
    Eigen::MatrixXd T_dev;
    std::shared_ptr<JHParameters> _param;
    JH(std::shared_ptr<JHParameters> parameters)
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
    //std::complex<double> Y_f(double p_s, double lam, std::complex<double> del_lam)
    //{
       //return _sig0 + _H * (lam + del_lam); 
    //}
    void Evaluate(const std::vector<QValues>& input, std::vector<QValues>& output, int i) override
    {
        int maxit = 10;
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
        stress += 0.5 * h * (stress * W_.transpose() + W_ * stress);

        /***********************************************************************
         * START CONSTITUTIVE MODEL HERE
         * 1) Calculate failure surface Y_failure
         * 2) Calculate Yield surface Y_yield = f(Y_failure)
         **********************************************************************/
        double p_n = T_vol.dot(sigma_n);
        auto s_n = T_dev * matrix_to_mandel(stress);
        auto s_tr = s_n + 2. * _mu * T_dev * d_eps * h;
        double s_tr_eq = sqrt(1.5 * s_tr.transpose() * s_tr);
        double alpha = 0.0;

        double p_s = p_n / _param.PHEL;
        double del_lambda = 0.0;
        double complex_step = 1e-10;
        std::complex<double> ih(0.,complex_step);
        std::complex<double> Y_y;
        //std::complex<double> Y_f;
        //std::complex<double> Y_r;
        double Y_f;
        double Y_r;
        std::complex<double> rate_factor;


        Y_f = _param.A * pow(p_s + _param.TS, _param.N) * _param.SIGMA_HEL;
        Y_r = _param.B * pow(p_s,_param.M) * _param.SIGMA_HEL;

        if (D_n == 0.0){
            Y_y = Y_f;
        } else {
           Y_y = Y_f*(1.-D_n) + D_n * Y_r;
        }
        
        if (s_tr_eq >= Y_y.real()){
            //plastic flow initiated
            const double e_p_f = _param.D1 * pow(p_s + _param.TS, _param.D2);
            double f = 0.0;
            double df = 0.0;
            int j = 0;
            do  {
                //calculate yield surface with complex step
                rate_factor = 1. + _param.C * log(((del_lambda+ih)/h)/_param.EPS0);
                if (D_n + del_lambda/e_p_f <= 1) {
                    Y_y = (Y_f + (D_n + (del_lambda + ih)/e_p_f) * (Y_r - Y_f)) * rate_factor; 
                } else {
                    Y_y = Y_r * rate_factor;
                }
                // do the newton step
                f = s_tr_eq - 3.*_mu * del_lambda - Y_y.real();
                df =  3.*_mu + Y_y.imag()/complex_step;
                del_lambda += del_lambda - f/df;

                j++;
            } while (abs(f)> 1e-10 && j < maxit);
            
            alpha = (1. - 3.*_mu * del_lambda / s_tr_eq);

            _internal_vars[LAMBDA].Add(del_lambda, i);
            // Update damage variable or set to 1.
            _internal_vars[DAMAGE].Set(fmin(D_n+del_lambda/e_p_f,1.0));
        
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
        _internal_vars[RHO].Set(_internal_vars[RHO].GetScalar(i) * factor_1.determinant() / factor_2.determinant(), i);
        /***********************************************************************
         * UPDATE ENERGY AND EOS
         **********************************************************************/
        //TODO: Update this part
        auto rho = 0.5 * (_internal_vars_0[RHO].GetScalar(i) + _internal_vars_1[RHO].GetScalar(i));
        auto eta = _internal_vars_1[RHO].GetScalar(i)/_rho0 - 1.;
        const Eigen::VectorXd s_12 = 0.5*(T_dev * sigma_n + s);
        auto e0 = e_n;
        auto e1 = e0;
        auto e_tilde = e0 + (h/rho) * (s_12.dot(T_dev * d_eps)-0.5*p_n*_internal_vars_0[RHO].GetScalar(i) * d_eps_vol);
        do{
            e0 = e1;
            e1 = e_tilde - 0.5*(h/rho)*d_eps_vol*_eos->Evaluate(eta ,e0);
        } while (std::abs(e1-e0)>1e-10);

        auto p = _eos->Evaluate(eta, e1);
        _internal_vars_1[E].Set(e1, i);
        /***********************************************************************
         * Combine deviatoric and volumetric stresses and apply stress rate
         **********************************************************************/

        stress = mandel_to_matrix(s - T_vol * p);
        
        stress += 0.5 * h * (stress * W_.transpose() + W_ * stress);
        
        output[SIGMA].Set(matrix_to_mandel(stress),i);
    }


    void Update(const std::vector<QValues>& input, int i) override
    {
        _internal_vars_0[E].Set(_internal_vars_1[E].GetScalar(i),i);
        _internal_vars_0[LAMBDA].Set(_internal_vars_1[LAMBDA].GetScalar(i), i);
        _internal_vars_0[RHO].Set(_internal_vars_1[RHO].GetScalar(i), i);
    }

    void Resize(int n) override
    {
        for (auto& qvalues : _internal_vars_0)
            qvalues.Resize(n);

        for (auto& qvalues : _internal_vars_1)
            qvalues.Resize(n);
    }

};
