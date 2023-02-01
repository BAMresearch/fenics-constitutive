#pragma once
#include "interfaces.h"
#include "stress_strain.h"
#include <tuple>
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
    double EFMIN = 1e-200;

};


template <Constraint TC> class JH2: public RefLawInterface
{
public:
    //std::vector<QValues> _internal_vars;
    std::vector<Eigen::VectorXd> _internal_vars;
    //Eigen::VectorXd T_vol;
    //Eigen::MatrixXd T_dev;
    std::shared_ptr<JH2Parameters> _param;
    
    JH2(std::shared_ptr<JH2Parameters> parameters, int n)
        : RefLawInterface(n), _param(parameters)
    {
        _internal_vars.resize(Q::LAST);
        //Accumulated plastic strain saved in LAMBDA
        _internal_vars[Q::LAMBDA] = Eigen::VectorXd::Zero(_n);
        //current density saved in rho
        _internal_vars[Q::RHO] = Eigen::VectorXd::Constant(_n, _param->RHO);

        _internal_vars[Q::DAMAGE] = Eigen::VectorXd::Zero(_n);
        _internal_vars[Q::PRESSURE] =Eigen::VectorXd::Zero(_n);

    }


    std::vector<Q> DefineInputs() const override
    {
        return {Q::GRAD_V, Q::SIGMA};
    }
    Eigen::VectorXd GetInternalVar(Q which)
    {
        return _internal_vars.at(which);
    }
    void EvaluateIP(int i, std::vector<Eigen::Ref<Eigen::VectorXd>>& inputs, double del_t) override
    {
        constexpr int L_size = Dim::G(TC)*Dim::G(TC);
        constexpr int sigma_size = Dim::StressStrain(TC);
        const FullTensor<TC> L_ = Eigen::Map<FullTensor<TC>>(inputs[Q::GRAD_V].segment<L_size>(i * L_size).data());
        const FullTensor<TC> D_ = 0.5 * (L_ + L_.transpose());
        
        const auto d_eps = TensorToMandel<TC>(D_);
        //const auto d_eps_vol = T_vol.dot(d_eps);
        
        auto sigma_view = inputs[Q::SIGMA].segment<sigma_size>(i * sigma_size);
        
        //const double lambda_n = _internal_vars[LAMBDA](i);
        //const double e_n = _internal_vars[E](i);
        const double D_n = _internal_vars[Q::DAMAGE](i);

        
        //auto stress = mandel_to_matrix(sigma_n);
        //stress += h * (stress * W_.transpose() + W_ * stress);

        /***********************************************************************
         * START CONSTITUTIVE MODEL HERE
         * 1) Calculate failure surface Y_failure
         * 2) Calculate Yield surface Y_yield = f(Y_failure)
         **********************************************************************/
        double p_n = - T_Vol<TC>.dot(sigma_view);// + _internal_vars[PRESSURE].GetScalar(i);
        auto s_n = T_Dev<TC> * sigma_view;
        MandelVector<TC> s_tr = s_n + 2. * _param->SHEAR_MODULUS * (T_Dev<TC> * d_eps) * del_t;
        const double s_tr_eq = sqrt(1.5 * s_tr.squaredNorm());
        const double d_eps_eq = sqrt((2./3.) * d_eps.squaredNorm());
        double alpha = 0.0;

        double p_s = p_n / _param->PHEL;
        double t_s = _param->T / _param->PHEL;
        double del_lambda = 0.0;
        double Y_yield = 0.0;
        //double dY_yield = 0.0;
        double Y_f;
        double Y_r;
        double rate_factor = 1.;


        Y_f = fmax(_param->A * pow(p_s + t_s, _param->N) * _param->SIGMAHEL, 0.0);
        Y_r = fmax(_param->B * pow(p_s,_param->M) * _param->SIGMAHEL,0.0);
        if (d_eps_eq >= _param->EPS0){
            rate_factor += _param->C * log(d_eps_eq/_param->EPS0);
        } 
        if (D_n == 0.0){
            Y_yield = Y_f*rate_factor;
        } else {
            Y_yield = (Y_f*(1.-D_n) + D_n * Y_r)*rate_factor;
        }
        if (s_tr_eq > Y_yield){
            const double e_p_f = fmax(_param->D1 * pow(p_s + t_s, _param->D2), _param->EFMIN);
            
            del_lambda =_param->MOGEL * (s_tr_eq-Y_yield) / (3.*_param->SHEAR_MODULUS);// + (Y_r-Y_f)/e_p_f);
            alpha = Y_yield/s_tr_eq;

            _internal_vars[Q::LAMBDA](i) += del_lambda;
            // Update damage variable or set to 1.
            _internal_vars[Q::DAMAGE](i) = fmin(D_n+del_lambda/e_p_f,1.0);
        
        } else {
            //elastic
            alpha = 1.0;
        }

        //Update deviatoric stress s
        //auto s = alpha * s_tr;
        /***********************************************************************
         * END CONSTITUTIVE MODEL HERE
         **********************************************************************/

        /***********************************************************************
         * UPDATE DENSITY
         * The density is updated using the explicit midpoint rule for the
         * deformation gradient.
         **********************************************************************/
/*         FullTensor<TC> factor_1 = FullTensor<TC>::Identity()-0.5*del_t*L_;
        FullTensor<TC> factor_2 = FullTensor<TC>::Identity()+0.5*del_t*L_;
        if (_internal_vars[RHO](i) == 0.0){
            _internal_vars[RHO](i)=_param->RHO * factor_1.determinant() / factor_2.determinant();
        } else {
            _internal_vars[RHO](i)*= factor_1.determinant() / factor_2.determinant();
        } */
        
        double f1 = del_t/2. * L_.trace();
        _internal_vars[Q::RHO](i) *= (1-f1)/(1+f1);
        /***********************************************************************
         * UPDATE ENERGY AND EOS
         **********************************************************************/
        
        const auto mu = _internal_vars[RHO](i)/_param->RHO -1.;
        
        const auto p = (mu > 0) ? _param->K1 * mu + _param->K2 * mu * mu + _param->K3 * mu * mu * mu + _internal_vars[Q::PRESSURE](i): _param->K1 * mu;
        
        const double D_new = _internal_vars[DAMAGE](i);
        if (D_new > D_n){
            const double Y_old = (D_n * Y_r + (1-D_n) * Y_f);// * _param->SIGMAHEL;
            const double Y_new = (D_new * Y_r + (1-D_new) * Y_f);// * _param->SIGMAHEL;
            const double U_old = (Y_old * Y_old) / (6. * _param->SHEAR_MODULUS);
            const double U_new = (Y_new * Y_new) / (6. * _param->SHEAR_MODULUS);
            const double del_U = U_old - U_new;
            
            const double del_P_n = _internal_vars[PRESSURE](i);
            double K1 = _param->K1;
            double del_P = -K1 * mu + sqrt(pow(K1 * mu + del_P_n,2)+2.*_param->BETA * K1 * del_U);
            _internal_vars[PRESSURE](i)=del_P;
            //}
        }

        /***********************************************************************
         * Combine deviatoric and volumetric stresses and apply stress rate
         **********************************************************************/

        sigma_view = alpha * s_tr - T_Id<TC> * p;
    }


    // void Update(const std::vector<QValues>& input, int i) override
    // {
    // }

    // void Resize(int n) override
    // {
    //     for (auto& qvalues : _internal_vars)
    //         qvalues.Resize(n);


    // }

};

template <Constraint TC> class JH2Nonlocal: public RefLawInterface
{
public:
    //std::vector<QValues> _internal_vars;
    std::vector<Eigen::VectorXd> _internal_vars;
    //Eigen::VectorXd T_vol;
    //Eigen::MatrixXd T_dev;
    std::shared_ptr<JH2Parameters> _param;
    
    JH2Nonlocal(std::shared_ptr<JH2Parameters> parameters, int n)
        : RefLawInterface(n), _param(parameters)
    {
        _internal_vars.resize(Q::LAST);
        //Accumulated plastic strain saved in LAMBDA
        _internal_vars[Q::LAMBDA] = Eigen::VectorXd::Zero(_n);
        //current density saved in rho
        _internal_vars[Q::RHO] = Eigen::VectorXd::Constant(_n, _param->RHO);

        _internal_vars[Q::DAMAGE] = Eigen::VectorXd::Zero(_n);
        _internal_vars[Q::PRESSURE] =Eigen::VectorXd::Zero(_n);
        _internal_vars[Q::NONLOCAL] = Eigen::VectorXd::Zero(_n);

    }


    std::vector<Q> DefineInputs() const override
    {
        return {Q::GRAD_V, Q::SIGMA, Q::NONLOCAL};
    }
    Eigen::VectorXd GetInternalVar(Q which)
    {
        return _internal_vars.at(which);
    }
    void EvaluateIP(int i, std::vector<Eigen::Ref<Eigen::VectorXd>>& inputs, double del_t) override
    {
        constexpr int L_size = Dim::G(TC)*Dim::G(TC);
        constexpr int sigma_size = Dim::StressStrain(TC);
        const FullTensor<TC> L_ = Eigen::Map<FullTensor<TC>>(inputs[Q::GRAD_V].segment<L_size>(i * L_size).data());
        const FullTensor<TC> D_ = 0.5 * (L_ + L_.transpose());
        
        const auto d_eps = TensorToMandel<TC>(D_);
        //const auto d_eps_vol = T_vol.dot(d_eps);
        
        auto sigma_view = inputs[Q::SIGMA].segment<sigma_size>(i * sigma_size);
        
        //const double lambda_n = _internal_vars[LAMBDA](i);
        //const double e_n = _internal_vars[E](i);
        const double D_n = _internal_vars[Q::DAMAGE](i);

        
        //auto stress = mandel_to_matrix(sigma_n);
        //stress += h * (stress * W_.transpose() + W_ * stress);

        /***********************************************************************
         * START CONSTITUTIVE MODEL HERE
         * 1) Calculate failure surface Y_failure
         * 2) Calculate Yield surface Y_yield = f(Y_failure)
         **********************************************************************/
        double p_n = - T_Vol<TC>.dot(sigma_view);// + _internal_vars[PRESSURE].GetScalar(i);
        auto s_n = T_Dev<TC> * sigma_view;
        MandelVector<TC> s_tr = s_n + 2. * _param->SHEAR_MODULUS * (T_Dev<TC> * d_eps) * del_t;
        const double s_tr_eq = sqrt(1.5 * s_tr.squaredNorm());
        const double d_eps_eq = sqrt((2./3.) * d_eps.squaredNorm());
        double alpha = 0.0;

        double p_s = p_n / _param->PHEL;
        double t_s = _param->T / _param->PHEL;
        double del_lambda = 0.0;
        double Y_yield = 0.0;
        //double dY_yield = 0.0;
        double Y_f;
        double Y_r;
        double rate_factor = 1.;

        const double e_p_f = fmax(_param->D1 * pow(p_s + t_s, _param->D2), _param->EFMIN);
        const auto p_nl = _internal_vars[Q::NONLOCAL](i);
        const auto del_p_nl = fmax(inputs[Q::NONLOCAL](i) - p_nl, 0.);
        _internal_vars[NONLOCAL](i) += del_p_nl;
        // Update damage variable or set to 1.
        //_internal_vars[DAMAGE].Set(fmin(D_n+input[NONLOCAL].GetScalar(i)/e_p_f,1.0), i);
        _internal_vars[DAMAGE](i) = fmin(D_n+del_p_nl/e_p_f,1.0);
        
        const double D_new = _internal_vars[DAMAGE](i);

        Y_f = fmax(_param->A * pow(p_s + t_s, _param->N) * _param->SIGMAHEL, 0.0);
        Y_r = fmax(_param->B * pow(p_s,_param->M) * _param->SIGMAHEL,0.0);
        if (d_eps_eq >= _param->EPS0){
            rate_factor += _param->C * log(d_eps_eq/_param->EPS0);
        } 
        if (D_new == 0.0){
            Y_yield = Y_f*rate_factor;
        } else {
            Y_yield = (Y_f*(1.-D_new) + D_new * Y_r)*rate_factor;
        }
        if (s_tr_eq > Y_yield){
            
            del_lambda =_param->MOGEL * (s_tr_eq-Y_yield) / (3.*_param->SHEAR_MODULUS);// + (Y_r-Y_f)/e_p_f);
            alpha = Y_yield/s_tr_eq;

            _internal_vars[Q::LAMBDA](i) += del_lambda;
        
        } else {
            //elastic
            alpha = 1.0;
        }

        //Update deviatoric stress s
        //auto s = alpha * s_tr;
        /***********************************************************************
         * END CONSTITUTIVE MODEL HERE
         **********************************************************************/

        /***********************************************************************
         * UPDATE DENSITY
         * The density is updated using the explicit midpoint rule for the
         * deformation gradient.
         **********************************************************************/
        
        double f1 = del_t/2. * L_.trace();
        _internal_vars[Q::RHO](i) *= (1-f1)/(1+f1);
        /***********************************************************************
         * UPDATE ENERGY AND EOS
         **********************************************************************/
        
        const auto mu = _internal_vars[Q::RHO](i)/_param->RHO -1.;
        
        const auto p = (mu > 0) ? _param->K1 * mu + _param->K2 * mu * mu + _param->K3 * mu * mu * mu + _internal_vars[Q::PRESSURE](i): _param->K1 * mu;
        
        if (D_new > D_n){
            const double Y_old = (D_n * Y_r + (1-D_n) * Y_f);// * _param->SIGMAHEL;
            const double Y_new = (D_new * Y_r + (1-D_new) * Y_f);// * _param->SIGMAHEL;
            const double U_old = (Y_old * Y_old) / (6. * _param->SHEAR_MODULUS);
            const double U_new = (Y_new * Y_new) / (6. * _param->SHEAR_MODULUS);
            const double del_U = U_old - U_new;
            
            const double del_P_n = _internal_vars[PRESSURE](i);
            double K1 = _param->K1;
            double del_P = -K1 * mu + sqrt(pow(K1 * mu + del_P_n,2)+2.*_param->BETA * K1 * del_U);
            _internal_vars[PRESSURE](i)=del_P;
            //}
        }

        /***********************************************************************
         * Combine deviatoric and volumetric stresses and apply stress rate
         **********************************************************************/

        sigma_view = alpha * s_tr - T_Id<TC> * p;
    }


    // void Update(const std::vector<QValues>& input, int i) override
    // {
    // }

    // void Resize(int n) override
    // {
    //     for (auto& qvalues : _internal_vars)
    //         qvalues.Resize(n);


    // }

};

template <Constraint TC> class JH2Improved: public RefLawInterface
{
public:
    //std::vector<QValues> _internal_vars;
    std::vector<Eigen::VectorXd> _internal_vars;
    //Eigen::VectorXd T_vol;
    //Eigen::MatrixXd T_dev;
    std::shared_ptr<JH2Parameters> _param;
    
    JH2Improved(std::shared_ptr<JH2Parameters> parameters, int n)
        : RefLawInterface(n), _param(parameters)
    {
        _internal_vars.resize(Q::LAST);
        //Accumulated plastic strain saved in LAMBDA
        _internal_vars[Q::LAMBDA] = Eigen::VectorXd::Zero(_n);
        //current density saved in rho
        _internal_vars[Q::RHO] = Eigen::VectorXd::Constant(_n, _param->RHO);

        _internal_vars[Q::DAMAGE] = Eigen::VectorXd::Zero(_n);
        _internal_vars[Q::PRESSURE] =Eigen::VectorXd::Zero(_n);

    }


    std::vector<Q> DefineInputs() const override
    {
        return {Q::GRAD_V, Q::SIGMA};
    }
    Eigen::VectorXd GetInternalVar(Q which)
    {
        return _internal_vars.at(which);
    }
    void EvaluateIP(int i, std::vector<Eigen::Ref<Eigen::VectorXd>>& inputs, double del_t) override
    {
        constexpr int L_size = Dim::G(TC)*Dim::G(TC);
        constexpr int sigma_size = Dim::StressStrain(TC);
        const FullTensor<TC> L_ = Eigen::Map<FullTensor<TC>>(inputs[Q::GRAD_V].segment<L_size>(i * L_size).data());
        const FullTensor<TC> D_ = 0.5 * (L_ + L_.transpose());
        
        const auto d_eps = TensorToMandel<TC>(D_);
        //const auto d_eps_vol = T_vol.dot(d_eps);
        
        auto sigma_view = inputs[Q::SIGMA].segment<sigma_size>(i * sigma_size);
        
        //const double lambda_n = _internal_vars[LAMBDA](i);
        //const double e_n = _internal_vars[E](i);
        const double D_n = _internal_vars[Q::DAMAGE](i);

        
        //auto stress = mandel_to_matrix(sigma_n);
        //stress += h * (stress * W_.transpose() + W_ * stress);

        /***********************************************************************
         * START CONSTITUTIVE MODEL HERE
         * 1) Calculate failure surface Y_failure
         * 2) Calculate Yield surface Y_yield = f(Y_failure)
         **********************************************************************/
        double p_n = - T_Vol<TC>.dot(sigma_view);// + _internal_vars[PRESSURE].GetScalar(i);
        auto s_n = T_Dev<TC> * sigma_view;
        MandelVector<TC> s_tr = s_n + 2. * _param->SHEAR_MODULUS * (T_Dev<TC> * d_eps) * del_t;
        const double s_tr_eq = sqrt(1.5 * s_tr.squaredNorm());
        const double d_eps_eq = sqrt((2./3.) * d_eps.squaredNorm());
        double alpha = 0.0;

        double p_s = p_n / _param->PHEL;
        double t_s = _param->T / _param->PHEL;
        double del_lambda = 0.0;
        double Y_yield = 0.0;
        //double dY_yield = 0.0;
        double Y_f;
        double Y_r;
        double rate_factor = 1.;


        Y_f = fmax(_param->A * pow(p_s + t_s, _param->N) * _param->SIGMAHEL, 0.0);
        Y_r = fmax(_param->B * pow(p_s,_param->M) * _param->SIGMAHEL,0.0);
        if (d_eps_eq >= _param->EPS0){
            rate_factor += _param->C * log(d_eps_eq/_param->EPS0);
        } 
        if (D_n == 0.0){
            Y_yield = Y_f*rate_factor;
        } else {
            Y_yield = (Y_f*(1.-D_n) + D_n * Y_r)*rate_factor;
        }
        if (s_tr_eq > Y_yield){
            const double e_p_f = fmax(_param->D1 * pow(p_s + t_s, _param->D2), _param->EFMIN);
            
            del_lambda =_param->MOGEL * (s_tr_eq-Y_yield) / (3.*_param->SHEAR_MODULUS);// + (Y_r-Y_f)/e_p_f);
            alpha = Y_yield/s_tr_eq;

            _internal_vars[Q::LAMBDA](i) += del_lambda;
            // Update damage variable or set to 1.
            _internal_vars[Q::DAMAGE](i) = fmin(D_n+del_lambda/e_p_f,1.0);
        
        } else {
            //elastic
            alpha = 1.0;
        }

        //Update deviatoric stress s
        //auto s = alpha * s_tr;
        /***********************************************************************
         * END CONSTITUTIVE MODEL HERE
         **********************************************************************/

        /***********************************************************************
         * UPDATE DENSITY
         * The density is updated using the explicit midpoint rule for the
         * deformation gradient.
         **********************************************************************/
/*         FullTensor<TC> factor_1 = FullTensor<TC>::Identity()-0.5*del_t*L_;
        FullTensor<TC> factor_2 = FullTensor<TC>::Identity()+0.5*del_t*L_;
        if (_internal_vars[RHO](i) == 0.0){
            _internal_vars[RHO](i)=_param->RHO * factor_1.determinant() / factor_2.determinant();
        } else {
            _internal_vars[RHO](i)*= factor_1.determinant() / factor_2.determinant();
        } */
        
        double f1 = del_t/2. * L_.trace();
        _internal_vars[Q::RHO](i) *= (1-f1)/(1+f1);
        /***********************************************************************
         * UPDATE ENERGY AND EOS
         **********************************************************************/
        
        const auto mu = _internal_vars[RHO](i)/_param->RHO -1.;
        
        const auto p = (mu > 0) ? _param->K1 * mu + _param->K2 * mu * mu + _param->K3 * mu * mu * mu + _internal_vars[Q::PRESSURE](i): _param->K1 * mu;
        
        const double D_new = _internal_vars[DAMAGE](i);
        if (D_new > D_n){
            const double Y_old = (D_n * Y_r + (1-D_n) * Y_f);// * _param->SIGMAHEL;
            const double Y_new = (D_new * Y_r + (1-D_new) * Y_f);// * _param->SIGMAHEL;
            const double U_old = (Y_old * Y_old) / (6. * _param->SHEAR_MODULUS);
            const double U_new = (Y_new * Y_new) / (6. * _param->SHEAR_MODULUS);
            const double del_U = U_old - U_new;
            
            const double del_P_n = _internal_vars[PRESSURE](i);
            double K1 = _param->K1;
            double del_P = -K1 * mu + sqrt(pow(K1 * mu + del_P_n,2)+2.*_param->BETA * K1 * del_U);
            _internal_vars[PRESSURE](i)=del_P;
            //}
        }

        /***********************************************************************
         * Combine deviatoric and volumetric stresses and apply stress rate
         **********************************************************************/

        sigma_view = alpha * s_tr - T_Id<TC> * p;
    }


    // void Update(const std::vector<QValues>& input, int i) override
    // {
    // }

    // void Resize(int n) override
    // {
    //     for (auto& qvalues : _internal_vars)
    //         qvalues.Resize(n);


    // }

};
