#pragma once
#include "interfaces.h"
#include "plasticity.h"
#include <tuple>
#include <eigen3/Eigen/Dense>
#include <math.h>
#include <complex>

enum RHT_PARAMETERS
{
    RHT_MID, //Material identification. A unique number or label not exceeding 8 characters must be specified.
    RHT_RO, //Mass density
    RHT_SHEAR, //Elastic shear modulus
    RHT_ONEMPA, //Unit conversion factor defining 1 Mpa in the pressure units used.
    RHT_EPSF, //Eroding plastic strain
    RHT_B0, //Parameter for polynomial EOS
    RHT_B1, //Parameter for polynomial EOS
    RHT_T1, //Parameter for polynomial EOS
    RHT_A,
    RHT_N,
    RHT_FC,
    RHT_FS_STAR,
    RHT_FT_STAR,
    RHT_Q0,
    RHT_B,
    RHT_T2,
    RHT_E0C,
    RHT_E0T,
    RHT_EC,
    RHT_ET,
    RHT_BETAC,
    RHT_BETAT,
    RHT_PTF,
    RHT_GC_STAR,
    RHT_GT_STAR,
    RHT_XI,
    RHT_D1,
    RHT_D2,
    RHT_EPM,
    RHT_AF,
    RHT_NF,
    RHT_GAMMA,
    RHT_A1,
    RHT_A2,
    RHT_A3,
    RHT_PEL,
    RHT_PCO,
    RHT_NP,
    RHT_ALPHA,
    RHT_GAMMAC,
    RHT_GAMMAT,
    RHT_Q1,
    RHT_LAST,
};

class RHT: public LawInterface
{
public:
    std::vector<double> _param;
    std::vector<QValues> _internal_vars_0;
    std::vector<QValues> _internal_vars_1;
    Eigen::VectorXd T_vol;
    Eigen::MatrixXd T_dev;
    double _mu;
    double _rho_0;
    std::unordered_map<std::string, double> parameters;
    RHT()
    {
        _param.resize(RHT_PARAMETERS::RHT_LAST);
        _internal_vars_0.resize(Q::LAST);
        _internal_vars_1.resize(Q::LAST);

        //_internal_vars_0[LAMBDA] = QValues(1);
        //_internal_vars_1[LAMBDA] = QValues(1);

        _internal_vars_0[DAMAGE] = QValues(1);
        _internal_vars_1[DAMAGE] = QValues(1);
        _internal_vars_0[KAPPA] = QValues(1);
        _internal_vars_1[KAPPA] = QValues(1);
        _internal_vars_0[LAMBDA] = QValues(1);
        _internal_vars_1[LAMBDA] = QValues(1);

        _internal_vars_0[E] = QValues(1);
        _internal_vars_1[E] = QValues(1);

        _internal_vars_0[POROSITY] = QValues(1);
        _internal_vars_1[POROSITY] = QValues(1);
        _internal_vars_0[RHO] = QValues(1);
        _internal_vars_1[RHO] = QValues(1);

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
        return _internal_vars_0.at(which).data;
    }
    double EOS(double rho, double e){
        eta = _param[RHT_ALPHA]*rho/(_internal_vars_0[POROSITY]*_param[RHT_RO])-1.;
        if (eta > 0.){
            return (_param[RHT_B0]+_param[RHT_B1]*eta)*internal_vars_0[POROSITY]*rho*e + _param[RHT_A1]*eta +_param[RHT_A2]*eta*eta + _param[RHT_A3]*eta*eta*eta;  
        } else {
            return _param[RHT_B0]*internal_vars_0[POROSITY]*rho*e + _param[RHT_T1]*eta +_param[RHT_T2]*eta*eta;
        }
    }
    std::complex<double> YFS(std::complex<double> p_s, std::complex<double> p_t_s, std::complex<double> fr)
    {
        if(3. * p_s.real() >= fr.real()){
            return _param[RHT_A] * pow(p_s - fr/3. +pow(fr/_param[RHT_A], 1./_param[RHT_N]),_param[RHT_N]);
        } else if(fr.real() > 3. * p_s.real() && 3.*p_s.real() >= 0.0){
            return fr * _param[RHT_FS_STAR]/_param[RHT_Q1] + 3.*p_s * (1.-_param[RHT_FS_STAR]/_param[RHT_Q1]);
        } else if(0. > 3.*p_s.real() && 3*p_s.real() >= 3.*p_t_s.real()){
            return fr * _param[RHT_FS_STAR]/_param[RHT_Q1] - 3.*p_s*(1./(_param[RHT_Q0]+_param[RHT_B]*p_s)-_param[RHT_FS_STAR]/(_param[RHT_Q1]*_param[RHT_FT_STAR])); 
        } else {
            return 0.0;
        }
    }
    std::complex<double> FC(std::complex<double> p_s,std::complex<double> p_c_s,std::complex<double> p_u_s)
    {
      if(p_s.real() >= p_c_s.real()) {
        return 0.0;
      } else if(p_c_s.real()>p_s.real() && p_s.real() >= p_u_s.real()) {
        return sqrt(1.-pow((p_s-p_u_s)/(p_c_s-p_u_s),2.0));
      } else {
        return 1.0;
      }
    }
    std::complex<double> FE(std::complex<double> p_s, std::complex<double> frc,std::complex<double> frt)
    {
      auto p3 = 3. * p_s.real();
      if(p3 >= frc.real()*_param[RHT_GC_STAR]){
        return _param[RHT_GC_STAR];
      } else if (frc.real() * _param[RHT_GC_STAR] > p3 && p3 >= -frt.real()*_param[RHT_FT_STAR]*_param[RHT_GT_STAR]) {
        auto nominator = (3.*p_s-frc * _param[RHT_GC_STAR]) * (_param[RHT_GT_STAR]-_param[RHT_GC_STAR]);
        auto denominator = (frc*_param[RHT_GT_STAR] + frt*_param[RHT_GT_STAR]*_param[RHT_FT_STAR]);
        return  _param[RHT_GC_STAR]-nominator/denominator;
      } else {
        return _param[RHT_GT_STAR];
      }
    }
    std::complex<double> FR(std::complex<double> p_s, std::complex<double> frc,std::complex<double> frt)
    {
      auto p3 = 3. * p_s.real();
      if(p3 >= frc.real()){
        return frc;
      } else if (frc.real() > p3 && p3 >= -frt.real()*_param[RHT_FT_STAR]) {
        return frc - (3.*p_s-frc)/(frc + frt *_param[RHT_FT_STAR]) * (frt-frc);
      } else {
        return frt;
      }
    }

    std::complex<double> FRC(std::complex<double> del_lam, double del_t)
    {
        auto d_lam = del_lam / del_t;
        if(d_lam.real() < _param[RHT_E0C]){
            return 1.
        } else if(d_lam.real() <= _param[RHT_EC]){
          return pow(d_lam/_param[RHT_E0C],_param[RHT_BETAC]);
        } else {
          return _param[RHT_GAMMAC] * pow(d_lam, 1./3.);
        }
    }
    std::complex<double> FRT(std::complex<double> del_lam, double del_t)
    {
        auto d_lam = del_lam / del_t;
        if(d_lam.real() < _param[RHT_E0T]){
            return 1.
        } else if(d_lam.real() <= _param[RHT_ET]){
          return pow(d_lam/_param[RHT_E0T],_param[RHT_BETAT]);
        } else {
          return _param[RHT_GAMMAT] * pow(d_lam, 1./3.);
        }
    }
    double R3(double theta, double p_s)
    {
        double c = cos(theta);
        double Q_ = _param[RHT_Q0] + _param[RHT_B] * p_s;
        double nom = 2.*(1.-Q_*Q_)*c +(2.*Q_-1.)*sqrt(4.*(1-Q_*Q_)*c*c+5.*Q_*Q_-4.*Q_);
        double denom = 4.*(1.-Q_*Q_)*c*c+(1.-2.*Q_)*(1.-2.*Q_);
        return nom/denom;
    }
    void Evaluate(const std::vector<QValues>& input, std::vector<QValues>& output, int i) override
    {
        int maxit = 10;
        Eigen::Matrix3d L_ = input[L].Get(i);
        Eigen::VectorXd sigma_n = input[SIGMA].Get(i);
        auto h = input[TIME_STEP].GetScalar(i);
        auto lambda = _internal_vars_0[LAMBDA].GetScalar(i);
        auto damage = _internal_vars_0[DAMAGE].GetScalar(i);
        auto eps_p_h = internal_vars_0[KAPPA].GetScalar(i);
        const auto D_ = 0.5 * (L_ + L_.transpose());
        const auto W_ = 0.5 * (L_ - L_.transpose());


        auto stress = mandel_to_matrix(sigma_n);
        stress += 0.5 * h * (stress * W_.transpose() + W_ * stress);

        /***********************************************************************
         * START CONSTITUTIVE MODEL HERE
         * 1) Calculate failure surface Y_failure
         * 2) Calculate Yield surface Y_yield = f(Y_failure)
         **********************************************************************/
        double p_n = T_vol.dot(sigma_n);
        auto s_n = T_dev * matrix_to_mandel(stress);
        auto s_tr = s_n + 2. * _param[RHT_SHEAR] * T_dev * matrix_to_mandel(D_ * h);
        double s_tr_eq = sqrt(1.5 * s_tr.transpose() * s_tr);
        double lode_angle = acos((27. * mandel_to_matrix(s_tr).determinant())/(2.*s_tr_eq));
        double p_c = _param[RHT_PCO]-(_param[RHT_PCO]-_param[RHT_PEL])*pow((_internal_vars_0[POROSITY].GetScalar(i)-1.)/(_param[RHT_ALPHA]-1.), 1./_param[RHT_NP]);
        double p_s = p_n / _param[RHT_FC];
        double p_c_s = p_c / _param[RHT_FC];
        double del_lam = 0.0;
        double complex_step = 1e-10;
        std::complex<double> ih(0.,complex_step);

        double r3 = R3(lode_angle, p_s); 

        auto Frc = FRC(del_lam + ih, h);
        auto Frt = FRT(del_lam + ih, h);
        auto Fr  = FR(p_s, Frc, Frt); 

        auto p_u_s = Frc*_param[RHT_GC_STAR]/3.;
        auto Fc = FC(p_s, p_c_s, p_u_s);
        auto Fe = FE(p_s, Frc, Frt);
        if(lambda == 0.0){
          ////no plastic flow so far
            auto gamma = Fe * Fc;
            auto Y_y = _param[RHT_FC]  * r3 * YFS(p_s/gamma, Fr)*gamma;
            if (s_tr_eq >= Y_y){
                //plastic flow initiated
                eps_p_h =Y_y.real()*(1.-Fc.real()*Fe.real())/(gamma.real()*3.*_param[RHT_SHEAR]*_param[RHT_XI]);
                auto eps_p_s = (del_lam + ih)/eps_p_h;
                double f = 0.0;
                double df = 0.0;
                int j = 0;
                do  {
                    //calculate yield surface with complex step
                    Frc = FRC(del_lam + ih, h);
                    Frt = FRT(del_lam + ih, h);
                    Fr  = FR(p_s, Frc, Frt); 
                    p_u_s = Frc*_param[RHT_GC_STAR]/3.+ _param[RHT_XI]*_param[RHT_SHEAR]*(del_lam+ih)/_param[RHT_FC];
                    Fc = FC(p_s, p_c_s, p_u_s);
                    Fe = FE(p_s, Frc, Frt);
                    gamma = eps_p_s + (1.-eps_p_s) * Fe * Fc;

                    Y_y = _param[RHT_FC]  * r3 * YFS(p_s/gamma, Fr)*gamma;

                    f = s_tr_eq - 3.*_param[RHT_SHEAR]*del_lam - Y_y.real();
                    df =  3.*_param[RHT_SHEAR] + Y_y.imag()/complex_step;
                    del_lambda += del_lambda - f/df;

                    j++;
                } while (abs(f)> 1e-10 && j < maxit);
                //TODO: update variables

            } else {
                //elastic
                //TODO update variables
            }
        } else if(lambda/eps_p_h < 1) {
            //plastic flow, but no damage so far
            auto eps_p_s = (del_lam + ih)/eps_p_h;
            auto gamma = eps_p_s + (1.-eps_p_s) * Fe * Fc;
            auto Y_y = _param[RHT_FC]  * r3 * YFS(p_s/gamma, Fr)*gamma;
            if (s_tr_eq >= Y_y){
            
            } else {
                //elastic
            }

        } else {
            //damage
        }
      //auto Y_f_star =
        //auto Y_f = f_c * Y_f_star * R_3;


      //stress += mandel_to_matrix(C * matrix_to_mandel(D * h));

        /***********************************************************************
         * END CONSTITUTIVE MODEL HERE
         **********************************************************************/
        stress += 0.5 * h * (stress * W_.transpose() + W_ * stress);

        /***********************************************************************
         * UPDATE DENSITY
         * The density is updated using the explicit midpoint rule for the
         * deformation gradient.
         **********************************************************************/
        auto factor_1 = Eigen::MatrixXd::Identity(3,3)-0.5*h*L_;
        auto factor_2 = Eigen::MatrixXd::Identity(3,3)+0.5*h*L_;
        _internal_vars_1[RHO].Set(_internal_vars_0[RHO].GetScalar(i) * factor_1.determinant() / factor_2.determinant(), i);
        /***********************************************************************
         * UPDATE ENERGY AND EOS
         **********************************************************************/

    }


    void Update(const std::vector<QValues>& input, int i) override
    {
        //TODO
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
