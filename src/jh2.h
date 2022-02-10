#pragma once
#include "interfaces.h"
#include "plasticity.h"
#include <tuple>
#include <eigen3/Eigen/Dense>
#include <math.h>
#include <complex>

enum JH2_PARAMETERS
{
    JH2_RHO,
    JH2_A,
    JH2_B,
    JH2_C,
    JH2_M,
    JH2_M,
    JH2_SFMAX,
    JH2_LAST,
};

class JH2: public LawInterface
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
    JH2()
    {
        _param.resize(RHT_PARAMETERS::RHT_LAST);
        _internal_vars_0.resize(Q::LAST);
        _internal_vars_1.resize(Q::LAST);

        //_internal_vars_0[LAMBDA] = QValues(1);
        //_internal_vars_1[LAMBDA] = QValues(1);

        _internal_vars_0[DAMAGE] = QValues(1);
        _internal_vars_1[DAMAGE] = QValues(1);
        _internal_vars_0[LAMBDA] = QValues(1);
        _internal_vars_1[LAMBDA] = QValues(1);

        _internal_vars_0[E] = QValues(1);
        _internal_vars_1[E] = QValues(1);

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
        eta = rho/_param[JH2_RHO]-1.;
        // case 1: no volumetric plastic strain
        if (eta > 0.){
            return _param[JH2_K1] 
        } else {

        }
    }
    std::complex<double> Yield(double p, double lam, double damage, std::complex<double> del_lam)
    {
        auto f1 = _param[JH2_A] * 
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
        
        double del_lam = 0.0;
        double complex_step = 1e-10;
        std::complex<double> ih(0.,complex_step);

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
