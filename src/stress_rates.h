#include "interfaces.h"
#include <iostream>
//#include <cmath>
template<Constraint TC>
FullTensor<TC> MandelToTensor(const MandelVector<TC> &sigma);

template<> FullTensor<FULL> MandelToTensor<FULL>(const MandelVector<FULL> &sigma)
{
    constexpr double factor = 1./1.4142135623730951; //known at compile time
    FullTensor<FULL> output;
    output << 
        sigma(0), factor * sigma(5), factor * sigma(4),
        factor * sigma(5), sigma(1), factor * sigma(3),
        factor * sigma(4), factor * sigma(3), sigma(2);
    return output;

}

template<Constraint TC>
MandelVector<TC> TensorToMandel(const FullTensor<TC> &sigma);

template<> MandelVector<FULL> TensorToMandel<FULL>(const FullTensor<FULL> &sigma)
{
    constexpr double factor = 1.4142135623730951; //known at compile time
    MandelVector<FULL> output;
    output << 
        sigma(0,0), sigma(1,1), sigma(2,2), factor * sigma(1,2), factor * sigma(0,2), factor * sigma(0,1);
    return output;
}

template <Constraint TC> 
void JaumannRotate(Eigen::Ref<Eigen::VectorXd> L, Eigen::Ref<Eigen::VectorXd> sigma, double del_t)
{
    const int l = Dim::G(TC)*Dim::G(TC);
    const int stress_strain = Dim::StressStrain(TC);
    int n_gauss = L.size()/l;
    FullTensor<TC> L_temp; 
    FullTensor<TC> W_temp; 
    for(int i=0;i<n_gauss;i++){
        L_temp = Eigen::Map<FullTensor<TC>>(L.segment<l>(i*l).data());
        W_temp = 0.5 * (L_temp - L_temp.transpose());
        auto sigma_view = sigma.segment<stress_strain>(i*stress_strain);
        W_temp *= MandelToTensor<TC>(sigma_view);
        sigma_view += del_t * TensorToMandel<TC>(W_temp+W_temp.transpose());  
    }
}
template <Constraint TC> 
void JaumannRotateFast(Eigen::Ref<Eigen::VectorXd> L, Eigen::Ref<Eigen::VectorXd> sigma, double del_t);

template<>
void JaumannRotateFast<FULL>(Eigen::Ref<Eigen::VectorXd> L, Eigen::Ref<Eigen::VectorXd> sigma, double del_t)
{
    //Apply the rotations directly on the mandel form of the stress. Yust trust me on the formulas;)
    //Approximately twice as fast as the other version.
    const int l = Dim::G(FULL)*Dim::G(FULL);
    const int stress_strain = Dim::StressStrain(FULL);
    int n_gauss = L.size()/l;
    const double root = 1.4142135623730951;

    FullTensor<FULL> L_temp; 
    //FullTensor<FULL> W_temp; 
    for(int i=0;i<n_gauss;i++){
        L_temp = Eigen::Map<FullTensor<FULL>>(L.segment<l>(i*l).data());
        auto W = 0.5 * (L_temp - L_temp.transpose());
        auto sigma_view = sigma.segment<stress_strain>(i*stress_strain);
        MandelVector<FULL> s = sigma_view;
        //just trust me on those formulas
        sigma_view(0) += del_t*(root*(s(4)*W(0,2)+s(5)*W(0,1)));
        sigma_view(1) += del_t*(root*(s(3)*W(1,2)-s(5)*W(0,1)));
        sigma_view(2) += del_t*(-root*(s(3)*W(1,2)+s(4)*W(0,2)));
        sigma_view(3) += del_t*(root*W(1,2)*(s(2)-s(1)) - s(4)*W(0,1) - s(5)*W(0,2));
        sigma_view(4) += del_t*(root*W(0,2)*(s(2)-s(0)) + s(3)*W(0,1) - s(5)*W(1,2));
        sigma_view(5) += del_t*(root*W(0,1)*(s(1)-s(0)) + s(3)*W(0,2) + s(4)*W(1,2));
        
        //sigma_view(0) += root * sigma_copy(4) * W_temp(0,2)+
    }
}