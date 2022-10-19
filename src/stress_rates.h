#include "interfaces.h"
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
    for(int i=0;i<n_gauss;i++){
        auto L_seg = L.segment<l>(i*l);
        FullTensor<TC> L_temp = Eigen::Map<FullTensor<TC>>(L_seg.data());
        FullTensor<TC> W_temp = 0.5 * (L_temp - L_temp.transpose());
        auto sigma_view = sigma.segment<stress_strain>(i*stress_strain);
        
        auto Wsig = W_temp * MandelToTensor<TC>(sigma_view);
        sigma_view += del_t * TensorToMandel<TC>(Wsig+Wsig.transpose());  
    }
}