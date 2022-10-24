#pragma once
#include "interfaces.h"
#include "linear_elastic.h"


template <Constraint TC> class HypoElastic : public RefLawInterface
{
    /***************************************************************
    *** This class is supposed to be used for explicit dynamics 
    *** calculations together with an objective stress rate.
    *** It is assumed that the stresses have already been rotated.
    ****************************************************************/
public:
    MandelMatrix<TC> _C;
    
    HypoElastic(double E, double nu, int n)
        : RefLawInterface(n)
    {
        _C = C<TC>(E, nu);
    }
    std::vector<Q> DefineInputs() const override
    {
        std::vector<Q> inputs = {Q::GRAD_V, Q::SIGMA};
        return inputs;
    }
    void EvaluateIP(int i, std::vector<Eigen::Ref<Eigen::VectorXd>>& inputs, double del_t) override
    {
        //preparation
        constexpr int dim = Dim::StressStrain(TC);
        constexpr int dim_L = Dim::G(TC)*Dim::G(TC);
        auto L_flat = inputs[Q::GRAD_V].segment<dim_L>(i*dim_L);
        FullTensor<TC> L = Eigen::Map<FullTensor<TC>>(L_flat.data());
        auto D = 0.5*(L+L.transpose());
        auto del_eps = del_t * TensorToMandel<TC>(D);
        
        auto sigma = inputs[Q::SIGMA].segment<dim>(i*dim);
        
        //The actual stress update
        sigma += _C * del_eps;

    }
};

