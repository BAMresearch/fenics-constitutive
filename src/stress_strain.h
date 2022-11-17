#pragma once
#include "interfaces.h"


template <Constraint TC>
MandelMatrix<TC> T_Dev;

template <Constraint TC>
MandelVector<TC> T_Vol;

template <Constraint TC>
MandelVector<TC> T_Id;

template <>
MandelMatrix<FULL> T_Dev<FULL> {
                {2./3., -1./3., -1./3., 0., 0., 0.},
                {-1./3., 2./3., -1./3., 0., 0., 0.},
                {-1./3., -1./3., 2./3., 0., 0., 0.},
                {0., 0., 0., 1., 0., 0.},
                {0., 0., 0., 0., 1., 0.},
                {0., 0., 0., 0., 0., 1.}};

template <>
MandelVector<FULL> T_Vol<FULL> {1./3.,1./3.,1./3.,0.,0.,0.};

template <>
MandelVector<FULL> T_Id<FULL> {1.,1.,1.,0.,0.,0.};

template <Constraint TC> 
void ApplyBBar(Eigen::Ref<Eigen::VectorXd> L, int n_per_cell)
{
    constexpr int l = Dim::G(TC)*Dim::G(TC);
    int n_cells = (L.size()/l)/n_per_cell;
    auto diag = (1./3.) * Eigen::Matrix<double, Dim::G(TC),1>::Ones(); 
    double d_eps_vol = 0.0;
    double trace;
    for(int i=0;i<n_cells;i++){
        d_eps_vol=0.0;
        for(int j=0;j<n_per_cell;j++){
            auto L_temp = Eigen::Map<FullTensor<TC>>(L.segment<l>((i * n_per_cell +j) * l).data());
            trace = L_temp.trace();
            L_temp.diagonal() -= trace * diag;
            d_eps_vol += trace;
        }
        d_eps_vol /= n_per_cell;
        for(int j=0;j<n_per_cell;j++){
            auto L_temp = Eigen::Map<FullTensor<TC>>(L.segment<l>((i * n_per_cell +j) * l).data());
            //trace = L_temp.trace();
            L_temp.diagonal() += d_eps_vol * diag;
        }
        
    }
}
