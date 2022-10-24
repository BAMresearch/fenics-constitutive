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