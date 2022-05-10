#pragma once
#include <dlfcn.h>
#include <exception>
#include <iostream>
#include <string>

namespace constants
{
// constants have internal linkage by default
const int ntens = 6;
const int dim = 3;
} // namespace constants

/*
 * Load a material following assumes that
 * `libname` is a shared library and includes
 *      void param0(double*);
 *      void eval(double*, double*);
 */

typedef void (*t_Param)(char[], bool*, int);
typedef void (*t_Eval)(double[], double[], double (*ddsdde)[constants::ntens], double*, double*, double*, double*,
                       double[], double[], double*, double[], double[], double[], double*, double*, double*, double*,
                       double*, char[], int*, int*, int*, int*, double[], int*, double[constants::dim],
                       double (*drot)[constants::dim], double*, double*, double (*dfgrd0)[constants::dim],
                       double (*dfgrd1)[constants::dim], int*, int*, int*, int*, int*, int*, int);


class UmatMaterial
{
public:
    UmatMaterial(std::string libName, std::string umatName, int numStateV, std::string fEval, std::string fParam = "")
        : _libName(libName)
        , _umatName(umatName)
        , _numStateV(numStateV)
    {
        _libHandle = dlopen(libName.c_str(), RTLD_LAZY);
        if (!_libHandle)
        {
            throw std::runtime_error("Cannot load " + libName + "!");
        }

        _f_eval = (t_Eval)dlsym(_libHandle, fEval.c_str());
        if (!_f_eval)
        {
            throw std::runtime_error("Cannot load function " + fEval + " from " + libName + "!");
        }

        if (fParam != "")
        {
            _f_param = (t_Param)dlsym(_libHandle, fParam.c_str());
            if (!_f_param)
            {
                throw std::runtime_error("Cannot load function " + fParam + " from " + libName + "!");
            }
        }
    }


    std::string Name()
    {
        return _umatName;
    }

    void Param(char name[], bool* initMat, int lenName)
    {
        _f_param(name, initMat, lenName);
    };

    int NumStatev()
    {
        return _numStateV;
    }

    void Eval(double stress[], double statev[], double (*ddsdde)[constants::ntens], double* sse, double* spd,
              double* scd, double* rpl, double ddsddt[], double drplde[], double* drpldt, double stran[],
              double dstran[], double time[], double* dtime, double* temp, double* dtemp, double* predef, double* dpred,
              char cmname[], int* ndi, int* nshr, int* ntens, int* nstatv, double props[], int* nprops,
              double coords[constants::dim], double (*drot)[constants::dim], double* pnewdt, double* celent,
              double (*dfgrd0)[constants::dim], double (*dfgrd1)[constants::dim], int* noel, int* npt, int* layer,
              int* kspt, int* kstep, int* kinc, int len)
    {
        _f_eval(stress, statev, ddsdde, sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran, time, dtime, temp,
                dtemp, predef, dpred, cmname, ndi, nshr, ntens, nstatv, props, nprops, coords, drot, pnewdt, celent,
                dfgrd0, dfgrd1, noel, npt, layer, kspt, kstep, kinc, len);
    }

private:
    const std::string _libName;
    const std::string _umatName;
    const int _numStateV;

    void* _libHandle;
    t_Eval _f_eval;
    t_Param _f_param;
};
