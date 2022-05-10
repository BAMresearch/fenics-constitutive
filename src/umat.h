#pragma once
#include "interfaces.h"

#include <string>
#include <dlfcn.h>
using namespace std;

namespace constants
{
// constants have internal linkage by default
const int ntens = 6;
const int dim = 3;
} // namespace constants

typedef void (*t_Param)(char[], bool*, int);
typedef void (*t_Eval)(double[], double[], double (*ddsdde)[constants::ntens], double*, double*, double*, double*,
                       double[], double[], double*, double[], double[], double[], double*, double*, double*, double*,
                       double*, char[], int*, int*, int*, int*, double[], int*, double[constants::dim],
                       double (*drot)[constants::dim], double*, double*, double (*dfgrd0)[constants::dim],
                       double (*dfgrd1)[constants::dim], int*, int*, int*, int*, int*, int*, int);


// extern "C"
//{
// void param0_sdchabox_(char[], bool*, int);
// void param0_sdgdp_(char[], bool*, int);
// void param0_sdcrcry_(char[], bool*, int);
//
// void kusdchabox_(double[], double[], double (*ddsdde)[constants::ntens], double*, double*, double*, double*,
//                 double[], double[], double*, double[], double[], double[], double*, double*, double*, double*,
//                 double*, char[], int*, int*, int*, int*, double[], int*, double[constants::dim],
//                 double (*drot)[constants::dim], double*, double*, double (*dfgrd0)[constants::dim],
//                 double (*dfgrd1)[constants::dim], int*, int*, int*, int*, int*, int*, int);
// void kusdgdp_(double[], double[], double (*ddsdde)[constants::ntens], double*, double*, double*, double*,
// double[],
//              double[], double*, double[], double[], double[], double*, double*, double*, double*, double*,
//              char[], int*, int*, int*, int*, double[], int*, double[constants::dim], double
//              (*drot)[constants::dim], double*, double*, double (*dfgrd0)[constants::dim], double
//              (*dfgrd1)[constants::dim], int*, int*, int*, int*, int*, int*, int);
// void kusdcrcry_(double[], double[], double (*ddsdde)[constants::ntens], double*, double*, double*, double*,
//                double[], double[], double*, double[], double[], double[], double*, double*, double*, double*,
//                double*, char[], int*, int*, int*, int*, double[], int*, double[constants::dim],
//                double (*drot)[constants::dim], double*, double*, double (*dfgrd0)[constants::dim],
//                double (*dfgrd1)[constants::dim], int*, int*, int*, int*, int*, int*, int);
// void umat_(double[], double[], double (*ddsdde)[constants::ntens], double*, double*, double*, double*, double[],
//           double[], double*, double[], double[], double[], double*, double*, double*, double*, double*, char[],
//           int*, int*, int*, int*, double[], int*, double[constants::dim], double (*drot)[constants::dim],
//           double*, double*, double (*dfgrd0)[constants::dim], double (*dfgrd1)[constants::dim], int*, int*, int*,
//           int*, int*, int*, int);
//}

class Umat : public MechanicsLaw
{
public:
    Umat(Constraint c, std::string cmname, std::string libName, int nstatv, std::string fEval, std::string fParam,
         const std::vector<double>* EulerAngles = 0)
        : MechanicsLaw(c)
        , _ntens(Dim::Q(c))
        , _stranPrev(Dim::Q(c))
        , _stressPrev(Dim::Q(c))
        , _cmname(cmname)
        , _nstatv(nstatv)
    {
        bool initMat = true;

        // initiate state variables
        _statevPrev = QValues(_nstatv);
        _statevEvaluate = QValues(_nstatv);

        // get orientation
        //_EulerAngles[0] = 0;
        //_EulerAngles = 0;

        if ((*EulerAngles).empty() == false)
        {
            if ((*EulerAngles).size() != 3)
            {
                std::cout << "ERROR: Orientation needs three Euler angles in °" << std::endl;
                throw std::exception();
            }
            // _EulerAngles = &((*EulerAngles).at(0));
            // 2710 std::copy((*EulerAngles).begin(), (*EulerAngles).end(), _EulerAngles);
            _EulerAngles = *EulerAngles;
            // std::cout << (*EulerAngles).at(0) << std::endl;
        }

        if (access(libName.c_str(), F_OK) == -1)
        {
            throw std::runtime_error("Library at " + libName + " does not exist!");
        }
        // load a shared fortran library
        _libHandle = dlopen(libName.c_str(), RTLD_LAZY);
        if (!_libHandle)
        {
            throw std::runtime_error("Cannot load library " + libName + "!");
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
            _f_param(&_cmname[0], &initMat, _cmname.length());
        }

        // not all constraints are implemented
        // PLANE_STRESS requires another stiffness matrix and must be transfered
        // to UMAT with ntens = 4!!!
        switch (c)
        {
        case PLANE_STRAIN:
            break;
        case FULL:
            break;
        default:
            std::cout << "ERROR: Constraint type " << c << " is not implemented..." << std::endl;
            throw std::exception();
        }
    }
    // int getNumberStatev()
    //{
    //    if (_cmname == "SDCHABOX")
    //    {
    //        return 29;
    //    }
    //    else if (_cmname == "SDGDP")
    //    {
    //        return 45;
    //    }
    //    else if (_cmname == "SDCRCRY")
    //    {
    //        return 37;
    //    }
    //    else if (_cmname == "UMAT")
    //    {
    //        return 0;
    //    }
    //    else
    //    {
    //        std::cout << "ERROR at getNumberStatev: Unknown UMAT " << _cmname << std::endl;
    //        throw std::exception();
    //    }
    //    }
    void Resize(int n) override
    {
        _statevPrev.Resize(n);
        _statevEvaluate.Resize(n);
        _stranPrev.Resize(n);
        _stressPrev.Resize(n);
    }

    // converts the order of strain/stress components in VectorXd to the order in array for UMAT
    void ConvertVoigtXd2arr(const Eigen::VectorXd& valueXd, double* value)
    {
        assert(valueXd.rows() == _ntens);

        if (_ntens == 6)
        {
            // FULL
            for (int j = 0; j != 3; j++)
            {
                value[j] = valueXd(j);
            }
            value[3] = valueXd(5);
            value[4] = valueXd(4);
            value[5] = valueXd(3);
            // for ( int j = 0; j != 6; j++ )
            // {
            //   std::cout << j << " " << value[j] << " " << valueXd(j) << std::endl;
            // }
        }
        else if (_ntens == 3)
        {
            // PLANE_STRAIN
            value[0] = valueXd(0); // eps11
            value[1] = valueXd(1); // eps22
            value[2] = 0.; // eps33
            value[3] = valueXd(2); // 2*eps12
            value[4] = 0.; // 2*eps13
            value[5] = 0.; // 2*eps23

            //       for ( int j = 0; j != 3; j++ )
            // 	  {
            // 	    value[j] = valueXd(j);
            //  	  }
        }
        //   std::cout << " what is value " << value << std::endl;
    }

    // converts the order of strain/stress components in UMAT array to VectorXd in FeniCS
    Eigen::VectorXd ConvertVoigtArr2vectorXd(const double value[], const int sizeOfValue)
    {
        assert(sizeOfValue == _ntens);

        Eigen::VectorXd valueXd(_ntens);

        if (_ntens == 6)
        {
            // FULL
            for (int j = 0; j != 3; j++)
            {
                valueXd(j) = value[j];
            }
            valueXd(3) = value[5];
            valueXd(4) = value[4];
            valueXd(5) = value[3];
        }
        else if (_ntens == 3)
        {
            // PLANE_STRAIN
            valueXd(0) = value[0]; // eps11
            valueXd(1) = value[1]; // eps22
            valueXd(2) = value[3]; // 2*eps12

            // for ( int j = 0; j != 3; j++ )
            //  {
            //  valueXd(j) = value[j];
            //}
        }

        return valueXd;
    }

    Eigen::MatrixXd ConvertDdsdde2matrixXd(const double (*ddsdde)[constants::ntens])
    {
        assert(constants::ntens < _ntens);

        Eigen::MatrixXd ddsddeXd(_ntens, _ntens);

        if (_ntens == 6)
        {
            // FULL
            for (int i = 0; i != _ntens; i++)
            {
                for (int j = 0; j != _ntens; j++)
                {
                    ddsddeXd(i, j) = ddsdde[j][i]; // c++ array = tr(fortran array)
                }
            }
            ddsddeXd.row(3).swap(ddsddeXd.row(5)); // swap 3th and 5th cols and rows
            ddsddeXd.col(3).swap(ddsddeXd.col(5)); // UMAT to fenics convention
        }
        else if (_ntens == 3)
        {
            // PLANE_STRAIN
            ddsddeXd(0, 0) = ddsdde[0][0];
            ddsddeXd(1, 1) = ddsdde[1][1];
            ddsddeXd(2, 2) = ddsdde[3][3];

            ddsddeXd(0, 1) = ddsdde[1][0];
            ddsddeXd(0, 2) = ddsdde[3][0];
            ddsddeXd(1, 2) = ddsdde[3][1];

            ddsddeXd(1, 0) = ddsddeXd(0, 1);
            ddsddeXd(2, 0) = ddsddeXd(0, 2);
            ddsddeXd(2, 1) = ddsddeXd(1, 2);

            // for ( int i = 0; i != _ntens; i++){
            //     for ( int j = 0; j != _ntens; j++){
            // 	ddsddeXd(i,j) = ddsdde[j][i];     // c++ array = tr(fortran array)
            //     }
            //   }
        }

        return ddsddeXd;
    }

    void Update(const Eigen::VectorXd& strain, int i) override
    {
        Eigen::VectorXd statevXd(_nstatv), stressXd(_ntens);

        stressXd = Evaluate(strain, i).first; // the routine overwrites _statevEvaluate
        _statevPrev.Set(_statevEvaluate.Get(i), i);
        _stranPrev.Set(strain, i);
        _stressPrev.Set(stressXd, i);

        //      std::cout << " OUTPUT FROM RESIZE =================" << std::endl;
        //      std::cout << _stressPrev.Get(i) << std::endl;
    }

    Eigen::VectorXd statev() const
    {
        return _statevPrev.data;
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> Evaluate(const Eigen::VectorXd& strain, int i) override
    {
        // for plain_strain we transform a strain vectors of length 3 to the full vectors of length 6
        // and send to abaqus; therefore for plain_strain and full, the strains and stresses have the
        // length of constant::ntens=6; once we convert the abaqus stress of length 6 to the fenics stress
        // of the length 6 for full or the length 3 for plain_strain, we need to provide the fenics
        int ntens = constants::ntens;
        int nprops;
        int ndi, nshr, noel, npt, layer, kspt, kstep, kinc;
        double stress[ntens], stran[ntens], dstran[ntens], ddsddt[ntens], drplde[ntens];
        double ddsdde[6][6], drot[3][3], dfgrd0[3][3], dfgrd1[3][3];

        double statev[_nstatv], time[2], coords[3];

        double sse, spd, scd, pnewdt, rpl, drpldt, dtime, temp, dtemp, predef, dpred, celent;

        temp = 973.15; // 400.;
        // dtime = 0.002;
        // time[0] = 0.024; time[1] = 0.024;
        time[0] = GetTime().first;
        time[1] = GetTime().first;
        dtime = GetTime().second - GetTime().first;

        // get stress, stran and statev from the history variables
        for (int j = 0; j != _nstatv; j++)
        {
            statev[j] = _statevPrev.Get(i)(j);
        }
        ConvertVoigtXd2arr(_stranPrev.Get(i), stran);
        ConvertVoigtXd2arr(_stressPrev.Get(i), stress);

        ConvertVoigtXd2arr(strain - _stranPrev.Get(i), dstran);

        // 2210      char cmname[strlen(constants::cmname)];
        // 1008      bool initMat = true;

        //     assign constants::cmname to cmname;
        //     std::copy(constants::cmname, constants::cmname + strlen(constants::cmname), cmname);
        // 2210 strcpy(cmname, constants::cmname);
        // char cmname[_cmname.length()];
        // strcpy(cmname, _cmname.c_str());
        // 2210 char cmname[strlen(_cmname)];
        // 2210 strcpy(cmname, _cmname);
        // 1008      param0_sdchabox_(cmname,&initMat,strlen(cmname));
        //      std::cout << "... after param0_" << std::endl;

        // get orientation
        // the props are used to transfer the Euler angles only!
        if (_EulerAngles.empty())
        {
            nprops = 0;
        }
        else
        {
            if (_EulerAngles.size() == 3)
            {
                nprops = 3;
            }
            else
            {
                nprops = 0;
            }
        }

        double props[nprops];
        if (_EulerAngles.size() == 3 and nprops == 3)
        {
            std::copy(_EulerAngles.begin(), _EulerAngles.end(), props);
        }

        _f_eval(stress, statev, ddsdde, &sse, &spd, &scd, &rpl, ddsddt, drplde, &drpldt, stran, dstran, time, &dtime,
                &temp, &dtemp, &predef, &dpred, &_cmname[0], &ndi, &nshr, &ntens, &_nstatv, props, &nprops, coords,
                drot, &pnewdt, &celent, dfgrd0, dfgrd1, &noel, &npt, &layer, &kspt, &kstep, &kinc, _cmname.length());

        Eigen::VectorXd statevXd(_nstatv);
        for (int j = 0; j != _nstatv; j++)
        {
            statevXd(j) = statev[j];
        }

        _statevEvaluate.Set(statevXd, i);

        return {ConvertVoigtArr2vectorXd(stress, _ntens), ConvertDdsdde2matrixXd(ddsdde)};
    }

private:
    const int _ntens;
    int _nstatv;

    // history variables
    QValues _statevPrev;
    QValues _statevEvaluate;
    QValues _stranPrev;
    QValues _stressPrev;

    // constitutive law name
    std::string _cmname;

    // orientation, optional
    std::vector<double> _EulerAngles;

    void* _libHandle;
    t_Eval _f_eval;
    t_Param _f_param;
};
