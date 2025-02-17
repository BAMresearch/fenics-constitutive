#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <eigen3/Eigen/Core>
#include <filesystem>
#include <string>
#include <iostream>
#include "umat.h"

// Get the current working directory
std::filesystem::path cwd = std::filesystem::current_path();

// Go to /build
std::string libpath = cwd.string() + "/src/linela/umat/build/umat_linear_elastic.so";

Eigen::Vector<double, 6> voigt_strain_from_grad_u(const Eigen::Matrix3d &grad_u)
{
    Eigen::Vector<double, 6> strain;
    strain(0) = grad_u(0, 0);
    strain(1) = grad_u(1, 1);
    strain(2) = grad_u(2, 2);
    strain(3) = (grad_u(0, 1) + grad_u(1, 0));
    strain(5) = (grad_u(1, 2) + grad_u(2, 1));
    strain(4) = (grad_u(0, 2) + grad_u(2, 0));
    return strain;
}

template <int HISTORY_DIM>
class Umat3D
{
public:
    Umat3D(std::vector<double> props, std::map<std::string, std::string> string_props)
        : _cmname(string_props["cmname"]), _props(props), _libHandle(string_props["libname"])
    {
        // init for labtools
        _f_eval = _libHandle.Load<t_Eval>(string_props["fEval"]);
        _f_param = _libHandle.Load<t_Param>(string_props["fParam"]);
        _f_param(&_cmname[0], _cmname.length());

    }

    Umat3D(double E, double nu) : _libHandle(libpath)
    {
        // init for linear elastic example
        _f_eval = _libHandle.Load<t_Eval>("umat_");
        _cmname = "UMAT";
        _props = {E, nu};
    }
    void evaluate(
        double time_start,
        double del_t,
        const Eigen::Ref<const Eigen::VectorXd> &del_grad_u,
        Eigen::Ref<Eigen::VectorXd> &stress,
        Eigen::Ref<Eigen::VectorXd> &tangent,
        std::map<std::string, Eigen::Ref<Eigen::VectorXd>> &history)
    {
        const int grad_u_dim = constants::dim * constants::dim;
        int n_ip = del_grad_u.size() / (grad_u_dim);

        auto umat_history = history.at("umat_history");
        auto umat_stran = history.at("umat_stran");
        
        // initialize arguments for UMAT call
        const int n_tangent = constants::ntens * constants::ntens;

        int nprops = _props.size();
        int nstatv = HISTORY_DIM;

        int ndi, nshr, noel, npt, layer, kspt, kstep, kinc;
        double sse, spd, scd, pnewdt, rpl, drpldt, dtime, temp, dtemp, predef, dpred, celent;
        double stress_array[constants::ntens], stran[constants::ntens], dstran[constants::ntens], ddsddt[constants::ntens], drplde[constants::ntens];
        double ddsdde[constants::ntens][constants::ntens], drot[constants::dim][constants::dim], dfgrd0[constants::dim][constants::dim], dfgrd1[constants::dim][constants::dim];
        double statev[HISTORY_DIM], time[2], coords[constants::dim];
        time[0] = time_start;
        time[1] = time_start;

        dtime = del_t;
        temp = 973.15;
        dtemp = 0.0;
        std::string cmname = _cmname;

        int ntens = constants::ntens; // needed for UMAT call

        for (int ip = 0; ip < n_ip; ip++)
        {
            //The conversion from the flat grad_u to a 3x3 matrix is actually not correct because
            //of different memory layouts (row vs column), but it works for the determination of the strain
            Eigen::Vector<double, constants::ntens> del_strain = voigt_strain_from_grad_u(del_grad_u.segment<grad_u_dim>(grad_u_dim * ip).reshaped(constants::dim, constants::dim));
            
            // Convert Stress and strain in Mandel notation to Abaqus/Voigt notation
            voigt_strain_to_arr(del_strain, dstran);
            mandel_stress_to_arr(stress.segment<constants::ntens>(constants::ntens * ip), stress_array);

            // get statev at _time_start
            Eigen::Vector<double, HISTORY_DIM> history_vector = umat_history.segment<HISTORY_DIM>(HISTORY_DIM * ip);
            for (int i = 0; i < HISTORY_DIM; i++)
            {
                statev[i] = history_vector(i);
            }

            // add strain increment to total strain
            for (int i = 0; i < constants::ntens; i++)
            {
                stran[i] = umat_stran[ip * constants::ntens + i];
                umat_stran[ip * constants::ntens + i] = stran[i] + dstran[i];
            }

            _f_eval(stress_array, statev, ddsdde, &sse, &spd, &scd, &rpl, ddsddt, drplde, &drpldt, stran, dstran, time, &dtime,
                    &temp, &dtemp, &predef, &dpred, &cmname[0], &ndi, &nshr, &ntens, &nstatv, &_props[0], &nprops, coords,
                    drot, &pnewdt, &celent, dfgrd0, dfgrd1, &noel, &npt, &layer, &kspt, &kstep, &kinc, cmname.length());

            umat_history.segment<HISTORY_DIM>(HISTORY_DIM * ip) = Eigen::Map<Eigen::VectorXd>(&statev[0], HISTORY_DIM);
            stress.segment<constants::ntens>(constants::ntens * ip) = voigt_stress_arr_to_mandel_stress(stress_array);

            tangent.segment<n_tangent>(n_tangent * ip) = voigt_tangent_arr_to_mandel_tangent_flat(ddsdde);
        }
    }


    std::map<std::string, int> history_dim()
    {
        return {{"umat_history", HISTORY_DIM}, {"umat_stran", constants::ntens}};
    }

    void voigt_strain_to_arr(const Eigen::Vector<double, constants::ntens> &value_eig, double *value) const
    {
        value[0] = value_eig(0);
        value[1] = value_eig(1);
        value[2] = value_eig(2);
        value[3] = value_eig(3);
        value[4] = value_eig(4);
        value[5] = value_eig(5);
    }
    void mandel_strain_to_arr(const Eigen::Vector<double, constants::ntens> &value_eig, double *value) const
    {
        constexpr double sqrt2 = 1.41421356237309504880168872420969808;
        value[0] = value_eig(0);
        value[1] = value_eig(1);
        value[2] = value_eig(2);
        value[3] = sqrt2 * value_eig(3);
        value[4] = sqrt2 * value_eig(4);
        value[5] = sqrt2 * value_eig(5);
    }
    void mandel_stress_to_arr(const Eigen::Vector<double, constants::ntens> &value_eig, double *value) const
    {
        constexpr double frac_sqrt2 = 0.707106781186547524400844362104849039;
        value[0] = value_eig(0);
        value[1] = value_eig(1);
        value[2] = value_eig(2);
        value[3] = frac_sqrt2 * value_eig(3);
        value[4] = frac_sqrt2 * value_eig(4);
        value[5] = frac_sqrt2 * value_eig(5);
    }
    Eigen::Vector<double, constants::ntens> voigt_stress_arr_to_mandel_stress(const double value[constants::ntens]) const
    {
        constexpr double sqrt2 = 1.41421356237309504880168872420969808;
        Eigen::Vector<double, constants::ntens> value_eig;
        value_eig(0) = value[0];
        value_eig(1) = value[1];
        value_eig(2) = value[2];
        value_eig(3) = sqrt2 * value[3];
        value_eig(4) = sqrt2 * value[4];
        value_eig(5) = sqrt2 * value[5];
        return value_eig;
    }
    Eigen::Vector<double, constants::ntens * constants::ntens> voigt_tangent_arr_to_mandel_tangent_flat(double (*ddsdde)[constants::ntens]) const
    {
        // converts the UMAT matrix to a flat row-major storage order and convert to
        // Mandel notation
        constexpr double sqrt2 = 1.41421356237309504880168872420969808;
        constexpr int n_tangent = constants::ntens * constants::ntens;
        double factor = 1.0;
        Eigen::Vector<double, n_tangent> tangent_flat;
        for (int i = 0; i < constants::ntens; i++)
        {
            for (int j = 0; j < constants::ntens; j++)
            {
                factor = 1.0;
                if (i > 2)
                {
                    factor *= sqrt2;
                }
                if (j > 2)
                {
                    factor *= sqrt2;
                }
                // factor is 1.0 for the D11 3x3 block, sqrt2 for the D12 3x3 block
                // sqrt2 for the D21 3x3 block, and 2.0 for the D22 3x3 block
                tangent_flat(i * constants::ntens + j) = factor * ddsdde[j][i]; // c++ array = tr(fortran array)
            }
        }

        return tangent_flat;
    }
protected:
    LibHandle _libHandle;
    t_Param _f_param;
    t_Eval _f_eval;

    // constitutive law name
    std::string _cmname;

    std::vector<double> _props;
};

namespace py = pybind11;

PYBIND11_MODULE(umat, m)
{
    m.doc() = "";
    py::class_<Umat3D<0>> elasticity_3d(m, "Elasticity3D");
    elasticity_3d.def(py::init<double, double>(), py::arg("E"), py::arg("nu"));
    elasticity_3d.def("evaluate", &Umat3D<0>::evaluate);
    elasticity_3d.def("history_dim", &Umat3D<0>::history_dim);


    py::class_<Umat3D<29>> umat_3d_29(m, "Umat3D29");
    umat_3d_29.def(py::init<std::vector<double>, std::map<std::string, std::string>>(), py::arg("parameters"), py::arg("string parameters"));
    umat_3d_29.def("evaluate", &Umat3D<29>::evaluate);
    umat_3d_29.def("history_dim", &Umat3D<29>::history_dim);
 

    py::class_<Umat3D<37>> umat_3d_37(m, "Umat3D37");
    umat_3d_37.def(py::init<std::vector<double>, std::map<std::string, std::string>>(), py::arg("parameters"), py::arg("string parameters"));
    umat_3d_37.def("evaluate", &Umat3D<37>::evaluate);
    umat_3d_37.def("history_dim", &Umat3D<37>::history_dim);

}
