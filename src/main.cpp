#include <pybind11/eigen.h>
#include "pybind11/pybind11.h"
#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"

#include <eigen3/Eigen/Dense>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include "interfaces.h"
#include "linear_elastic.h"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"
#include <iostream>
#include <numeric>
#include <cmath>
namespace py = pybind11;

// Examples


inline double scalar_func(double i, double j)
{
    return std::sin(i) + std::cos(j);
}

// Python Module and Docstrings
class Test
{
public:
    
    Test(xt::pyarray<double> &a, double s)
    //:_internal{a}{
    {
        _internal = &a;
        _s = s;
    }
    void DoSomething()
    {
        std::cout << _internal;

        //auto sum = _internal;
        //return sum(0);
    }
    void DoSomethingWith(xt::pyarray<double> &a)
    {
        a*=_s;
    }
    xt::pyarray<double>* _internal;
    double _s;
    //xt::pyarray<double>   _internal;
};
class Test2
{
public:
    xt::pyarray<double> _internal;
    
    double DoSomething()
    {
        auto sum = xt::sum(_internal);
        return sum(0);
    }
};

//class TestEigen
//{
//public:
    //Eigen::VectorXd _internal;
    
    //TestEigen(Eigen::Ref<Eigen::MatrixXd> a)
    //{
        //_internal  = a;
    //}
    //double DoSomething()
    //{
        //auto sum = _internal.sum();
        //return sum;
    //}
//};
inline xt::pytensor<double, 1>& scale_xtensor(xt::pytensor<double, 1>& v) {
    v*=1.000000001;
    return v;
}
inline xt::pyarray<double>& scale_xarray(xt::pyarray<double>& v) {
    v*=1.000000001;
    return v;
}
xt::pyarray<double> scale_xt_copy(xt::pyarray<double> v) {
    return v*1.000000001;
}
void test_views(xt::pytensor<double, 1>& v)
{
    //auto slice = xt::reshape_view(xt::view(v, xt::range(2,6)),{2,2});
    auto slice = GetQuadratureView(v,0,2,2);
    xt::pyarray<double> insert{{42.,42.},{42.,42.}};
    slice = insert;

}
void scale_by_2(Eigen::Ref<Eigen::VectorXd> v) {
    v *= 2;
}
void test_views_eigen(Eigen::Ref<Eigen::VectorXd> v, Eigen::Ref<Eigen::VectorXd> insert) {
    auto segment = v.segment<4>(3);
    segment = insert;
}
PYBIND11_MODULE(cofe, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        Constitutive models for fenicsx

        .. currentmodule:: cofe

        .. autosummary::
           :toctree: _generate

           example1
           example2
           readme_example1
           vectorize_example1
    )pbdoc";

    //m.def("example1", example1, "Return the first element of an array, of dimension at least one");
    //m.def("example2", example2, "Return the the specified array plus 2");

    //m.def("readme_example1", readme_example1, "Accumulate the sines of all the values of the specified array");

    m.def("vectorize_example1", xt::pyvectorize(scalar_func), "Add the sine and and cosine of the two specified values");
    m.def("scale_xa", scale_xarray);
    m.def("scale_xt", scale_xtensor);
    m.def("scale_copy", scale_xt_copy);
    m.def("test_views", test_views);
    m.def("test_views_eigen", test_views_eigen);
    m.def("scale_eigen", scale_by_2);

    pybind11::class_<Test, std::shared_ptr<Test>> test(m, "Test");
    test.def(pybind11::init<xt::pyarray<double>&, double>());
    //test.def_readonly("array", &Test::_internal);
    test.def("do", &Test::DoSomething);
    test.def("do_with", &Test::DoSomethingWith);
    m.def("matvec", matvec);
    
    //pybind11::class_<TestEigen, std::shared_ptr<TestEigen>> test_eigen(m, "TestEigen");
    //test_eigen.def(pybind11::init<Eigen::Ref<Eigen::MatrixXd>>());
    //test_eigen.def_readonly("array", &TestEigen::_internal);
    //test_eigen.def("do", &TestEigen::DoSomething);
    
    pybind11::class_<Test2, std::shared_ptr<Test2>> test2(m, "Test2");
    
    test2.def(pybind11::init<>());
    test2.def_readwrite("array", &Test2::_internal);
    test2.def("do", &Test2::DoSomething);
    
    pybind11::class_<LinearElastic, std::shared_ptr<LinearElastic>> linear_elastic(m, "LinearElastic");
    linear_elastic.def(pybind11::init<double, double, bool, int>());
    linear_elastic.def_readwrite("C", &LinearElastic::_C);
    linear_elastic.def("evaluate", &LinearElastic::EvaluateAll);
    
    pybind11::class_<EigenLinearElastic, std::shared_ptr<EigenLinearElastic>> eigen_linear_elastic(m, "EigenLinearElastic");
    eigen_linear_elastic.def(pybind11::init<double, double, bool, int>());
    eigen_linear_elastic.def_readwrite("C", &EigenLinearElastic::_C);
    eigen_linear_elastic.def("evaluate", &EigenLinearElastic::EvaluateAll);
}
