#pragma once
#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor/xview.hpp"

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

enum Q
{
    LAMBDA,
    KAPPA,
    LAST,
};
//Eigen::MatrixXd Get(Eigen::int i) const
//{
  /*TODO: Same as in Set.
   *
   * */
    //Eigen::VectorXd ip_values = data.segment(_rows * _cols * i, _rows * _cols);
    //return Eigen::Map<Eigen::MatrixXd>(ip_values.data(), _cols, _rows).transpose();
//}

inline auto GetQuadratureView(xt::pytensor<double, 1>& v, int i, int rows, int cols=1)
{
    int size = rows*cols;
    auto data_slice = xt::reshape_view(xt::view(v, xt::range(size * i, size * (i+1))),{rows,cols});
    return data_slice;
}
inline auto GetVectorView(xt::pytensor<double, 1>& v, int i, int rows)
{
    auto data_slice = xt::reshape_view(xt::view(v, xt::range(rows * i, rows * (i+1))),{rows});
    return data_slice;
}

class QValues
{
public:
    xt::pytensor<double,1> data;
    int _rows = 0;
    int _cols = 0;

    QValues() = default;
    
    QValues(int rows, int cols=1)
        :_rows(rows),_cols(cols)
    {
    }
    void Resize(int n)
    {
        data = xt::zeros<double>({n * _cols * _rows});
    }
    auto GetTensorView(int i)
    {
        int size = _rows*_cols;
        auto data_slice = xt::view(data, xt::range(size * i, size * (i+1)));
        return xt::reshape_view(data_slice,{_rows,_cols});
    }
    auto GetVectorView(int i)
    {
        auto data_slice = xt::view(data, xt::range(_rows * i, _rows * (i+1)));
        return xt::reshape_view(data_slice,{_rows});
    }
    double GetScalar(int i)
    {
        return data(i);
    }
    void Set(xt::xarray<double>& m, int i)
    {
        auto view = GetTensorView(i);
        view = m;
    }
    void Set(double s, int i)
    {
        data(i) = s;
    }
    bool IsUsed()
    {
        return _rows != 0;
    }

};
class MechanicsLawInterface
{
public:
    MechanicsLawInterface(bool tangents, int n)
        : _tangents(tangents), _n(n)
    {
    }

    inline virtual void EvaluateIP(
            int i,
            xt::pytensor<double, 1>& eps_vector,
            xt::pytensor<double, 1>& sigma_vector,
            xt::pytensor<double, 1>& tangents_vector,
            double del_t
            ) = 0;

    //virtual void EvaluateIP(
            //int i,
            //const xt::pyarray<double>& L_vector,
            //xt::pyarray<double>& sigma_vector,
            //double del_t
            //) = 0;
    

    void UpdateIP(int i)
    {
    }
    
    void EvaluateAll(
            xt::pytensor<double, 1>& eps_vector,
            xt::pytensor<double, 1>& sigma_vector,
            xt::pytensor<double, 1>& tangents_vector,
            double del_t)
    {
        for(int i=0;i<_n;i++){
            EvaluateIP(i, eps_vector, sigma_vector, tangents_vector, del_t);
        }
    }
    void UpdateAll()
    {
        for(int i=0;i<_n;i++){
            UpdateIP(i);
        }
    }
    //virtual xt::pyarray<double>& GetInternalVar(Q which)
    //{
    //}

    //virtual void Resize(int n) = 0;
    
    //const bool _incremental_strains;

    const bool _tangents;
    const int _n;
};

class EigenMechanicsLawInterface
{
public:
    EigenMechanicsLawInterface(bool tangents, int n)
        : _tangents(tangents), _n(n)
    {
    }

    inline virtual void EvaluateIP(
            int i,
            Eigen::Ref<Eigen::VectorXd> eps_vector,
            Eigen::Ref<Eigen::VectorXd> sigma_vector,
            Eigen::Ref<Eigen::VectorXd> tangents_vector,
            double del_t
            ) = 0;

    //virtual void EvaluateIP(
            //int i,
            //const xt::pyarray<double>& L_vector,
            //xt::pyarray<double>& sigma_vector,
            //double del_t
            //) = 0;
    

    void UpdateIP(int i)
    {
    }
    
    void EvaluateAll(
            Eigen::Ref<Eigen::VectorXd> eps_vector,
            Eigen::Ref<Eigen::VectorXd> sigma_vector,
            Eigen::Ref<Eigen::VectorXd> tangents_vector,
            double del_t)
    {
        for(int i=0;i<_n;i++){
            EvaluateIP(i, eps_vector, sigma_vector, tangents_vector, del_t);
        }
    }
    void UpdateAll()
    {
        for(int i=0;i<_n;i++){
            UpdateIP(i);
        }
    }
    //virtual xt::pyarray<double>& GetInternalVar(Q which)
    //{
    //}

    //virtual void Resize(int n) = 0;
    
    //const bool _incremental_strains;

    const bool _tangents;
    const int _n;
};

inline xt::pytensor<double,1> matvec(xt::pytensor<double, 2> A, xt::pytensor<double,1> x)
{
    //std::cout << "I am in matvec\n";
    auto shape_A = A.shape();
    //auto shape_x = x.shape(); 
    xt::pytensor<double,1> output = xt::zeros_like(x);
    for(int i=0;i<shape_A[0];i++)
    {
        for(int j =0;j<shape_A[1];j++)
        {
            output(i)+=A(i,j)*x(j);
        }
    }
    return output;
}
