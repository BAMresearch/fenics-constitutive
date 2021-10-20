#include "interfaces.h" 
#include <Eigen/Dense>

typedef Eigen::Matrix<double, 9,1> Vector9d;
typedef Eigen::Matrix<double, 6,1> Vector6d;

Vector6d matrix_to_mandel(Eigen::Matrix3d M){
    Vector6d v;
    double s = sqrt(2.0);
    v << M(0,0), M(1,1), M(2,2), s * M(1,2), s * M(0,2), s * M(0,1);
    return v;
}

Eigen::Matrix3d mandel_to_matrix(Vector6d v){
    Eigen::Matrix3d M;
    double s = 1.0 / sqrt(2);
    M << 
        v(0), s * v(5), s * v(4),
        s * v(5), v(1), s * v(3),
        s * v(4), s * v(3), v(2);
    return M;
}

struct ObjectiveStressRate
{
    virtual Eigen::VectorXd Rotate(Eigen::VectorXd& sigma, double h) = 0;
};

class JaumannStressRate : public ObjectiveStressRate
{
public:
    //Eigen::VectorXd _W;
    QValues _W;
    JaumannStressRate(int n)
    {
        _W = QValues(3,3);
        _W.Resize(n);
    }
    void Set(Eigen::VectorXd& L_)
    {
        _W.data = L_;
        //Eigen::Matrix3d L_temp;
        //Eigen::Matrix3d W_temp;

        for(int i = 0;i < _W.data.size()/9;i++)
        {
            auto L_temp = _W.Get(i);
            auto W_temp = 0.5 * (L_temp - L_temp.transpose());

            _W.Set(W_temp, i);
        }
    }

    Eigen::VectorXd Rotate(Eigen::VectorXd& sigma, double h) override
    {
        
        const int n = _W.data.size() / 9;

        for(int i = 0;i < n;i++)
        {
            auto W_temp = _W.Get(i);

            auto stress = mandel_to_matrix(sigma.segment<6>(i*6));
            stress += h * (stress * W_temp.transpose() + W_temp * stress);
            sigma.segment<6>(i*6) = matrix_to_mandel(stress);

        }
        return sigma;
    }
};

Eigen::VectorXd strain_increment(Eigen::VectorXd L_, double h)
{
    const int n = L_.size()/9;
    Eigen::VectorXd strain(n*6);
    Eigen::MatrixXd L_temp(3,3);
    

    for(int i = 0;i < n;i++)
    {
        auto Lsegment = L_.segment<9>(i * 9);
        L_temp = Eigen::Map<Eigen::Matrix3d>(Lsegment.data());

        strain.segment<6>(i*6) =  matrix_to_mandel((h * 0.5) * (L_temp + L_temp.transpose()));
    }
    return strain;
}


class HookesLaw : public LawInterface
{
public:

    Eigen::MatrixXd _C;
    bool _total_strains;
    bool _tangent;

    HookesLaw(double E, double nu, bool total_strains = true, bool tangent = true)
    : _total_strains(total_strains),
    _tangent(tangent)
    {
        const double l = E * nu / (1 + nu) / (1 - 2 * nu);
        const double m = E / (2.0 * (1 + nu));
        _C.setZero(6,6);
        
        _C << 2*m+l, l, l, 0., 0., 0.,
             l, 2*m+l, l, 0., 0., 0.,
             l, l, 2*m+l, 0., 0., 0.,
             0., 0., 0., 2*m, 0., 0.,
             0., 0., 0., 0., 2*m, 0.,
             0., 0., 0., 0., 0., 2*m;
    }
    void DefineOutputs(std::vector<QValues>& out) const override
    {
        out[SIGMA] = QValues(6);
        if (_tangent)
            out[DSIGMA_DEPS] = QValues(6,6);
    }

    void DefineInputs(std::vector<QValues>& input) const override
    {
        input[EPS] = QValues(6);
        if (!_total_strains)
            input[SIGMA] = QValues(6);

    }
    void Evaluate(const std::vector<QValues>& input, std::vector<QValues>& out, int i) override
    {
        auto strain = input[EPS].Get(i);
        if (_total_strains) {
            out[SIGMA].Set(_C*strain, i);
        } else {
            auto sig = input[SIGMA].Get(i);
            out[SIGMA].Set(sig + _C*strain, i);
        }

        if (_tangent)
            out[DSIGMA_DEPS].Set(_C, i);
    }
};

class Hypoelasticity : public LawInterface
{
public:

    Eigen::MatrixXd C;

    Hypoelasticity(double E, double nu)
    {
        const double l = E * nu / (1 + nu) / (1 - 2 * nu);
        const double m = E / (2.0 * (1 + nu));
        C.setZero(6,6);
        
        C << 2*m+l, l, l, 0., 0., 0.,
             l, 2*m+l, l, 0., 0., 0.,
             l, l, 2*m+l, 0., 0., 0.,
             0., 0., 0., 2*m, 0., 0.,
             0., 0., 0., 0., 2*m, 0.,
             0., 0., 0., 0., 0., 2*m;
        
    }
    void DefineOutputs(std::vector<QValues>& out) const override
    {
        out[SIGMA] = QValues(6);
    }

    void DefineInputs(std::vector<QValues>& input) const override
    {
        input[L] = QValues(3,3);
        input[SIGMA] = QValues(6);
        input[TIME_STEP] = QValues(1);
    }
    void Evaluate(const std::vector<QValues>& input, std::vector<QValues>& out, int i) override
    {
        auto L_temp = input[L].Get(i);
        auto sig = input[SIGMA].Get(i);
        auto h = input[TIME_STEP].GetScalar(i);
        const auto D = 0.5 * (L_temp + L_temp.transpose());
        const auto W = 0.5 * (L_temp - L_temp.transpose());

        auto stress = mandel_to_matrix(sig);
        stress += 0.5 * h * (stress * W.transpose() + W * stress);

        stress += mandel_to_matrix(C * matrix_to_mandel(D * h));
        
        stress += 0.5 * h * (stress * W.transpose() + W * stress);
        out[SIGMA].Set(matrix_to_mandel(stress), i);
    }
};
