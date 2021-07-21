#include "interfaces.h" 
#include <Eigen/Dense>

Eigen::VectorXd matrix_to_mandel(Eigen::MatrixXd M){
    Eigen::VectorXd v(6);
    double s = sqrt(2);
    v << M(0,0), M(1,1), M(2,2), s * M(1,2), s * M(0,2), s * M(0,1);
    return v;
}

Eigen::MatrixXd mandel_to_matrix(Eigen::VectorXd v){
    Eigen::MatrixXd M(3,3);
    double s = 1.0 / sqrt(2);
    M << 
        v(0), s * v(5), s * v(4),
        s * v(5), v(1), s * v(3),
        s * v(4), s * v(3), v(2);
    return M;
}


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
