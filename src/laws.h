#pragma once
#include "interfaces.h"

class LinearElastic : public IpBase
{
public:
    LinearElastic(double E, double nu, Constraint TC)
    {
        _C = C(E, nu, TC);
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate(const Eigen::VectorXd& strain, int i = 0) override
    {
        return {_C * strain, _C};
    }

    virtual int qdim() const override
    {
        return _C.rows();
    }

private:
    Eigen::MatrixXd _C;
};

