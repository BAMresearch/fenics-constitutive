#pragma once
#include "interfaces.h"

class LinearElastic : public IpBase
{
public:
    LinearElastic(double E, double nu, Constraint TC)
    {
        _C = C(E, nu, TC);
    }

    virtual void evaluate(const Eigen::VectorXd& strain, int i, Eigen::Ref<Eigen::VectorXd> stress,
                          Eigen::Ref<Eigen::MatrixXd> dstress) override
    {
        stress = _C * strain;
        dstress = _C;
    }

    virtual int qdim() const override
    {
        return _C.rows();
    }

private:
    Eigen::MatrixXd _C;
};

