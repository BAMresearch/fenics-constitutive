#include <iostream>
#include "definitions.h"

class ConstitutiveBase
{
public:
    virtual void integrate(const Eigen::VectorXd& all_strains) = 0;

    Eigen::VectorXd _stress;
    Eigen::VectorXd _dstress;
};

template <Constraint TC>
class ConstitutiveBaseFix : public ConstitutiveBase
{
public:
    virtual void integrate(const Eigen::VectorXd& all_strains)
    {
        constexpr int q = Dim::Q(TC);
    }
};

template <Constraint TC>
class ConstitutiveIpBase
{
public:
    virtual void evaluateIp(const V<TC>& strain, Eigen::Ref<V<TC>> stress, Eigen::Ref<M<TC>> dstress) = 0;
};

template <Constraint TC>
class LinearElastic : public ConstitutiveIpBase<TC>
{
public:
    LinearElastic(double E, double nu)
    {
        _C = C<TC>(E, nu);
    }

    virtual void evaluateIp(const V<TC>& strain, Eigen::Ref<V<TC>> stress, Eigen::Ref<M<TC>> dstress)
    {
        stress = _C * strain;
        dstress = _C;
    }

private:
    M<TC> _C;
};

int main(int argc, char* argv[])
{

    std::cout << Dim::G(Constraint::PLANE_STRAIN) << std::endl;
    std::cout << Dim::Q(Constraint::PLANE_STRAIN) << std::endl;

    std::cout << M<Constraint::PLANE_STRAIN>::Zero() << std::endl;


    LinearElastic<PLANE_STRAIN> constitutive(42, 0.2);
    M<PLANE_STRAIN> dstress;
    V<PLANE_STRAIN> stress;

    constitutive.evaluateIp({1, 0, 0}, stress, dstress);

    std::cout << stress << "\n" << dstress << std::endl;

    return 0;
}
