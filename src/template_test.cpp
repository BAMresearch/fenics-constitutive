#include <iostream>
#include "laws.h"


int main(int argc, char* argv[])
{
    std::cout << Dim::G(Constraint::PLANE_STRAIN) << std::endl;
    std::cout << Dim::Q(Constraint::PLANE_STRAIN) << std::endl;

    std::cout << M<Constraint::PLANE_STRAIN>::Zero() << std::endl;

    int n = 1000;
    Eigen::VectorXd strains = Eigen::VectorXd::Random(3 * n);

    LinearElastic law(42, 0.2, PLANE_STRAIN);
    Base constitutive(law);
    constitutive.evaluate(strains);


    return 0;
}
