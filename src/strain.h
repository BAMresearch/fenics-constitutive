#pragma once

#include <Eigen/Core>

namespace constitutive
{
Eigen::Vector3d timesTwo(Eigen::Vector3d a)
{
    return a * 2.;
}
} // namespace constitutive
