#include "utility.h"

/**
 * 求出重力到世界系[0, 0, 1]的旋转矩阵
 */
Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = Utility::R2ypr(R0).x();    // 取出yaw
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;  // 乘上-yaw, 相当于把yaw抵消了
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
