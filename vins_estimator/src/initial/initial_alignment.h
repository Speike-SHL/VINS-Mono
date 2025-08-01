#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

/**
 * @brief 图像帧类, 包含图像帧中的特征点信息、时间戳、旋转矩阵、平移向量以及预积分信息
 * @param _points 图像帧中的特征点信息 特征点id -> vector<pair<相机id, 特征点信息(归一化坐标3x1, 像素坐标2x1, 速度2x1)>>
 * @param _t 时间戳
 */
class ImageFrame
{
    public:
        ImageFrame(){};
        ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t):t{_t},is_key_frame{false}
        {
            points = _points;
        };
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;
        double t;
        Matrix3d R;    // Rc0bk
        Vector3d T;    // tc0ck
        IntegrationBase *pre_integration;   // 该图像帧对应的预积分对象
        bool is_key_frame;
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);
