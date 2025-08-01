#pragma once 

#include <vector>
#include "../parameters.h"
using namespace std;

#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Dense>
using namespace Eigen;
#include <ros/console.h>

/**
 * 初始化旋转外参的类
 * This class help you to calibrate extrinsic rotation between imu and camera when your totally don't konw the extrinsic parameter
 * // NOTE 平移外参由于尺度在初始化时不进行标定
 */
class InitialEXRotation
{
public:
	InitialEXRotation();
    bool CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result);
private:
	Matrix3d solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres);

    double testTriangulation(const vector<cv::Point2f> &l,
                             const vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
    int frame_count;    // 初始化旋转外参类的帧计数器, 初始为0

    vector< Matrix3d > Rc;      // 记录由对极几何得到的两帧相机间的R
    vector< Matrix3d > Rimu;    // 记录由IMU预积分得到的两帧相机间的R
    vector< Matrix3d > Rc_g;    // 将由IMU预积分得到的旋转, 使用之前算出的ric, 预测此时两帧相机间的相对旋转
    Matrix3d ric;               // 相机到IMU的旋转外参
};


