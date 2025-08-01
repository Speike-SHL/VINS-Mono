#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

/**
 * 特征点对应于某一帧的类, 管理了特征点在某一帧中的信息
 * 比如
 */
class FeaturePerFrame
{
  public:
    /**
     * @param[in] _point 特征点信息(归一化坐标3x1, 像素坐标2x1, 速度2x1)
     */
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
    }
    double cur_td;
    Vector3d point;     // 归一化相机坐标
    Vector2d uv;        // 特征点在该帧的像素坐标
    Vector2d velocity;  // 特征点速度
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

/**
 * 每个特征点的类, 管理了单个特征点的信息.
 * 比如特征点id, 深度, 求解状态, 是否被边缘化
 * 以及该特征点在对应帧中的属性
 */
class FeaturePerId
{
  public:
    const int feature_id;   // 特征点的id
    int start_frame;        // 在滑窗中是被第几帧先看到的
    // 存储该特征点在每一帧中的信息。因为是光流法而不是特征点法, 所以从start_frame开始, 每一帧都会看到该特征点
    // 否则即使是同一个特征点也会id不同。而特征点法可能会出现中间断帧的情况
    vector<FeaturePerFrame> feature_per_frame;

    int used_num;     // 被几帧看到, 等于feature_per_frame.size()
    bool is_outlier;
    bool is_margin;
    double estimated_depth; // 特征点估计的深度
    int solve_flag;         // 求解状态(是否被正确三角化) 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();
};

/**
 * 特征点管理类
 */
class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature; // 管理特征点的链表, 每个元素都是一个FeaturePerId对象
    int last_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;       // 指向estimator中的Rs
    Matrix3d ric[NUM_OF_CAM]; // 每个相机的旋转外参, 但实际只有一个相机
};

#endif
