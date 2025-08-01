#pragma once 
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;


/**
 * structure from motion 特征
 * 储存特征点id, 是否被三角化, 三角化后的三维坐标和深度等
 */
struct SFMFeature
{
    bool state;			// 是否被三角化
    int id;				// 特征点id
    vector<pair<int,Vector2d>> observation;	// 所有观测到该特征点的图像帧id 以及在该帧中的归一化像素坐标
    double position[3];	// 三角化后的三维坐标
    double depth;		// 三角化后的深度
};

/**
 * 构建BA问题的重投影误差, 给Ceres求解
 * 重载了()运算符, 便于使用Ceres的自动求导
 */
struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	/**
	 * 重载()运算符, 定义了如何求残差, 便于使用Ceres的自动求导,
	 * @param[in] 参数块1 相机旋转四元数
	 * @param[in] 参数块2 相机平移向量
	 * @param[in] 参数块3 三维点坐标
	 * @param[out] 残差, 必须放在最后
	 */
	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);	// 先使用四元数旋转三维点转到相机系下
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];	// 再加上相机平移向量得到相机坐标系下的三维点坐标
		T xp = p[0] / p[2];	// 进一步得到相机系下的归一化坐标
    	T yp = p[1] / p[2];
    	residuals[0] = xp - T(observed_u);	// 跟现有的观测构造重投影误差
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	// 创建使用自动求导的CostFunction
	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
		// AutoDiffCostFunction<代价函数结构体, 输出残差维度, 参数块1维度, 参数块2维度, 参数块3维度>
		// 这里的参数块顺序要和operator()中的参数顺序一致
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>(
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;	// 观测到的归一化像素坐标u
	double observed_v;	// 观测到的归一化像素坐标v
};

class GlobalSFM
{
public:
	GlobalSFM();
	bool construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
	bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
							Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  vector<SFMFeature> &sfm_f);

	int feature_num;
};
