#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

/**
 * 对特征点进行三角化
 * @param[in] Pose0 相机0位姿 [Rcw, tcw]
 * @param[in] Pose1 相机1位姿 [Rcw, tcw]
 * @param[in] point0 相机0特征点归一化像素坐标
 * @param[in] point1 相机1特征点归一化像素坐标
 * @param[out] point_3d 三角化后的三维坐标
 * 图片注释 // [src/VINS-Mono/support_files/image_comment/initial_sfm.triangulatePoint.jpg]
 */
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	// SVD分解求三角化
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


/**
 * 使用PnP求解相机位姿
 * @param[in, out] R_initial 第i帧姿态, 传入时使用i-1帧的姿态作为初始值
 * @param[in, out] P_initial 第i帧位置, 传入时使用i-1帧的位置作为初始值
 * @param[in] i 第i帧索引
 * @param[in] sfm_f 所有特征点信息
 */
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;	// 储存第i帧中的2D归一化像素坐标
	vector<cv::Point3f> pts_3_vector;	// 储存第i帧中特征点在世界坐标系(实际上为参考帧)中的3D坐标
	for (int j = 0; j < feature_num; j++)	// 遍历所有特征点
	{
		if (sfm_f[j].state != true)			// 如果特征点没有被三角化过, 跳过, PnP需要找到3D点
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == i)	// 遍历3D点的所有观测, 如果该3D点被第i帧观测了
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;	// 该特征点在第i帧的归一化像素坐标
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);	// 该特征点在世界坐标系(实际上为参考帧)中的3D坐标
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}

	// 如果特征点数量小于15, 则认为特征点跟踪不稳定
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		// 特征点对小于10, PnP失败
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);	// eigen矩阵转换为cv::Mat
	cv::Rodrigues(tmp_r, rvec);		// 旋转矩阵转为旋转向量
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	// 以rvec, t为初始值, 使用PnP求解相机位姿
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);	// 旋转向量转为旋转矩阵
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp; 
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;
}

/**
 * 根据两帧索引和位姿, 三角化两帧间的特征点
 * @param[in] frame0 第一帧索引
 * @param[in] Pose0 第一帧位姿 [Rcw, tcw]
 * @param[in] frame1 第二帧索引
 * @param[in] Pose1 第二帧位姿 [Rcw, tcw]
 * @param[in,out] sfm_f sfm的三维点对象
 */
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	// 遍历每个特征点, feature_num 就是 sfm_f.size(), 在construct中被赋值
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		// 遍历特征点对应的观测帧, 如果观测帧中有 frame0 和 frame1, 则三角化
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = sfm_f[j].observation[k].second;
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
		// 特征点被 frame0 和 frame1 同时观测到, 进行三角化
		if (has_0 && has_1)
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
/**
 * 纯视觉SFM
 * 根据已有的参考帧与最后一帧的位姿, 恢复出滑窗中各关键帧位姿和3D点坐标
 * @param[in] frame_num 滑窗中关键帧总数
 * @param[out] q 滑窗内所有关键帧相对于参考帧的旋转
 * @param[out] T 滑窗内所有关键帧相对于参考帧的平移
 * @param[in] l 参考帧id
 * @param[in] relative_R 参考帧与最后一帧的旋转矩阵 R_l_end
 * @param[in] relative_T 参考帧与最后一帧的平移向量 t_l_end
 * @param[in] sfm_f 用来做SFM的特征点
 * @param[out] sfm_tracked_points 	< 特征点id, 特征点在参考帧中的3D坐标 >
 * @return true or false
 */
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size();
	//cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view
	// 参考帧初始化, 设置为原点, 和世界系重合 R_wl, t_wl
	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
	// 设置最后一帧在世界系下的位姿, R_w_end = R_l_end, t_w_end = t_l_end
	q[frame_num - 1] = q[l] * Quaterniond(relative_R);
	T[frame_num - 1] = relative_T;
	//cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
	//cout << "init t_l " << T[l].transpose() << endl;

	//rotate to cam frame
	// 把cam -> world转为world -> cam
	Matrix3d c_Rotation[frame_num];			// 每一帧的旋转矩阵 R_iw
	Vector3d c_Translation[frame_num];		// 每一帧的平移向量 t_iw
	Quaterniond c_Quat[frame_num];			// 每一帧的旋转四元数 q_iw
	double c_rotation[frame_num][4];		// 用于ceres-solver, 因为只接收double类型
	double c_translation[frame_num][3];		// 用于ceres-solver, 因为只接收double类型
	Eigen::Matrix<double, 3, 4> Pose[frame_num];	// 每一帧的位姿矩阵 [R_iw | t_iw]

	c_Quat[l] = q[l].inverse();						// R_wl -> R_lw
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]); // t_wl -> t_lw
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];		// [R_lw | t_lw]
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();						// R_w_end -> R_end_w
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]); // t_w_end -> t_end_w	
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];		// [R_end_w | t_end_w]
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];


	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1;
	// STEP 1 : 先三角化 参考帧(l) 和 最后一帧(frame_num - 1) 间共视的路标点, 得到3D坐标
	// STEP 2 : 因为是光流法, 上面三角化出的3D点一定会被 l+1 --- frame_num-2 帧观测到. 因此使用PnP求解 l+1 --- frame_num-2 帧的位姿
	//          并进一步使用对应帧 和 最后一帧 间共视的路标点进行三角化
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		if (i > l)
		{
			Matrix3d R_initial = c_Rotation[i - 1];		// 使用上一帧的位姿作为初始值求解PnP
			Vector3d P_initial = c_Translation[i - 1];
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}

	//3: triangulate l-----l+1 l+2 ... frame_num -2
	// STEP 3: 由于某些点不能被最后一帧观测到, 所有再进一步三角化 l+1 --- frame_num-2 帧 与 参考帧 l 间共视的路标点
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);

	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	// STEP 4: 开始使用PnP求解参考帧以前各帧的位姿, 并三角化各帧与参考帧l间共视的路标点
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}
	//5: triangulate all other points
	// STEP 5: 最后再三角化所有剩余的特征点, 因为此时所有滑窗中帧的位姿都已经知道了。三角化时取该特征点最早和最晚的观测帧, 这样平移更充分
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)	// 被三角化过, 直接跳过
			continue;
		if ((int)sfm_f[j].observation.size() >= 2)	// 只有该点的观测大于2才三角化
		{
			Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;	// 取该特征点的最早观测帧
			point0 = sfm_f[j].observation[0].second;
			int frame_1 = sfm_f[j].observation.back().first;// 取该特征点的最晚观测帧
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/
	//full BA
	// STEP 6: 求出滑窗中所有帧的位姿和所有特征点的3D坐标后, 再进行一次全局BA. 因为之前的解都是线性推出来的
	// NOTE 这里参数块只添加了位姿, 没添加3D点, 实际上这里的BA只优化了位姿, 没优化3D点
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();	// 用于四元数在欧式空间的参数化
	//cout << " begin full BA " << endl;
	// 把每帧的位姿参数块加入BA问题中, 同时固定参考帧的旋转 以及 参考帧和最后一帧的平移
	for (int i = 0; i < frame_num; i++)
	{
		//double array for ceres
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);	// 添加每帧的旋转参数块, 并使用四元数参数化
		problem.AddParameterBlock(c_translation[i], 3);							// 添加每帧的平移参数块
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);		// 固定参考帧的旋转, 因为所有位姿都是相对于参考帧的
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);	// 固定参考帧和最后一帧的平移, fix尺度
		}
	}

	// 遍历每个特征点
	for (int i = 0; i < feature_num; i++)
	{
		if (sfm_f[i].state != true)	// 如果该特征点没有三角化成功, 则跳过
			continue;
		// 遍历每个特征点对应的观测, 构造重投影误差加入BA问题中. 即每个特征点的每个观测
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			int l = sfm_f[i].observation[j].first;	// 观测的索引
			// 构建代价函数
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());
			// 添加残差块, (代价函数, 核函数, 旋转参数块, 平移参数块, 3D点坐标)
    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
    								sfm_f[i].position);	 
		}
	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;	// 线性求解器类型
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;			// 最大求解时间
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);			// 开始求解
	//std::cout << summary.BriefReport() << "\n";
	// 判断终止条件是否是因为收敛, 如果是超时的话看一下最后残差是否满足要求
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}
	// 取出结果, 并把Tcw -> Twc
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse();
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{

		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}

	// < 特征点id, 特征点在参考帧中的3D坐标 >
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}

