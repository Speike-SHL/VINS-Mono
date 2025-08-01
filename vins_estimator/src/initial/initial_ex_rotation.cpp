#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    ric = Matrix3d::Identity();
}

/**
 * 标定IMU和相机间的旋转外参
 * @param[in] 两个相机帧间的匹配点对(归一化坐标)
 * @param[in] IMU预积分算出的两个相机帧间的旋转
 * @param[out] 标定结果: 相机到IMU的旋转外参
 * // 图片注释 [src/VINS-Mono/support_files/image_comment/initial_ex_rotation.CalibrationExRotation.jpg]
 */
bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{
    // 每标定一次，frame_count加1, 同时储存历史的Rc, Rimu, Rc_g, 用于构造多帧超定的A阵
    frame_count++;
    Rc.push_back(solveRelativeR(corres));               // Rckck+1
    Rimu.push_back(delta_q_imu.toRotationMatrix());     // Rbkbk+1
    // Rck+1ck = rcb * Rbkbk+1 * rbc
    // 实际上是使用当前的IMU预积分得到的旋转, 结合之前算出的外参, 预测此时两帧相机间的旋转应该什么样
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric);  

    Eigen::MatrixXd A(frame_count * 4, 4);
    A.setZero();
    int sum_ok = 0;
    for (int i = 1; i <= frame_count; i++)
    {
        // 计算实际的相机旋转和IMU预测的相机旋转间的差异
        Quaterniond r1(Rc[i]);
        Quaterniond r2(Rc_g[i]);
        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        ROS_DEBUG(
            "%d %f", i, angular_distance);

        // 使用核函数, 如果差异过大就减低权重
        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        Matrix4d L, R;

        double w = Quaterniond(Rc[i]).w();
        Vector3d q = Quaterniond(Rc[i]).vec();
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Quaterniond R_ij(Rimu[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);
    }

    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    Matrix<double, 4, 1> x = svd.matrixV().col(3);
    Quaterniond estimated_R(x);
    ric = estimated_R.toRotationMatrix().inverse();
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;
    Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>();
    // 当初始化旋转外参的帧数大于等于WINDOW_SIZE 且 奇异值不病态时, 认为标定成功
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25)
    {
        calib_ric_result = ric;
        return true;
    }
    else
        return false;
}

/**
 * 利用对极约束求解相机帧间的R12
 * @param[in] 两个相机帧间的匹配点对(归一化坐标)
 */
Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres)
{
    if (corres.size() >= 9)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        cv::Mat E = cv::findFundamentalMat(ll, rr); // 因为传入的是归一化坐标, 所以这个函数求出的实际为本质矩阵
        cv::Mat_<double> R1, R2, t1, t2;
        decomposeE(E, R1, R2, t1, t2);              // 进行本质矩阵的分解, 得到四组解(R1, t1), (R1, t2), (R2, t1), (R2, t2)

        if (determinant(R1) + 1.0 < 1e-09)          // 旋转矩阵的行列式应为1, 如果分解结果为-1, 对E取反重新分解
        {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }

        // 判断哪组解是正确的, 三角化出的点在相机系下能有正的深度
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;

        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j); // 将R21转为R12
        return ans_R_eigen;
    }
    return Matrix3d::Identity();
}

/**
 * 利用三角化对R和t进行测试
 * @param[in] l, r 左右相机归一化坐标的匹配点对
 * @param[in] R, t 旋转矩阵和平移向量
 * @return 这个R,t 能正确三角化出l,r中匹配点对的比例. 比例越大说明这个R,t越可信
 */
double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                          const vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat pointcloud;
    // l 帧到世界系的位姿, 设为单位阵 Tlw
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    // r 帧到世界系的位姿, 设为R,t    Trw
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    // 进行三角化, 得到三维点在世界系下的齐次坐标(4,1)pointcloud
    cv::triangulatePoints(P, P1, l, r, pointcloud);
    int front_count = 0;
    for (int i = 0; i < pointcloud.cols; i++)
    {
        // 将齐次坐标归一化转为三维坐标并都转到各自的相机系下
        double normal_factor = pointcloud.col(i).at<float>(3);
        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor); // Tlw * pw
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);// Trw * pw
        
        // 如果特征点在各自相机系下都有正的深度, 记数加一
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    return 1.0 * front_count / pointcloud.cols;
}

/**
 * 使用SVD分解本质矩阵, 十四讲(7.15)
 */
void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}
