#include "initial_alignment.h"

/**
 * 求解陀螺零偏, 同时利用求解出来的零偏重新进行预积分
 * @param[in] all_image_frame 所有图像帧
 * @param[out] Bgs 陀螺零偏, 指向滑窗中每帧对应的陀螺零偏
 * @note 这部分公式见论文V-B-1)
 */
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        // q_ij = qc0bk.inv * qc0bk+1
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        // 旋转对零偏的雅可比
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        // ɣ.inv * q_ij 求虚部
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A; // 乘tmp_A.T是因为LDLT分解需要对称正定
        b += tmp_A.transpose() * tmp_b;

    }
    delta_bg = A.ldlt().solve(b);   // LDLT分解求陀螺零偏
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    // 把估计出的陀螺零偏添加给滑窗中每帧
    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    // 遍历每帧图像, 每帧图像对应的预积分量都重新预积分
    // 因为之前的预积分量都是假设陀螺零偏为0算的, 所以第一次进行零偏预估可能会变化比较大
    // 为了准确性, 重新预积分。但后面就可以使用雅可比进行预积分的近似更新了
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}

/**
 * 根据给定的重力向量, 求解与重力向量垂直的两个基向量b1,b2
 * b1,b2,g0 两两正交
 * @return bc = [b1; b2] 3x2
 */
MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

/**
 * 重力细化, 实际上这里把所有的状态量都重新算了, 只不过给了一个重力的先验
 * @note 论文V-B-3)
 * // 图片注释 [src/VINS-Mono/support_files/image_comment/initial_alignment.RefineGravity.jpg]
 */
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;  // 3 * 每帧的速度 + 剩2自由度的重力 + 1个尺度

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)  // 迭代细化4次
    {
        MatrixXd lxly(3, 2);        // 在重力切平面上的两个基向量
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;

            // 操作与LinearAlignment类似
            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(n_state - 3);    // 取出两个基向量上的扰动
            g0 = (g0 + lxly * dg).normalized() * G.norm();  // 更新重力
            //double s = x(n_state - 1);
    }   
    g = g0;
}

/**
 * 视觉惯性对齐主要代码, 先估计出每帧图像的速度、相对于参考帧的重力、尺度
 * 然后进行重力细化, 对估计出的重力进一步调整，
 * @note 论文5-B-2)
 * // 图片注释 [src/VINS-Mono/support_files/image_comment/initial_alignment.LinearAlignment.jpg]
 */
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;  // 3 * 每帧的速度 + 重力 + 尺度

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);    // frame_i是第k帧，frame_j是第k+1帧

        MatrixXd tmp_A(6, 10);  // 论文公式(11)的H
        tmp_A.setZero();
        VectorXd tmp_b(6);      // 论文公式(10)的Z
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        // 论文公式(11)
        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();   // -I*Δt
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        // 系数除100, X中的尺度就会放大100, 方便保证尺度的稳定性
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        // 组合成包含所有状态量的大矩阵, 因为上面只包含了两个时刻的v
        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    // 增强数值稳定性, 左右两边都乘了, 不影响x结果
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    // 取出尺度
    double s = x(n_state - 1) / 100.0;    // 因为上面系数也除了100. 例 2x = 6, x = 3; 0.02x = 6, x = 300 / 100
    ROS_DEBUG("estimated scale: %f", s);
    // 取出重力方向并检查
    g = x.segment<3>(n_state - 4);
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }

    RefineGravity(all_image_frame, g, x);
    // 重力细化之后的尺度
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

/**
 * 视觉惯性对齐
 * @param[in] all_image_frame 所有图像帧 <时间戳, 图像帧> 包含每帧的位姿和预积分量
 * @param[out] Bgs 计算出陀螺零偏后给estimator对象使用, 给滑窗中每帧设置零偏
 * @param[out] g 重力向量
 * @param[out] x 其他状态量
 * @return true or false
 */
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    solveGyroscopeBias(all_image_frame, Bgs);

    if(LinearAlignment(all_image_frame, g, x))
        return true;
    else 
        return false;
}
