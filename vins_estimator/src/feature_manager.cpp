#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

/**
 * 设置特征管理中的相机旋转外参
 */
void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

/**
 * 遍历所有特征点, 返回有效特征点的数目(当点被少于2帧观测到就是无效)
 */
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

/**
 * 将传入的特征点信息添加到特征管理器中, 同时判断倒数第二帧是否为关键帧
 * @param[in] frame_count 当前帧数
 * @param[in] image 传入的当前图像帧 特征点id -> vector<pair<相机id, 特征点信息(归一化坐标3x1, 像素坐标2x1, 速度2x1)>>
 * @param[in] td 相机和IMU时间同步的时间差
 * @return true: 是关键帧, false: 不是关键帧
 * @note 关键帧的选择标准见论文IV-A部分的解读
 */
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    // 遍历每个特征点
    for (auto &id_pts : image)
    {
        // 用特征点信息构造一个帧对象
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        // 取出特征点id并查找该id的特征点是否已经存在
        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        // 如果不存在, 新建一个特征点对象放到特征管理中, 同时将上面建立的帧对象放到该特征点对象中
        // 该特征点对象的start_frame即为传入的这个frame_count
        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        // 如果该特征点已经存在, 则直接将上面建立的帧对象放到该特征点对象中
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;   // QUERY 是这个意思吗 ？ 统计当前帧跟踪到的特征点数量
        }
    }

    // NOTE 关键帧的判断  关键帧的选择标准见论文IV-A部分的解读
    // 如果是前两帧,直接认为是关键帧; 或者如果追踪到的特征点比较少了,即特征关联已经比较弱了,也认为是关键帧
    if (frame_count < 2 || last_track_num < 20)
        return true;

    // 否则进行归一化坐标距离的计算判断是否是关键帧
    for (auto &it_per_id : feature)
    {
        // 遍历每个特征点对象, 判断该特征点对象的起始帧是不是小于等于倒数第三帧,并且
        // 结束帧是不是大于等于倒数第二帧, 如果满足, 则说明该特征点在倒数第三帧和倒数第二帧之间都有出现
        // 然后计算归一化坐标距离
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    // 为0, 说明没有一个特征点能被倒数第三帧和倒数第二帧关联上, 则认为是关键帧
    // 因为此时跟踪质量很差, 为了避免特征跟踪完全丢失
    // QUERY 为什么, 不是说明出问题了吗
    if (parallax_num == 0)
    {
        return true;
    }
    // 否则根据平均的归一化坐标距离以及设置的阈值判断是否是关键帧
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

/**
 * 得到在两帧中被共同观测到的特征点 在两帧中各自的归一化坐标
 * @param[in] frame_count_l 左帧号
 * @param[in] frame_count_r 右帧号
 */
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        // 遍历所有特征点, 判断特征点最早出现是否在左帧前, 最晚出现是否在右帧后
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

/**
 * 把给定的深度赋值给各个特征点作为逆深度
 * @param[in] x 给定的深度
 */
void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

/**
 * 得到所有特征点的逆深度
 * @return 所有特征点的逆深度数组
 */
VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

/**
 * 利用观测到该特征点的所有帧来三角化特征点, 然后赋值给特征管理中所有特征点的估计深度
 * @param[in] Ps tc0ck 所有关键帧相对于参考帧的平移
 * @param[in] tic 相机平移外参
 * @param[in] ric 相机旋转外参
 * // 图片注释 [src/VINS-Mono/support_files/image_comment/initial_sfm.triangulatePoint.jpg]
 */
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    // 遍历所有特征点, 对每个特征点进行三角化
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0)  // 代表已经三角化过了
            continue;
        int imu_i = it_per_id.start_frame;  // 特征点最早出现的关键帧索引
        int imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        // 第一个观察到该特征点的关键帧在参考帧下的位姿 Tc0ci
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];// tc0ck = tc0ck + Rc0bk * tbc, tic传进来为0
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];            // Rc0ck = Rc0bk * Rbc
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        // 遍历所有看到该特征点的关键帧, 用这些关键帧来三角化该特征点
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            
            // imu_j指向的关键帧在参考帧下的位姿 Tc0cj
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            // 求 Tcicj, 即把imu_i和imu_j相对于参考帧的位姿变为相对于imu_i的位姿
            Eigen::Vector3d t = R0.transpose() * (t1 - t0); // tcicj = Rcic0 * (tc0cj - tc0ci)
            Eigen::Matrix3d R = R0.transpose() * R1;        // Rcicj = Rcic0 * Rc0cj

            // 组成3x4的增广形式, [Rcjci | -Rcjci * tcicj], 即第一帧相对于第二帧的
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            // 同样类似于initial_sfm.triangulatePoint()函数中的方法构建方程
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        // 解除该特征点的齐次坐标(4x1)
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        // 用第3维Z 除以 第4维, 即把Z进行归一化, 得到深度
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;     // 赋值深度
        //it_per_id->estimated_depth = INIT_DEPTH;

        // 深度太小说明三角化失败, 给一个初始深度
        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

/**
 * @param[in] it_per_id 特征点对象
 * @param[in] frame_count 当前帧的序号
 * @return 该特征点对象在倒数第二帧和倒数第三帧下归一化像素坐标的距离
 */
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    // 找到该特征点在当前倒数第三帧和导数第二帧下的帧对象
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;   // 特征点在倒数第二帧下的归一化坐标

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;   // 特征点在倒数第三帧下的归一化坐标
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;  // 倒数第二帧和倒数第三帧下特征点的归一化坐标差

    double dep_i_comp = p_i_comp(2);        // 下面没必要, 因为p_i本身就是归一化坐标
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    // 直接就是两个归一化像素坐标的距离
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}
