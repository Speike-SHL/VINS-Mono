#include "feature_tracker.h"

// 当前的最大id
int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

/// 使用双指针法, 对vector进行“瘦身”, 只保留status为true的元素
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

/// 使用双指针法, 对vector进行“瘦身”, 只保留status为true的元素
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

/// @brief 设置mask, 把当前帧flow_pts内MIN_DIST范围内的像素点置为0, 不能在此周围提点
///        同时根据追踪次数, 把track_cnt, ids, forw_pts进行排序(一个圆范围内仅保留追踪次数最多的点)
///        如果圆内已经有特征点了, 该区域内的其他点都被会删除
void FeatureTracker::setMask()
{
    // 初始化一个和图像大小一样的mask为全白
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    // pair<特征点被跟踪的次数, pair<特征点当前帧的像素坐标, 特征点id>>
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    // 被跟踪次数多的排在前面, 因为追踪次数越多越稳定
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    // 空间清除, 方便下面重新填入
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    // 在特征点周围MIN_DIST的范围内, 把mask置为0, 表示不能在当前区域提取新的特征点
    // 同时根据排序顺序把track_cnt, ids, forw_pts重新填入(一个圆范围内仅保留追踪次数最多的点)
    // 因此如果点过于密集, 会被剔除
    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}


/// @brief 把新提取的特征点加入到forw_pts, ids, track_cnt中
///        ids为-1表示新提取的特征点, 在feature_tracker_node.cpp中会加上id
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

/**
 * @param[in] _img 输入图像
 * @param[in] _cur_time 图像的时间戳
 * @brief
 * 1、图像均衡化预处理
 * 2、光流追踪
 * 3、提取新的特征点（如果发布）
 * 4、所有特征点去畸变，计算速度
 * @note 并不发布, 发布在feature_tracker_node.cpp中
 */
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    // 图像均衡化: 图像太暗或太亮时, 提特征点比较困难, 所以均衡化一下, 提高对比度, 方便提取角点
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    // forw表示当前帧, cur表示上一帧, 第一次输入图像prev_img没有用到
    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();   // 清空一下当前帧图像的特征点信息

    // 如果上一帧有特征点, 就可以进行光流追踪了
    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        // 进行光流追踪, forw_pts和status是追踪后的特征点和追踪状态
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        // 使用图像边界剔除outlier
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        // 对下面所有vector进行“瘦身”, 节省空间
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }
    
    // 所有的追踪次数加一
    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        // 使用对极约束去除外点
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        // 设置mask, 避免在特征点周围提取新的特征点. 同时根据追踪次数对track_cnt, ids, forw_pts进行排序
        // 如果一个圆内有多个特征点, 只保留追踪次数最多的那个
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        // 计算需要新提取的特征点数目
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())    // 没有地方提点了
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            // 在mask中提取新的特征点. 发布才会提特征点, 但是发不发布都要进行追踪
            // 0.01是设置提升点的最低质量
            // MIN_DIST是设置特征点之间的最小距离
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        // 把新提取的特征点加入到forw_pts, ids, track_cnt中
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;   // 上面三个量都没用到
    cur_img = forw_img;     // 上一帧图像 = 当前帧图像
    cur_pts = forw_pts;     // 上一帧特征点 = 当前帧特征点
    undistortedPoints();
    prev_time = cur_time;
}

/// @brief 使用对极约束剔除outlier
void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)   // 点的数目要大于8, 因为八点法求本质矩阵
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            // 得到上一帧特征点的归一化坐标
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            // 将归一化坐标投影到虚拟相机的像素坐标
            // FOCAL_LENGTH 是在 parameters.cpp 中写死的一个固定值
            // 这样的好处是, 因为上面求出归一化坐标已经和相机内参无关了, 然后这里再投影到一个虚拟相机上
            // 在下面计算基础矩阵的时候, 不同的相机点就都用一个FOCAL_LENGTH, 这样F_THRESHOLD值就可以用
            // 一个统一的标准来确定
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        // 使用RANSAC计算基础矩阵
        // F_THRESHOLD是图像2点到极线的距离, 超过这个阈值被认为是outlier
        // 0.99是RANSAC计算时的置信度, 决定了RANSAC的迭代次数
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        // 使用计算基础矩阵得到的status, 剔除outlier, 对vector进行瘦身
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

/// @brief 给新的特征点附上id, 越界就返回false
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

/// @brief 当前帧的所有点进行去畸变, 同时计算特征点速度, 用来后续时间戳标定
/// @note 这个函数中的cur代表当前帧, 因为readImage函数中在调用之前, 已经把forw赋值给cur了
void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());

    // 对当前帧的所有点进行去畸变
    // cur_un_pts存储的是归一化平面上的坐标
    // cur_un_pts_map存储的是id和归一化平面上的坐标的映射关系
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    // 上一帧map不为空, 说明不是第一帧, 计算点的速度(新提的点速度为0)
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        // prev_un_pts_map为空, 说明是第一帧, 所有点速度为0
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
