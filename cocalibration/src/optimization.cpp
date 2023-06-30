/** headings **/
#include <optimization.h>
#include <common_lib.h>
#include <algorithm>

ofstream outfile;

struct QuaternionOmniFunctor {
    template <typename T>
    bool operator()(const T *const q_, const T *const t_, const T *const intrinsic_, T *cost) const {
        Eigen::Quaternion<T> q{q_[0], q_[1], q_[2], q_[3]};
        Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();
        Eigen::Matrix<T, 3, 1> t(t_);
        Eigen::Matrix<T, K_OMNI_INT, 1> intrinsic(intrinsic_);
        Eigen::Matrix<T, 3, 1> lidar_point = R * lid_point_.cast<T>() + t;
        Eigen::Matrix<T, 2, 1> projection = IntrinsicTransform(intrinsic, lidar_point);
        T res, val;
        kde_interpolator_.Evaluate(projection(0) * T(kde_scale_), projection(1) * T(kde_scale_), &val);
        res = T(weight_) * (T(kde_val_) - val);
        cost[0] = res;
        cost[1] = res;
        cost[2] = res;
        return true;
    }

    QuaternionOmniFunctor(const Vec3D lid_point,
                    const double weight,
                    const double ref_val,
                    const double scale,
                    const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &interpolator)
                    : lid_point_(std::move(lid_point)), kde_interpolator_(interpolator), weight_(std::move(weight)), kde_val_(std::move(ref_val)), kde_scale_(std::move(scale)) {}

    static ceres::CostFunction *Create(const Vec3D &lid_point,
                                       const double &weight,
                                       const double &kde_val,
                                       const double &kde_scale,
                                       const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &interpolator) {
        return new ceres::AutoDiffCostFunction<QuaternionOmniFunctor, 3, ((7)-3), 3, K_OMNI_INT>(
                new QuaternionOmniFunctor(lid_point, weight, kde_val, kde_scale, interpolator));
    }

    const Vec3D lid_point_;
    const double weight_;
    const double kde_val_;
    const double kde_scale_;
    const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &kde_interpolator_;
};

struct QuaternionPinholeFunctor {
    template <typename T>
    bool operator()(const T *const q_, const T *const t_, const T *const intrinsic_, T *cost) const {
        Eigen::Quaternion<T> q{q_[0], q_[1], q_[2], q_[3]};
        Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();
        Eigen::Matrix<T, 3, 1> t(t_);
        Eigen::Matrix<T, K_PINHOLE_INT, 1> intrinsic(intrinsic_);
        Eigen::Matrix<T, 3, 1> lidar_point = R * lid_point_.cast<T>() + t;
        Eigen::Matrix<T, 2, 1> projection = IntrinsicTransform(intrinsic, lidar_point);
        T res, val;
        kde_interpolator_.Evaluate(projection(0) * T(kde_scale_), projection(1) * T(kde_scale_), &val);
        res = T(weight_) * (T(kde_val_) - val);
        cost[0] = res;
        cost[1] = res;
        cost[2] = res;
        return true;
    }

    QuaternionPinholeFunctor(const Vec3D lid_point,
                    const double weight,
                    const double ref_val,
                    const double scale,
                    const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &interpolator)
                    : lid_point_(std::move(lid_point)), kde_interpolator_(interpolator), weight_(std::move(weight)), kde_val_(std::move(ref_val)), kde_scale_(std::move(scale)) {}

    static ceres::CostFunction *Create(const Vec3D &lid_point,
                                       const double &weight,
                                       const double &kde_val,
                                       const double &kde_scale,
                                       const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &interpolator) {
        return new ceres::AutoDiffCostFunction<QuaternionPinholeFunctor, 3, ((7)-3), 3, K_PINHOLE_INT>(
                new QuaternionPinholeFunctor(lid_point, weight, kde_val, kde_scale, interpolator));
    }

    const Vec3D lid_point_;
    const double weight_;
    const double kde_val_;
    const double kde_scale_;
    const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &kde_interpolator_;
};

double project2Image(OmniProcess &omni, LidarProcess &lidar, std::vector<double> &params, std::string record_path, int kNumIntrinsic, double bandwidth) {
    ofstream outfile;
    Ext_D extrinsic = Eigen::Map<Param_D1>(params.data()).head(7);
    Pinhole_Int_D intrinsic = Eigen::Map<Param_D1>(params.data()).tail(kNumIntrinsic);
    EdgeCloud::Ptr ocam_edge_cloud = omni.ocamEdgeCloud;
    EdgeCloud::Ptr lidar_edge_cloud (new EdgeCloud);
    Mat4D T_mat = transformMat(extrinsic);
    std::cout << T_mat << std::endl;
    pcl::transformPointCloud(*lidar.lidarEdgeCloud, *lidar_edge_cloud, T_mat);

    Vec3D lidar_point;
    Vec2D projection;
    cv::Mat fusion_image = cv::imread(omni.cocalibImagePath, cv::IMREAD_UNCHANGED);
    cv::Mat fusion_image1 = cv::Mat::zeros(fusion_image.rows, fusion_image.cols, CV_8UC3);
    for (auto &point : lidar_edge_cloud->points) {
        lidar_point << point.x, point.y, point.z;
        projection = IntrinsicTransform(intrinsic, lidar_point);
        int u = (int)round(projection(0));
        if (u < 0) {
            u = 0;
        }
        else if (u > (omni.cocalibImage.rows - 1)) {
            u = (omni.cocalibImage.rows - 1);
        }
        int v = (int)round(projection(1));
        if (v < 0) {
            v = 0;
        }
        else if (v > (omni.cocalibImage.cols - 1)) {
            v = (omni.cocalibImage.cols - 1);
        }

        point.x = projection(0);
        point.y = projection(1);
        point.z = 0;

        if (u > 0 && v > 0 && u < (omni.kImageSize.first-1) && v < (omni.kImageSize.second-1)) {
            fusion_image.at<cv::Vec3b>(u, v)[0] = 0;   // b
            fusion_image.at<cv::Vec3b>(u, v)[1] = 255; // g
            fusion_image.at<cv::Vec3b>(u, v)[2] = 0;   // r
            fusion_image1.at<cv::Vec3b>(u, v)[0] = 0;   // b
            fusion_image1.at<cv::Vec3b>(u, v)[1] = 255; // g
            fusion_image1.at<cv::Vec3b>(u, v)[2] = 0;   // r            
            if (MESSAGE_EN) {outfile << u << "," << v << endl; }
        }
    }
    if (MESSAGE_EN) {outfile.close(); }
    double proj_error = lidar.getEdgeDistance(ocam_edge_cloud, lidar_edge_cloud, 30);
    /** generate fusion image **/
    cv::imwrite(record_path, fusion_image);
    return proj_error;
}

std::vector<double> QuaternionOmniCalib(OmniProcess &omni,
                                    LidarProcess &lidar,
                                    double bandwidth,
                                    std::vector<double> init_params_vec,
                                    std::vector<double> lb,
                                    std::vector<double> ub,
                                    bool lock_intrinsic) {
    Param_D1 init_params = Eigen::Map<Param_D1>(init_params_vec.data());
    Ext_D extrinsic = init_params.head(7);
    Eigen::Matrix<double, K_OMNI_INT+(7), 1> q_vector;
    Eigen::AngleAxisd angle_axis (extrinsic[0], Eigen::Vector3d(extrinsic[1], extrinsic[2], extrinsic[3]));
    Eigen::Quaterniond quaternion(angle_axis);
    // ceres::EigenQuaternionManifold *q_manifold = new ceres::EigenQuaternionManifold();
    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
    
    const int kParams = q_vector.size();
    const double scale = KDE_SCALE;
    q_vector.tail(K_OMNI_INT + 3) = init_params.tail(K_OMNI_INT + 3);
    q_vector.head(4) << quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z();
    double params[kParams];
    memcpy(params, &q_vector(0), q_vector.size() * sizeof(double));

    /********* Initialize Ceres Problem *********/
    ceres::Problem problem;
    // problem.AddParameterBlock(params, ((7)-3), q_manifold);
    problem.AddParameterBlock(params, ((7)-3), q_parameterization);
    problem.AddParameterBlock(params+((7)-3), 3);
    problem.AddParameterBlock(params+(7), K_OMNI_INT);
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.05);

    /********* Kernel Density Estimation *********/
    std::vector<double> fisheye_kde = omni.Kde(bandwidth, scale);
    double *kde_val = new double[fisheye_kde.size()];
    memcpy(kde_val, &fisheye_kde[0], fisheye_kde.size() * sizeof(double));
    ceres::Grid2D<double> grid(kde_val, 0, omni.kImageSize.first * scale, 0, omni.kImageSize.second * scale);
    double ref_val = *max_element(fisheye_kde.begin(), fisheye_kde.end());
    ceres::BiCubicInterpolator<ceres::Grid2D<double>> interpolator(grid);

    double weight = sqrt(50000.0f / lidar.lidarEdgeCloud->size());
    for (auto &point : lidar.lidarEdgeCloud->points) {
        Vec3D lid_point = {point.x, point.y, point.z};
        problem.AddResidualBlock(QuaternionOmniFunctor::Create(lid_point, weight, ref_val, scale, interpolator),
                            loss_function,
                            params, params+((7)-3), params+(7));
    }   
    if (lock_intrinsic) {
        problem.SetParameterBlockConstant(params + (7));
    }
    for (int i = 0; i < kParams; ++i) {
        if (i < ((7)-3)) {
            problem.SetParameterLowerBound(params, i, (q_vector[i]-Q_LIM));
            problem.SetParameterUpperBound(params, i, (q_vector[i]+Q_LIM));
        }
        if (i >= ((7)-3) && i < (7)) {
            problem.SetParameterLowerBound(params+((7)-3), i-((7)-3), lb[i]);
            problem.SetParameterUpperBound(params+((7)-3), i-((7)-3), ub[i]);
        }
        else if (i >= (7) && !lock_intrinsic) {
            problem.SetParameterLowerBound(params+(7), i-(7), lb[i]);
            problem.SetParameterUpperBound(params+(7), i-(7), ub[i]);
        }
    }

    /********* Initial Options *********/
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = MESSAGE_EN;
    options.num_threads = std::thread::hardware_concurrency();
    options.max_num_iterations = 200;
    options.gradient_tolerance = 1e-6;
    options.function_tolerance = 1e-12;
    options.use_nonmonotonic_steps = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    /********* 2D Image Visualization *********/
    Param_D1 result = Eigen::Map<MatD(K_PINHOLE_INT+(7), 1)>(params);
    Eigen::Quaternion quaternion_export(params[0], params[1], params[2], params[3]);
    Eigen::AngleAxisd angle_axis_export(quaternion_export);
    result.head(4) << angle_axis_export.angle(), angle_axis_export.axis().x(), angle_axis_export.axis().y(), angle_axis_export.axis().z();
    std::vector<double> result_vec(&result[0], result.data()+result.cols()*result.rows());
    
    /** Save Results**/
    std::string fusion_image_path = omni.RESULT_PATH + "/fusion_image_" + std::to_string((int)bandwidth) + ".png";
    std::string cocalib_result_path= lidar.RESULT_PATH + "/cocalib_" + std::to_string((int)bandwidth) + ".txt";
    double proj_error = project2Image(omni, lidar, result_vec, fusion_image_path, K_OMNI_INT, bandwidth);
    saveResults(cocalib_result_path, result_vec, bandwidth, summary.initial_cost, summary.final_cost, proj_error);
    return result_vec;
}

std::vector<double> QuaternionPinholeCalib(OmniProcess &omni,
                                    LidarProcess &lidar,
                                    double bandwidth,
                                    std::vector<double> init_params_vec,
                                    std::vector<double> lb,
                                    std::vector<double> ub,
                                    bool lock_intrinsic) {
    Param_D1 init_params = Eigen::Map<Param_D1>(init_params_vec.data());
    Ext_D extrinsic = init_params.head(7); // angle axisd + translation
    Eigen::Matrix<double, K_PINHOLE_INT+(7), 1> q_vector;
    Eigen::AngleAxisd angle_axis (extrinsic[0], Eigen::Vector3d(extrinsic[1], extrinsic[2], extrinsic[3]));
    Eigen::Quaterniond quaternion(angle_axis);
    // ceres::EigenQuaternionManifold *q_manifold = new ceres::EigenQuaternionManifold();
    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
     
    const int kParams = q_vector.size();
    const double scale = KDE_SCALE;
    q_vector.head(4) << quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z();
    q_vector.tail(3 + K_PINHOLE_INT) = init_params.tail(3 + K_PINHOLE_INT);
    double params[kParams];
    memcpy(params, &q_vector(0), q_vector.size() * sizeof(double));

    /********* Initialize Ceres Problem *********/
    ceres::Problem problem;
    // problem.AddParameterBlock(params, ((7)-3), q_manifold);
    problem.AddParameterBlock(params,   4, q_parameterization); // quaternion
    problem.AddParameterBlock(params+4, 3); // translation
    problem.AddParameterBlock(params+7, K_PINHOLE_INT); // intrinsics
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.05);

    /********* Kernel Density Estimation *********/
    std::vector<double> fisheye_kde = omni.Kde(bandwidth, scale);
    double *kde_val = new double[fisheye_kde.size()];
    memcpy(kde_val, &fisheye_kde[0], fisheye_kde.size() * sizeof(double));
    ceres::Grid2D<double> grid(kde_val, 0, omni.kImageSize.first * scale, 0, omni.kImageSize.second * scale);
    double ref_val = *max_element(fisheye_kde.begin(), fisheye_kde.end());
    ceres::BiCubicInterpolator<ceres::Grid2D<double>> interpolator(grid);

    double weight = sqrt(50000.0f / lidar.lidarEdgeCloud->size());
    for (auto &point : lidar.lidarEdgeCloud->points) {
        Vec3D lid_point = {point.x, point.y, point.z};
        problem.AddResidualBlock(QuaternionPinholeFunctor::Create(lid_point, weight, ref_val, scale, interpolator),
                            loss_function,
                            params, params+4, params+7);
    }   
    if (lock_intrinsic) {
        problem.SetParameterBlockConstant(params + (7));
    }
    for (int i = 0; i < kParams; ++i) {
        if (i < 4) {
            problem.SetParameterLowerBound(params, i, (q_vector[i]-Q_LIM));
            problem.SetParameterUpperBound(params, i, (q_vector[i]+Q_LIM));
        }
        if (i >= 4 && i < 7) {
            problem.SetParameterLowerBound(params+4, i-4, lb[i]);
            problem.SetParameterUpperBound(params+4, i-4, ub[i]);
        }
        else if (i >= 7 && !lock_intrinsic) {
            problem.SetParameterLowerBound(params+7, i-7, lb[i]);
            problem.SetParameterUpperBound(params+7, i-7, ub[i]);
        }
    }

    /********* Initial Options *********/
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = MESSAGE_EN;
    options.num_threads = std::thread::hardware_concurrency();
    options.max_num_iterations = 200;
    options.gradient_tolerance = 1e-6;
    options.function_tolerance = 1e-12;
    options.use_nonmonotonic_steps = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    /********* 2D Image Visualization *********/
    Param_D1 result = Eigen::Map<MatD(K_PINHOLE_INT+(7), 1)>(params);
    Eigen::Quaternion quaternion_export(params[0], params[1], params[2], params[3]);
    Eigen::AngleAxisd angle_axis_export(quaternion_export);
    result.head(4) << angle_axis_export.angle(), angle_axis_export.axis().x(), angle_axis_export.axis().y(), angle_axis_export.axis().z();
    std::vector<double> result_vec(&result[0], result.data()+result.cols()*result.rows());
    /** Save Results**/
    std::string fusion_image_path = omni.RESULT_PATH + "/fusion_image_" + std::to_string((int)bandwidth) + ".png";
    std::string cocalib_result_path= lidar.RESULT_PATH + "/cocalib_" + std::to_string((int)bandwidth) + ".txt";
    double proj_error = project2Image(omni, lidar, result_vec, fusion_image_path, K_PINHOLE_INT, bandwidth);
    saveResults(cocalib_result_path, result_vec, bandwidth, summary.initial_cost, summary.final_cost, proj_error);
    return result_vec;
}

std::vector<double> QuaternionOmniMultiCalib(
                                    std::vector<OmniProcess> cam_vec,
                                    std::vector<LidarProcess> lidar_vec,
                                    double bandwidth,
                                    std::vector<double> init_params_vec,
                                    std::vector<double> lb,
                                    std::vector<double> ub,
                                    bool lock_intrinsic) {
    Param_D init_params = Eigen::Map<Param_D>(init_params_vec.data());
    Ext_D extrinsic = init_params.head(7);
    MatD(K_OMNI_INT+(7), 1) q_vector;
    Eigen::AngleAxisd angle_axis (extrinsic[0], Eigen::Vector3d(extrinsic[1], extrinsic[2], extrinsic[3]));
    Eigen::Quaterniond quaternion(angle_axis);
    // ceres::EigenQuaternionManifold *q_manifold = new ceres::EigenQuaternionManifold();
    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
    
    const int kParams = q_vector.size();
    const double scale = KDE_SCALE;
    q_vector.tail(K_OMNI_INT + 3) = init_params.tail(K_OMNI_INT + 3);
    q_vector.head(4) << quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z();
    double params[kParams];
    memcpy(params, &q_vector(0), q_vector.size() * sizeof(double));

    /********* Initialize Ceres Problem *********/
    ceres::Problem problem;
    problem.AddParameterBlock(params, ((7)-3), q_parameterization);
    problem.AddParameterBlock(params+((7)-3), 3);
    problem.AddParameterBlock(params+(7), K_OMNI_INT);
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.05);

    /********* Fisheye KDE *********/
    std::vector<double> ref_vals;
    std::vector<ceres::Grid2D<double>> grids;
    std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> interpolators;
    for (int idx = 0; idx < cam_vec.size(); idx++) {
        cout << "Camera KDE construction: spot index " << idx << endl;
        std::vector<double> fisheye_kde = cam_vec[idx].Kde(bandwidth, scale);
        double *kde_val = new double[fisheye_kde.size()];
        memcpy(kde_val, &fisheye_kde[0], fisheye_kde.size() * sizeof(double));
        ceres::Grid2D<double> grid(kde_val, 0, cam_vec[idx].kImageSize.first * scale, 0, cam_vec[idx].kImageSize.second * scale);
        grids.push_back(grid);
        double ref_val = *max_element(fisheye_kde.begin(), fisheye_kde.end());
        ref_vals.push_back(ref_val);
    }
    const std::vector<ceres::Grid2D<double>> kde_grids(grids);
    for (int idx = 0; idx < cam_vec.size(); idx++) {
        ceres::BiCubicInterpolator<ceres::Grid2D<double>> interpolator(kde_grids[idx]);
        interpolators.push_back(interpolator);
    }
    const std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> kde_interpolators(interpolators);

    for (int idx = 0; idx < lidar_vec.size(); idx++) {
        EdgeCloud &edge_cloud = *lidar_vec[idx].lidarEdgeCloud;
        double weight = sqrt(50000.0f / edge_cloud.size());
        
        for (auto &point : edge_cloud.points) {
            Vec3D lid_point = {point.x, point.y, point.z};
            problem.AddResidualBlock(QuaternionOmniFunctor::Create(lid_point, weight, ref_vals[idx], scale, kde_interpolators[idx]),
                                loss_function, params, params+((7)-3), params+(7));
        }
    }
    if (lock_intrinsic) {
        problem.SetParameterBlockConstant(params + (7));
    }

    for (int i = 0; i < kParams; ++i) {
        if (i < ((7)-3)) {
            problem.SetParameterLowerBound(params, i, (q_vector[i]-Q_LIM));
            problem.SetParameterUpperBound(params, i, (q_vector[i]+Q_LIM));
        }
        if (i >= ((7)-3) && i < (7)) {
            problem.SetParameterLowerBound(params+((7)-3), i-((7)-3), lb[i]);
            problem.SetParameterUpperBound(params+((7)-3), i-((7)-3), ub[i]);
        }
        else if (i >= (7) && !lock_intrinsic) {
            problem.SetParameterLowerBound(params+(7), i-(7), lb[i]);
            problem.SetParameterUpperBound(params+(7), i-(7), ub[i]);
        }
    }

    /********* Initial Options *********/
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = MESSAGE_EN;
    options.num_threads = std::thread::hardware_concurrency();
    options.max_num_iterations = 200;
    options.gradient_tolerance = 1e-6;
    options.function_tolerance = 1e-12;
    options.use_nonmonotonic_steps = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    /********* 2D Image Visualization *********/
    Param_D1 result = Eigen::Map<MatD(K_PINHOLE_INT+(7), 1)>(params);
    Eigen::Quaternion quaternion_export(params[0], params[1], params[2], params[3]);
    Eigen::AngleAxisd angle_axis_export(quaternion_export);
    result.head(4) << angle_axis_export.angle(), angle_axis_export.axis().x(), angle_axis_export.axis().y(), angle_axis_export.axis().z();
    std::vector<double> result_vec(&result[0], result.data()+result.cols()*result.rows());

    for (int idx = 0; idx < cam_vec.size(); idx++) {
        std::string fusion_image_path = cam_vec[idx].RESULT_PATH + "/fusion_image_" + to_string((int)bandwidth) + ".png";
        std::string cocalib_result_path= lidar_vec[idx].RESULT_PATH + "/cocalib_" + to_string((int)bandwidth) + ".txt";
        double proj_error = project2Image(cam_vec[idx], lidar_vec[idx], result_vec, fusion_image_path, K_PINHOLE_INT, bandwidth);
        saveResults(cocalib_result_path, result_vec, bandwidth, summary.initial_cost, summary.final_cost, proj_error);
    }
    return result_vec;
}

std::vector<double> QuaternionPinholeMultiCalib(
                                    std::vector<OmniProcess> cam_vec,
                                    std::vector<LidarProcess> lidar_vec,
                                    double bandwidth,
                                    std::vector<double> init_params_vec,
                                    std::vector<double> lb,
                                    std::vector<double> ub,
                                    bool lock_intrinsic) {
    Param_D init_params = Eigen::Map<Param_D>(init_params_vec.data());
    Ext_D extrinsic = init_params.head(7);
    MatD(K_PINHOLE_INT+7, 1) q_vector;
    Eigen::AngleAxisd angle_axis (extrinsic[0], Eigen::Vector3d(extrinsic[1], extrinsic[2], extrinsic[3]));
    Eigen::Quaterniond quaternion(angle_axis);
    // ceres::EigenQuaternionManifold *q_manifold = new ceres::EigenQuaternionManifold();
    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
    
    const int kParams = q_vector.size();
    const double scale = KDE_SCALE;
    q_vector.tail(K_PINHOLE_INT + 3) = init_params.tail(K_PINHOLE_INT + 3);
    q_vector.head(4) << quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z();
    double params[kParams];
    memcpy(params, &q_vector(0), q_vector.size() * sizeof(double));

    /********* Initialize Ceres Problem *********/
    ceres::Problem problem;
    problem.AddParameterBlock(params, (7-3), q_parameterization);
    problem.AddParameterBlock(params+(7-3), 3);
    problem.AddParameterBlock(params+(7), K_PINHOLE_INT);
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.05);

    /********* Fisheye KDE *********/
    std::vector<double> ref_vals;
    std::vector<ceres::Grid2D<double>> grids;
    std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> interpolators;
    for (int idx = 0; idx < cam_vec.size(); idx++) {
        cout << "Camera KDE construction: spot index " << idx << endl;
        std::vector<double> fisheye_kde = cam_vec[idx].Kde(bandwidth, scale);
        double *kde_val = new double[fisheye_kde.size()];
        memcpy(kde_val, &fisheye_kde[0], fisheye_kde.size() * sizeof(double));
        ceres::Grid2D<double> grid(kde_val, 0, cam_vec[idx].kImageSize.first * scale, 0, cam_vec[idx].kImageSize.second * scale);
        grids.push_back(grid);
        double ref_val = *max_element(fisheye_kde.begin(), fisheye_kde.end());
        ref_vals.push_back(ref_val);
    }
    const std::vector<ceres::Grid2D<double>> kde_grids(grids);
    for (int idx = 0; idx < cam_vec.size(); idx++) {
        ceres::BiCubicInterpolator<ceres::Grid2D<double>> interpolator(kde_grids[idx]);
        interpolators.push_back(interpolator);
    }
    const std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> kde_interpolators(interpolators);

    for (int idx = 0; idx < lidar_vec.size(); idx++) {
        EdgeCloud &edge_cloud = *lidar_vec[idx].lidarEdgeCloud;
        double weight = sqrt(50000.0f / edge_cloud.size());
        
        for (auto &point : edge_cloud.points) {
            Vec3D lid_point = {point.x, point.y, point.z};
            problem.AddResidualBlock(QuaternionPinholeFunctor::Create(lid_point, weight, ref_vals[idx], scale, kde_interpolators[idx]),
                                loss_function, params, params+((7)-3), params+(7));
        }
    }
    if (lock_intrinsic) {
        problem.SetParameterBlockConstant(params + (7));
    }

    for (int i = 0; i < kParams; ++i) {
        if (i < ((7)-3)) {
            problem.SetParameterLowerBound(params, i, (q_vector[i]-Q_LIM));
            problem.SetParameterUpperBound(params, i, (q_vector[i]+Q_LIM));
        }
        if (i >= ((7)-3) && i < (7)) {
            problem.SetParameterLowerBound(params+((7)-3), i-((7)-3), lb[i]);
            problem.SetParameterUpperBound(params+((7)-3), i-((7)-3), ub[i]);
        }
        else if (i >= (7) && !lock_intrinsic) {
            problem.SetParameterLowerBound(params+(7), i-(7), lb[i]);
            problem.SetParameterUpperBound(params+(7), i-(7), ub[i]);
        }
    }

    /********* Initial Options *********/
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = MESSAGE_EN;
    options.num_threads = std::thread::hardware_concurrency();
    options.max_num_iterations = 200;
    options.gradient_tolerance = 1e-6;
    options.function_tolerance = 1e-12;
    options.use_nonmonotonic_steps = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    /********* 2D Image Visualization *********/
    Param_D1 result = Eigen::Map<MatD(K_PINHOLE_INT+(7), 1)>(params);
    Eigen::Quaternion quaternion_export(params[0], params[1], params[2], params[3]);
    Eigen::AngleAxisd angle_axis_export(quaternion_export);
    result.head(4) << angle_axis_export.angle(), angle_axis_export.axis().x(), angle_axis_export.axis().y(), angle_axis_export.axis().z();
    std::vector<double> result_vec(&result[0], result.data()+result.cols()*result.rows());

    for (int idx = 0; idx < cam_vec.size(); idx++) {
        std::string fusion_image_path = cam_vec[idx].RESULT_PATH + "/fusion_image_" + to_string((int)bandwidth) + ".png";
        std::string cocalib_result_path= lidar_vec[idx].RESULT_PATH + "/cocalib_" + to_string((int)bandwidth) + ".txt";
        double proj_error = project2Image(cam_vec[idx], lidar_vec[idx], result_vec, fusion_image_path, K_PINHOLE_INT, bandwidth);
        saveResults(cocalib_result_path, result_vec, bandwidth, summary.initial_cost, summary.final_cost, proj_error);
    }
    return result_vec;
}

void costAnalysis(OmniProcess &omni,
                  LidarProcess &lidar,
                  std::vector<double> init_params_vec,
                  std::vector<double> result_vec,
                  double bandwidth) {
    const double scale = KDE_SCALE;

    /********* Fisheye KDE *********/
    std::vector<double> fisheye_kde = omni.Kde(bandwidth, scale);
    double *kde_val = new double[fisheye_kde.size()];
    memcpy(kde_val, &fisheye_kde[0], fisheye_kde.size() * sizeof(double));
    ceres::Grid2D<double> grid(kde_val, 0, omni.kImageSize.first * scale, 0, omni.kImageSize.second * scale);
    double ref_val = *max_element(fisheye_kde.begin(), fisheye_kde.end());
    ceres::BiCubicInterpolator<ceres::Grid2D<double>> interpolator(grid);

    /***** Correlation Analysis *****/
    Param_D params_mat = Eigen::Map<Param_D>(result_vec.data());
    Ext_D extrinsic;
    Omni_Int_D intrinsic;
    std::vector<double> results;
    std::vector<double> input_x;
    std::vector<double> input_y;
    std::vector<const char*> name = {
            "alpha", "rx", "ry", "rz",
            "tx", "ty", "tz",
            "u0", "v0",
            "a0", "a1", "a2", "a3", "a4",
            "c", "d", "e"};
    int steps[3] = {1, 1, 1};
    int param_idx[3] = {0, 3, 7};
    const double step_size[3] = {0.0002, 0.001, 0.01};
    const double deg2rad = M_PI / 180;
    double offset[3] = {0, 0, 0};

    /** update evaluate points in 2D grid **/
    for (int m = 0; m < 7; m++) {
        extrinsic = params_mat.head(7);
        intrinsic = params_mat.tail(K_OMNI_INT);
        if (m < 3) {
            steps[0] = 201;
            steps[1] = 1;
            steps[2] = 1;
            param_idx[0] = m;
            param_idx[1] = 3;
            param_idx[2] = 7;
        }
        else if (m < 7){
            steps[0] = 1;
            steps[1] = 201;
            steps[2] = 1;
            param_idx[0] = 0;
            param_idx[1] = m;
            param_idx[2] = 7;
        }
        else {
            steps[0] = 1;
            steps[1] = 1;
            steps[2] = 201;
            param_idx[0] = 0;
            param_idx[1] = 3;
            param_idx[2] = m;
        }

        double normalize_weight = sqrt(1.0f / lidar.lidarEdgeCloud->size());

        /** Save & terminal output **/
        string analysis_filepath = omni.DATASET_PATH + "/log/";
        if (steps[0] > 1) {
            analysis_filepath = analysis_filepath + name[param_idx[0]] + "_";
        }
        if (steps[1] > 1) {
            analysis_filepath = analysis_filepath + name[param_idx[1]] + "_";
        }
        if (steps[2] > 1) {
            analysis_filepath = analysis_filepath + name[param_idx[2]] + "_";
        }
        outfile.open(analysis_filepath + "_bw_" + to_string(int(bandwidth)) + "_result.txt", ios::out);
        if (steps[0] > 1) {
            outfile << init_params_vec[param_idx[0]] << "\t" << result_vec[param_idx[0]] << endl;
        }
        if (steps[1] > 1) {
            outfile << init_params_vec[param_idx[1]] << "\t" << result_vec[param_idx[1]] << endl;
        }
        if (steps[2] > 1) {
            outfile << init_params_vec[param_idx[2]] << "\t" << result_vec[param_idx[2]] << endl;
        }
        
        for (int i = -int((steps[0]-1)/2); i < int((steps[0]-1)/2)+1; i++) {
            offset[0] = i * step_size[0];
            extrinsic(param_idx[0]) = params_mat(param_idx[0]) + offset[0];
            
            for (int j = -int((steps[1]-1)/2); j < int((steps[1]-1)/2)+1; j++) {
                offset[1] = j * step_size[1];
                extrinsic(param_idx[1]) = params_mat(param_idx[1]) + offset[1];

                for (int n = -int((steps[2]-1)/2); n < int((steps[2]-1)/2)+1; n++) {
                    offset[2] = n * step_size[2];
                    intrinsic(param_idx[2]-7) = params_mat(param_idx[2]) + offset[2];
                
                    double step_res = 0;
                    int num_valid = 0;
                    /** Evaluate cost funstion **/
                    for (auto &point : lidar.lidarEdgeCloud->points) {
                        double val;
                        double weight = normalize_weight;
                        Eigen::Vector4d lidar_point4 = {point.x, point.y, point.z, 1.0};
                        Mat4D T_mat = transformMat(extrinsic);
                        Vec3D lidar_point = (T_mat * lidar_point4).head(3);
                        Vec2D projection = IntrinsicTransform(intrinsic, lidar_point);
                        interpolator.Evaluate(projection(0) * scale, projection(1) * scale, &val);
                        Pair &bounds = omni.kEffectiveRadius;
                        if ((pow(projection(0) - intrinsic(0), 2) + pow(projection(1) - intrinsic(1), 2)) > pow(bounds.first, 2)
                            && (pow(projection(0) - intrinsic(0), 2) + pow(projection(1) - intrinsic(1), 2)) < pow(bounds.second, 2)) {
                            step_res += pow(weight * val, 2);
                        }
                    }
                    if (steps[0] > 1) {
                        outfile << offset[0] + params_mat(param_idx[0]) << "\t";
                    }
                    if (steps[1] > 1) {
                        outfile << offset[1] + params_mat(param_idx[1]) << "\t";
                    }
                    if (steps[2] > 1) {
                        outfile << offset[2] + params_mat(param_idx[2]) << "\t";
                    }
                    outfile << step_res << endl;
                }
            }
        }
        outfile.close();
    }
}
