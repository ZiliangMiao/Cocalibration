/** basic **/
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
/** opencv **/
#include <opencv2/opencv.hpp>
/** ros **/
#include <ros/ros.h>
#include <ros/package.h>
/** pcl **/
#include <pcl/common/io.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/conditional_removal.h>
/** heading **/
#include "optimization.h"
#include "common_lib.h"
/** namespace **/
using namespace std;
using namespace cv;

typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
// Edit your calibration object/data here. Vector size is the number of frames 
void roughCalib(OmniProcess &omni,
                LidarProcess &lidar,
                Vector7d &ext_params,  // Initial extrinsic parameters
                double search_resolution_rotation,  
                double search_resolution_translation,
                double search_time_rotation,
                double search_time_translation) {
    Eigen::Vector3d fix_adjust_euler(0, 0, 0);
    Eigen::Vector3d fix_adjust_translation(ext_params[4], ext_params[5], ext_params[6]);  // Initial translation
    Eigen::Matrix3d rot;  // Initial Rotation
    rot = Eigen::AngleAxisd(ext_params[0], Eigen::Vector3d(ext_params[1], ext_params[2], ext_params[3]));

    // Intrinsic parameters
    Eigen::Matrix<double, K_PINHOLE_INT, 1> int_params;
    int_params << 2561.3, 1180.2, 1035.9, -0.1474, 0.1150;

    // Generate combinations of different 6-DoF parameters
    // Number of combinations = (2 * search_time_rotation + 1)^3 * (2 * search_time_translation + 1)^3
    std::vector<Vector6d> adjust_params;
    for (int roll_idx = -search_time_rotation; roll_idx <= search_time_rotation; ++roll_idx) {
        for (int pitch_idx = -search_time_rotation; pitch_idx <= search_time_rotation; ++pitch_idx) {
            for (int yaw_idx = -search_time_rotation; yaw_idx <= search_time_rotation; ++yaw_idx) {
                for (int x_idx = -search_time_translation; x_idx <= search_time_translation; ++x_idx) {
                    for (int y_idx = -search_time_translation; y_idx <= search_time_translation; ++y_idx) {
                        for (int z_idx = -search_time_translation; z_idx <= search_time_translation; ++z_idx) {
                            Vector6d adjust_param;
                            adjust_param[0] = roll_idx * search_resolution_rotation;
                            adjust_param[1] = pitch_idx * search_resolution_rotation;
                            adjust_param[2] = yaw_idx * search_resolution_rotation;
                            adjust_param[3] = x_idx * search_resolution_translation;
                            adjust_param[4] = y_idx * search_resolution_translation;
                            adjust_param[5] = z_idx * search_resolution_translation;
                            adjust_params.push_back(adjust_param);
                        }
                    }
                }
            }
        }
    }
    std::cout << "Searching among " << adjust_params.size() << " combinations." << std::endl;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();


    // KDE construction
    double bandwidth = 4.0;
    std::vector<double> cam_kde = omni.Kde(bandwidth, KDE_SCALE);
    double *kde_val = new double[cam_kde.size()];
    memcpy(kde_val, &cam_kde[0], cam_kde.size() * sizeof(double));
    ceres::Grid2D<double> grid(kde_val, 0, omni.kImageSize.first * KDE_SCALE, 0, omni.kImageSize.second * KDE_SCALE);
    double ref_val = *max_element(cam_kde.begin(), cam_kde.end());
    ceres::BiCubicInterpolator<ceres::Grid2D<double>> interpolator(grid);

    // Searching
    for (size_t i = 0; i < adjust_params.size(); ++i) {
        // Record how long does it take to search among 1000 groups of parameters
        if (i > 0 && (i % 1000 == 0)) {
            std::cout << "i = " << i << std::endl;
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>> (t2 - t1);
            std::cout << "Searching costs " << time_used.count() << " seconds." << std::endl;
            t1 = t2;
        }

        Eigen::Vector3d adjust_euler = fix_adjust_euler;
        adjust_euler[0] = adjust_params[i][0];
        adjust_euler[1] = adjust_params[i][1];
        adjust_euler[2] = adjust_params[i][2];
        Eigen::Matrix3d adjust_rotation_matrix;
        adjust_rotation_matrix = Eigen::AngleAxisd(adjust_euler[0], Eigen::Vector3d::UnitZ()) *
                                 Eigen::AngleAxisd(adjust_euler[1], Eigen::Vector3d::UnitY()) *
                                 Eigen::AngleAxisd(adjust_euler[2], Eigen::Vector3d::UnitX());
        Eigen::Matrix3d test_rot = adjust_rotation_matrix * rot;  // New rotation matrix
        Eigen::AngleAxisd test_vector;
        test_vector.fromRotationMatrix(test_rot); // New rotation vector

        // New translation
        Eigen::Vector3d adjust_translation;
        adjust_translation[0] = fix_adjust_translation[0] + adjust_params[i][3];
        adjust_translation[1] = fix_adjust_translation[1] + adjust_params[i][4];
        adjust_translation[2] = fix_adjust_translation[2] + adjust_params[i][5];

        // New extrinsic parameters
        Vector7d test_params;
        test_params << test_vector.angle(), test_vector.axis()[0], test_vector.axis()[1], test_vector.axis()[2],
                       adjust_translation[0], adjust_translation[1], adjust_translation[2];
        
        double max_cost = 1000000.0; // Maximum cost/energy
        double cost = 0; // Current cost/energy
        // for (size_t j = 0; j < calibs.size(); j++)  // Calculate the sum of cost/energy of all the frames

        double normalize_weight = sqrt(1.0f / lidar.lidarEdgeCloud->size());
        cout << "Size of Edge Point Cloud: " << lidar.lidarEdgeCloud->size() << endl;
        Mat4D T_mat = transformMat(test_params);
        pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*lidar.lidarEdgeCloud, *edge_cloud_transformed, T_mat);

        for (auto &point : edge_cloud_transformed->points) {
            double val;
            Vec3D lidar_point = {point.x, point.y, point.z};
            Vec2D projection = IntrinsicTransform(int_params, lidar_point);
            interpolator.Evaluate(projection(0) * KDE_SCALE, projection(1) * KDE_SCALE, &val);
            cost += pow(normalize_weight * val, 2);
        }
        
        // If this is a better group of extrinsic parameters, then update 
        if (cost > max_cost) {
            std::cout << "Rough calibration min cost: " << cost << ". Index: " << i << std::endl;
            max_cost = cost;
            ext_params[0] = test_params[0];
            ext_params[1] = test_params[1];
            ext_params[2] = test_params[2];
            ext_params[3] = test_params[3];
            ext_params[4] = test_params[4];
            ext_params[5] = test_params[5];
            ext_params[6] = test_params[6];
        }
    }
    cout << "Final rough calibration results: " << ext_params[0] << " " << ext_params[1] << " " << ext_params[2] << " "
         << ext_params[3] << " " << ext_params[4] << " " << ext_params[5] << " " << ext_params[6] << endl;
}

void dataProcessing(OmniProcess &omni, LidarProcess &lidar, bool kOmniCamera, bool kOmniLiDAR) {
    /********* Pre Processing *********/
    cout << "----------------- Camera Processing ---------------------" << endl;
    cout << omni.cocalibImagePath << endl;
    omni.loadCocalibImage();
    if (kOmniCamera) {// omni camera edge extraction
        omni.edgeExtraction(omni.cocalibImagePath, omni.EDGE_PATH, "camera_omni");
    }
    else {// pinhole camera edge extraction
        omni.edgeExtraction(omni.cocalibImagePath, omni.EDGE_PATH, "camera_pinhole");
    }
    omni.generateEdgeCloud();
    cout << "----------------- LiDAR Processing ---------------------" << endl;
    if (kOmniLiDAR) {
        lidar.cartToSphere();
        lidar.sphereToPlane();
        lidar.edgeExtraction(lidar.flatImagePath, lidar.EDGE_PATH, "lidar_mid360");
    }
    else {
        lidar.cartToPlane();
        lidar.edgeExtraction(lidar.flatImagePath, lidar.EDGE_PATH, "lidar_avia");
    }
    lidar.generateEdgeCloud();
}

int main(int argc, char** argv) {
    /***** ROS Initialization *****/
    ros::init(argc, argv, "main");
    ros::NodeHandle nh;

    /***** ROS Parameters Server *****/
    bool kOmniCamera; // if true, calibrate the omnidirectional camera, otherwise calibrate the pinhole camera
    bool kOmniLiDAR; // if true, use mid-360 lidar, otherwise use avia or mid-40
    bool kMultiSpotOpt;
    bool kRoughCalib; // if the initial value have large deviation, please use the rough calib (grid search)
    bool kCeresOpt;
    bool kParamsAnalysis;
    int kNumSpot;
    nh.param<bool>("switch/kOmniCamera", kOmniCamera, false);
    nh.param<bool>("switch/kOmniLiDAR", kOmniLiDAR, false);
    nh.param<bool>("switch/kMultiSpotOpt", kMultiSpotOpt, false);
    nh.param<bool>("switch/kRoughCalib", kRoughCalib, false);
    nh.param<bool>("switch/kCeresOpt", kCeresOpt, false);
    nh.param<bool>("switch/kParamsAnalysis", kParamsAnalysis, false);
    nh.param<int>("essential/kNumSpot", kNumSpot, 1);
    /** Initialization **/
    std::vector<double> bw;
    nh.param<vector<double>>("cocalib/bw", bw, {32, 16, 8, 4, 2, 1});
    // Extrinsic Params
    double alpha, rx, ry, rz; // rotation: angle axised (radian)
    double tx, ty, tz; // translation
    double alpha_range, rx_range, ry_range, rz_range, tx_range, ty_range, tz_range;
    nh.param<double>("cocalib/alpha", alpha, 0.00);
    nh.param<double>("cocalib/rx", rx, 0.00);
    nh.param<double>("cocalib/ry", ry, 0.00);
    nh.param<double>("cocalib/rz", rz, 0.00);
    nh.param<double>("cocalib/tx", tx, 0.00);
    nh.param<double>("cocalib/ty", ty, 0.00);
    nh.param<double>("cocalib/tz", tz, 0.00);
    nh.param<double>("cocalib/rx_range", rx_range, 0.00);
    nh.param<double>("cocalib/ry_range", ry_range, 0.00);
    nh.param<double>("cocalib/rz_range", rz_range, 0.00);
    nh.param<double>("cocalib/tx_range", tx_range, 0.00);
    nh.param<double>("cocalib/ty_range", ty_range, 0.00);
    nh.param<double>("cocalib/tz_range", tz_range, 0.00);
    // Intrinsic Params
    int kNumIntrinsic;
    nh.param<int>("essential/kNumIntrinsic", kNumIntrinsic, 5); // 5 for pinhole camera, 10 for omnidirectional camera
    if (kOmniCamera) { // check whether the number of intrinsic parameters defined in define.h is correct
        ROS_ASSERT_MSG((kNumIntrinsic == K_OMNI_INT), "Number of the omni camera intrinsics are not consistent, please check .yaml and define.h!\n");    
    }
    else {
        ROS_ASSERT_MSG((kNumIntrinsic == K_PINHOLE_INT), "Number of the pinhole camera intrinsics are not consistent, please check .yaml and define.h!\n");
    }

    std::vector<double> params_init = {};
    std::vector<double> params_range = {};
    if (kOmniCamera) {
        double u0, v0, a0, a1, a2, a3, a4, c, d, e;
        double u0_range, v0_range, a0_range, a1_range, a2_range, a3_range, a4_range, c_range, d_range, e_range;
        nh.param<double>("cocalib/u0", u0, 1024);
        nh.param<double>("cocalib/v0", v0, 1201);
        nh.param<double>("cocalib/a0", a0, 0.00);
        nh.param<double>("cocalib/a1", a1, 0.00);
        nh.param<double>("cocalib/a2", a2, 0.00);
        nh.param<double>("cocalib/a3", a3, 0.00);
        nh.param<double>("cocalib/a4", a4, 0.00);
        nh.param<double>("cocalib/c", c, 1.00);
        nh.param<double>("cocalib/d", d, 0.00);
        nh.param<double>("cocalib/e", e, 0.00);
        nh.param<double>("cocalib/u0_range", u0_range, 0.00);
        nh.param<double>("cocalib/v0_range", v0_range, 0.00);
        nh.param<double>("cocalib/a0_range", a0_range, 0.00);
        nh.param<double>("cocalib/a1_range", a1_range, 0.00);
        nh.param<double>("cocalib/a2_range", a2_range, 0.00);
        nh.param<double>("cocalib/a3_range", a3_range, 0.00);
        nh.param<double>("cocalib/a4_range", a4_range, 0.00);
        nh.param<double>("cocalib/c_range", c_range, 0.00);
        nh.param<double>("cocalib/d_range", d_range, 0.00);
        nh.param<double>("cocalib/e_range", e_range, 0.00);
        params_init = {
        alpha, rx, ry, rz, tx, ty, tz,
        u0, v0, a0, a1, a2, a3, a4, c, d, e};
        params_range = {
            alpha_range, rx_range, ry_range, rz_range, tx_range, ty_range, tz_range,
            u0_range, v0_range,
            a0_range, a1_range, a2_range, a3_range, a4_range,
            c_range, d_range, e_range};
    }
    else {
        double f, cx, cy, k1, k2;
        double f_range, cx_range, cy_range, k1_range, k2_range;
        nh.param<double>("cocalib/f", f, 0.00);
        nh.param<double>("cocalib/cx", cx, 1224.00);
        nh.param<double>("cocalib/cy", cy, 1024.00);
        nh.param<double>("cocalib/k1", k1, 0.0);
        nh.param<double>("cocalib/k2", k2, 0.0);
        nh.param<double>("cocalib/f_range", f_range, 0.0);
        nh.param<double>("cocalib/cx_range", cx_range, 0.0);
        nh.param<double>("cocalib/cy_range", cy_range, 0.0);
        nh.param<double>("cocalib/k1_range", k1_range, 0.0);
        nh.param<double>("cocalib/k2_range", k2_range, 0.0);
        params_init = {
            alpha, rx, ry, rz, tx, ty, tz, f, cx, cy, k1, k2};
        params_range = {
            alpha_range, rx_range, ry_range, rz_range, tx_range, ty_range, tz_range,
            f_range, cx_range, cy_range, k1_range, k2_range};
    }
    std::vector<double> params_cocalib(params_init);
    cout << "CHECK ROS PARAMS!" << rx << " " << ry << " " << rz << " " << endl;

    /***** Calibration and Optimization Cost Analysis *****/
    std::vector<double> lb(params_range.size()), ub(params_range.size());
    for (int i = 0; i < params_range.size(); ++i) {
        ub[i] = params_init[i] + params_range[i];
        lb[i] = params_init[i] - params_range[i];
    }
    if (kOmniCamera) {
        // Eigen::Matrix<double, 3, 17> params_mat;
        // params_mat.row(0) = Eigen::Map<Eigen::Matrix<double, 1, 17>>(params_init.data());
        // params_mat.row(1) = params_mat.row(0) - Eigen::Map<Eigen::Matrix<double, 1, 17>>(params_range.data());
        // params_mat.row(2) = params_mat.row(0) + Eigen::Map<Eigen::Matrix<double, 1, 17>>(params_range.data());
    }

    /***** Ceres Optimization *****/
    if (!kMultiSpotOpt && kNumSpot==1) {
        cout << "----------------- Single Spot Optimization ---------------------" << endl;
        // Class Object Initialization
        OmniProcess omni("/single_opt", 0);
        LidarProcess lidar("/single_opt", 0);
        lidar.ext_ = Eigen::Map<Param_D1>(params_init.data()).head(7);
        if (kOmniCamera) {
            omni.omni_int_ = Eigen::Map<Param_D1>(params_init.data()).tail(kNumIntrinsic);
        }
        else {
            omni.pinhole_int_  = Eigen::Map<Param_D1>(params_init.data()).tail(kNumIntrinsic);
            lidar.pinhole_int_ = Eigen::Map<Param_D1>(params_init.data()).tail(kNumIntrinsic);
        }
        // Folder Check
        CheckFolder(lidar.DATASET_PATH);
        CheckFolder(lidar.OPT_PATH);
        CheckFolder(lidar.EDGE_PATH);
        CheckFolder(lidar.RESULT_PATH);
        // Data Pre Processing
        dataProcessing(omni, lidar, kOmniCamera, kOmniLiDAR);
        // Init Viz
        std::string fusion_image_path_init = omni.RESULT_PATH + "/fusion_image_init.png";
        std::string cocalib_result_path_init = lidar.RESULT_PATH + "/cocalib_init.txt";
        double proj_error = project2Image(omni, lidar, params_init, fusion_image_path_init, kNumIntrinsic, 0); // 0 - invalid bandwidth to initialize the visualization
        saveResults(cocalib_result_path_init, params_init, 0, 0, 0, proj_error);
        if (kRoughCalib && !kOmniCamera && !kOmniLiDAR) {
            cout << "----------------- Rough Calibration ---------------------" << endl;
            Vector7d ext_params;
            ext_params << alpha, rx, ry, rz, tx, ty, tz;
            roughCalib(omni, lidar, ext_params, DEG2RAD(0.8), 0.20, 5, 2);
            roughCalib(omni, lidar, ext_params, DEG2RAD(0.1), 0.05, 5, 2);
        }
        if (kCeresOpt) {
            cout << "----------------- Ceres Optimization ---------------------" << endl;
            for (int i = 0; i < bw.size(); i++) {
                double bandwidth = bw[i];
                vector<double> init_params_vec(params_cocalib);
                if (kOmniCamera) {
                    params_cocalib = QuaternionOmniCalib(omni, lidar, bandwidth, params_cocalib, lb, ub, false);
                    if (kParamsAnalysis) {
                        costAnalysis(omni, lidar, init_params_vec, params_cocalib, bandwidth);
                    }
                }
                else {
                    params_cocalib = QuaternionPinholeCalib(omni, lidar, bandwidth, params_cocalib, lb, ub, false);
                }
            }
        }
    }
    if (kMultiSpotOpt && kNumSpot != 1) {
        cout << "----------------- Multi Spot Optimization ---------------------" << endl;
        // Class Object Initialization
        OmniProcess cam_0("/multi_opt", 0);
        OmniProcess cam_1("/multi_opt", 1);
        OmniProcess cam_2("/multi_opt", 2);
        OmniProcess cam_3("/multi_opt", 3);
        OmniProcess cam_4("/multi_opt", 4);
        OmniProcess cam_5("/multi_opt", 5);
        OmniProcess cam_6("/multi_opt", 6);
        OmniProcess cam_7("/multi_opt", 7);
        LidarProcess lidar_0("/multi_opt", 0);
        LidarProcess lidar_1("/multi_opt", 1);
        LidarProcess lidar_2("/multi_opt", 2);
        LidarProcess lidar_3("/multi_opt", 3);
        LidarProcess lidar_4("/multi_opt", 4);
        LidarProcess lidar_5("/multi_opt", 5);
        LidarProcess lidar_6("/multi_opt", 6);
        LidarProcess lidar_7("/multi_opt", 7);
        std::vector<OmniProcess> cam_vec = {cam_0, cam_1, cam_2, cam_3, cam_4, cam_5, cam_6, cam_7};
        std::vector<LidarProcess> lidar_vec = {lidar_0, lidar_1, lidar_2, lidar_3, lidar_4, lidar_5, lidar_6, lidar_7};
        
        ROS_ASSERT_MSG((lidar_vec.size() == cam_vec.size()), "The number of spots are not equal for camera and lidar!");
        for (int idx = 0; idx < kNumSpot; idx++) {
            lidar_vec[idx].ext_ = Eigen::Map<Param_D1>(params_init.data()).head(7);
            if (kOmniCamera) {
                cam_vec[idx].omni_int_ = Eigen::Map<Param_D1>(params_init.data()).tail(kNumIntrinsic);
            }
            else {
                cam_vec[idx].pinhole_int_  = Eigen::Map<Param_D1>(params_init.data()).tail(kNumIntrinsic);
                lidar_vec[idx].pinhole_int_ = Eigen::Map<Param_D1>(params_init.data()).tail(kNumIntrinsic);
            }
            CheckFolder(lidar_vec[idx].DATASET_PATH);
            CheckFolder(lidar_vec[idx].OPT_PATH);
            CheckFolder(lidar_vec[idx].EDGE_PATH);
            CheckFolder(lidar_vec[idx].RESULT_PATH);
            dataProcessing(cam_vec[idx], lidar_vec[idx], kOmniCamera, kOmniLiDAR);
            // Init Viz
            std::string fusion_image_path_init = cam_vec[idx].RESULT_PATH + "/fusion_image_init.png";
            std::string cocalib_result_path_init = lidar_vec[idx].RESULT_PATH + "/cocalib_init.txt";
            double proj_error = project2Image(cam_vec[idx], lidar_vec[idx], params_init, fusion_image_path_init, kNumIntrinsic, 0); // 0 - invalid bandwidth to initialize the visualization
            saveResults(cocalib_result_path_init, params_init, 0, 0, 0, proj_error);
        }
        if (kRoughCalib && !kOmniCamera && !kOmniLiDAR) {
            cout << "----------------- Rough Calibration ---------------------" << endl;
            Vector7d ext_params;
            ext_params << alpha, rx, ry, rz, tx, ty, tz;
            roughCalib(cam_0, lidar_0, ext_params, DEG2RAD(0.8), 0.20, 5, 2);
            roughCalib(cam_0, lidar_0, ext_params, DEG2RAD(0.1), 0.05, 5, 2);
        }
        if (kCeresOpt) {
            cout << "----------------- Ceres Optimization ---------------------" << endl;
            for (int i = 0; i < bw.size(); i++) {
                double bandwidth = bw[i];
                vector<double> init_params_vec(params_cocalib);
                if (kOmniCamera) {
                    params_cocalib = QuaternionOmniMultiCalib(cam_vec, lidar_vec, bandwidth, params_cocalib, lb, ub, false);
                }
                else {
                    params_cocalib = QuaternionPinholeMultiCalib(cam_vec, lidar_vec, bandwidth, params_cocalib, lb, ub, false);
                }
            }
        }
    }
    return 0;
}
