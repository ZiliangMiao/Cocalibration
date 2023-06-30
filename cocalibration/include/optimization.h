// basic
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <thread>
// eigen
#include <Eigen/Core>
// ros
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <ros/package.h>
// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
// ceres
#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/rotation.h"
#include "glog/logging.h"
// pcl
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
// headings
#include <omni_process.h>
#include <lidar_process.h>
#include <define.h>

using namespace std;

#define Q_LIM   (0.30)

inline double getDouble(double x) {
    return static_cast<double>(x);
}

template <typename SCALAR, int N>
inline double getDouble(const ceres::Jet<SCALAR, N> &x) {
    return static_cast<double>(x.a);
}

double project2Image(OmniProcess &fisheye, LidarProcess &lidar, std::vector<double> &params, std::string record_path, int kNumIntrinsic, double bandwidth);

std::vector<double> QuaternionOmniCalib(OmniProcess &fisheye,
                                    LidarProcess &lidar,
                                    double bandwidth,
                                    std::vector<double> init_params_vec,
                                    std::vector<double> lb,
                                    std::vector<double> ub,
                                    bool lock_intrinsic);

std::vector<double> QuaternionPinholeCalib(OmniProcess &fisheye,
                                    LidarProcess &lidar,
                                    double bandwidth,
                                    std::vector<double> init_params_vec,
                                    std::vector<double> lb,
                                    std::vector<double> ub,
                                    bool lock_intrinsic);

std::vector<double> QuaternionOmniMultiCalib( 
                                    std::vector<OmniProcess> cam_vec,
                                    std::vector<LidarProcess> lidar_vec,
                                    double bandwidth,
                                    std::vector<double> init_params_vec,
                                    std::vector<double> lb,
                                    std::vector<double> ub,
                                    bool lock_intrinsic);

std::vector<double> QuaternionPinholeMultiCalib( 
                                    std::vector<OmniProcess> cam_vec,
                                    std::vector<LidarProcess> lidar_vec,
                                    double bandwidth,
                                    std::vector<double> init_params_vec,
                                    std::vector<double> lb,
                                    std::vector<double> ub,
                                    bool lock_intrinsic);

void costAnalysis(OmniProcess &fisheye,
                        LidarProcess &lidar,
                        std::vector<double> init_params_vec,
                        std::vector<double> result_vec,
                        double bandwidth);