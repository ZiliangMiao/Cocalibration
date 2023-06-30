// basic
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <sys/stat.h>

// eigen
#include <Eigen/Core>

// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// sensor_msg
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>

// headings
#include "define.h"

using namespace std;

int CheckFolder(std::string spot_path) {
    int md = 0; /** 0 means the folder is already exist or has been created successfully **/
    if (0 != access(spot_path.c_str(), 0)) {
        /** if this folder not exist, create a new one **/
        md = mkdir(spot_path.c_str(), S_IRWXU);
    }
    return md;
}

// int ReadFileList(std::string &folder_path, std::vector<std::string> &file_list) {
//     DIR *dp;
//     struct dirent *dir_path;
//     if ((dp = opendir(folder_path.c_str())) == nullptr) {
//         return 0;
//     }
//     int num = 0;
//     while ((dir_path = readdir(dp)) != nullptr) {
//         std::string name = std::string(dir_path->d_name);
//         if (name != "." && name != "..") {
//             file_list.push_back(name);
//             num++;
//         }
//     }
//     closedir(dp);
//     cout << "read file list success" << endl;
//     return num;
// }

template <typename PointType>
void loadPcd(string filepath, pcl::PointCloud<PointType> &cloud, const char* name="") {
    ROS_INFO("Loading %s cloud.\n Filepath: %s", name, filepath.c_str());
    int status = pcl::io::loadPCDFile<PointType>(filepath, cloud);
    if (MESSAGE_EN) {
        ROS_INFO("Loaded %ld points into %s cloud.\n", cloud.points.size(), name);
    }
}

Eigen::Matrix4f LoadTransMat(std::string trans_path){
    std::ifstream load_stream;
    load_stream.open(trans_path);
    Eigen::Matrix4f trans_mat = Eigen::Matrix4f::Identity();
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
            load_stream >> trans_mat(j, k);
        }
    }
    load_stream.close();
    return trans_mat;
}

template <typename T>
Eigen::Matrix<T, 4, 4> transformMat(Eigen::Matrix<T, 7, 1> &extrinsic){
    Eigen::Matrix<T, 3, 3> R = Eigen::AngleAxisd(extrinsic[0], Eigen::Matrix<T, 3, 1>(extrinsic[1], extrinsic[2], extrinsic[3])).toRotationMatrix();
    Eigen::Matrix<T, 4, 4> T_mat;
    T_mat << R(0,0), R(0,1), R(0,2), extrinsic(4),
        R(1,0), R(1,1), R(1,2), extrinsic(5),
        R(2,0), R(2,1), R(2,2), extrinsic(6),
        T(0.0), T(0.0), T(0.0), T(1.0);
    return T_mat;
}

// template <typename T>
// Eigen::Matrix<T, 4, 4> transformMat(Eigen::Matrix<T, 6, 1> &extrinsic){
//     /***** R = Rz * Ry * Rx *****/
//     Eigen::Matrix<T, 3, 3> R;
//     R = Eigen::AngleAxis<T>(extrinsic(2), Eigen::Matrix<T, 3, 1>::UnitZ())
//         * Eigen::AngleAxis<T>(extrinsic(1), Eigen::Matrix<T, 3, 1>::UnitY())
//         * Eigen::AngleAxis<T>(extrinsic(0), Eigen::Matrix<T, 3, 1>::UnitX());

//     Eigen::Matrix<T, 4, 4> T_mat;
//     T_mat << R(0,0), R(0,1), R(0,2), extrinsic(3),
//         R(1,0), R(1,1), R(1,2), extrinsic(4),
//         R(2,0), R(2,1), R(2,2), extrinsic(5),
//         T(0.0), T(0.0), T(0.0), T(1.0);
//     return T_mat;
// }

// Intrinsic Transformation for Pinhole Camera
template <typename T>
Eigen::Matrix<T, 2, 1> IntrinsicTransform(Eigen::Matrix<T, K_PINHOLE_INT, 1> &intrinsic, Eigen::Matrix<T, 3, 1> &point){
    Eigen::Matrix<T, 2, 1> distorted_projection;

    T xo = point[0] / point[2];
    T yo = point[1] / point[2];
    T r2 = xo * xo + yo * yo;
    T r4 = r2 * r2;
    T distortion = 1.0 + intrinsic[3] * r2 + intrinsic[4] * r4;

    T xd = xo * distortion;
    T yd = yo * distortion;    
    T ud = intrinsic[0] * xd + intrinsic[1];
    T vd = intrinsic[0] * yd + intrinsic[2];
    distorted_projection[0] = vd;
    distorted_projection[1] = ud;

    return distorted_projection;
}

// Intrinsic Transformation for Omnidirectional Camera
// template <typename T>
// Eigen::Matrix<T, 2, 1> IntrinsicTransform(Eigen::Matrix<T, 7, 1> &intrinsic, Eigen::Matrix<T, 3, 1> &point){
    
//     Eigen::Matrix<T, K_OMNI_INT, 1> intrinsic_;
//     intrinsic_.head(7) = intrinsic;
//     intrinsic_.tail(3) << T(1), T(0), T(0);
//     Eigen::Matrix<T, 2, 1> projection = IntrinsicTransform(intrinsic_, point);

//     return projection;
// }

template <typename T>
Eigen::Matrix<T, 2, 1> IntrinsicTransform(Eigen::Matrix<T, K_OMNI_INT, 1> &intrinsic, Eigen::Matrix<T, 3, 1> &point){
    
    Eigen::Matrix<T, 2, 1> uv_0{intrinsic(0), intrinsic(1)};
    Eigen::Matrix<T, 5, 1> a_;
    Eigen::Matrix<T, 2, 2> affine;
    Eigen::Matrix<T, 2, 2> affine_inv;
    T theta, xy_radius, uv_radius;
    Eigen::Matrix<T, 2, 1> projection;
    Eigen::Matrix<T, 2, 1> undistorted_projection;

    a_ << intrinsic(2), intrinsic(3), intrinsic(4), intrinsic(5), intrinsic(6);
    affine << intrinsic(7), intrinsic(8), intrinsic(9), T(1);

    theta = acos(point(2) / sqrt((point(0) * point(0)) + (point(1) * point(1)) + (point(2) * point(2))));
    uv_radius = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4);
    xy_radius = sqrt(point(1) * point(1) + point(0) * point(0));
    projection = {uv_radius / xy_radius * point(0) + uv_0(0), uv_radius / xy_radius * point(1) + uv_0(1)};
    affine_inv.row(0) << affine(1, 1) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1)), - affine(0, 1) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1));
    affine_inv.row(1) << - affine(1, 0) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1)), affine(0, 0) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1));
    undistorted_projection = affine_inv * projection;
    return undistorted_projection;
}

void saveResults(std::string &record_path, std::vector<double> params, double bandwidth, double initial_cost, double final_cost, double proj_error) {
    const std::vector<const char*> name = {
            "alpha", "rx", "ry", "rz",
            "tx", "ty", "tz",
            "f", "cx",
            "cy", "k1", "k2"};

    ofstream write;
    const bool title = (bandwidth <= 0);

    std::cout << "Params size: " << std::endl;

    std::string output = title ? ("Parameters:\n") : ("Bandwidth = " + to_string(int(bandwidth)) + ":\n");
    ios_base::openmode mode = title ? (ios::out) : (ios::app);
    
    output += "Result:\n[";
    for (int i = 0; i < name.size(); i++) {
        if (i > 0) {
            output += ", ";
        }
        output += title ? (name[i]) : (to_string(params[i]));
    }
    output += "]\n";
    if (!title) {
        output += "Initial cost: " + to_string(initial_cost) + "\n" +
                "Final cost: " + to_string(final_cost) + "\n" + 
                "Average projection error: " + to_string(proj_error) + "\n";
    }
    
    write.open(record_path, mode);
    write << output << endl;
    write.close();
    
    ROS_INFO("%s", output.c_str());
}
