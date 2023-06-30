/** headings **/
#include <lidar_process.h>
#include <common_lib.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>

/** namespace **/
using namespace std;
using namespace cv;
using namespace Eigen;

LidarProcess::LidarProcess(std::string opt_path, int spot_idx){
    /** Param **/
    ros::param::get("essential/kDatasetName", this->DATASET_NAME);
    ros::param::get("essential/kNumSpot", this->NUM_SPOT);
    ros::param::get("essential/kLidarTopic", this->TOPIC_NAME);
    ros::param::get("essential/kFlatRows", this->kFlatImageSize.first);
    ros::param::get("essential/kFlatCols", this->kFlatImageSize.second);

    this->lidarCartCloud.reset(new pcl::PointCloud<PointI>);
    this->lidarPlaneCloud.reset(new pcl::PointCloud<PointI>);
    this->lidarPolarCloud.reset(new pcl::PointCloud<PointI>);
    this->lidarEdgeCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    /** Path **/
    // this->PKG_PATH = ros::package::getPath("cocalibration");
    this->PKG_PATH = "/home/yoga/projects/catkin_ws/src/Hybrid_Mapping_Cocalibration/cocalibration";
    this->DATASET_PATH = this->PKG_PATH + "/data/" + this->DATASET_NAME;
    this->OPT_PATH = this->DATASET_PATH + opt_path;
    this->PYSCRIPT_PATH = this->PKG_PATH + "/python_scripts/image_process/edge_extraction.py";

    this->EDGE_PATH = this->OPT_PATH + "/edges_" + std::to_string(spot_idx);
    this->RESULT_PATH = this->OPT_PATH + "/results_" + std::to_string(spot_idx);
    this->cocalibCloudPath = this->OPT_PATH + "/lidar_cloud_" + std::to_string(spot_idx) + ".pcd";
    this->flatImagePath = this->OPT_PATH + "/flat_lidar_image_" + std::to_string(spot_idx) + ".png";
    this->lidarEdgeCloudPath = this->EDGE_PATH + "/lidar_edge_clou.pcd";
    this->lidarEdgeImagePath = this->EDGE_PATH + "/lidar_edge_image.png";
    this->lidarPolarCloudPath = this->OPT_PATH + "/lidar_polar_cloud_" + std::to_string(spot_idx) + ".pcd";
    this->lidarPlaneCloudPath = this->OPT_PATH + "/lidar_plane_cloud_" + std::to_string(spot_idx) + ".pcd";
}

/** Data Pre-processing **/
void LidarProcess::cartToPlane() {// for non-omni lidar only, intrinsic transformation only
    cout << "----- LiDAR: CartToPlane -----" << endl;
    pcl::io::loadPCDFile(this->cocalibCloudPath, *this->lidarCartCloud);

    // pcl::CropBox<PointI> box_filter;
    // box_filter.setMin(Eigen::Vector4f(-25.0, -25.0, -25.0, 1.0));
    // box_filter.setMax(Eigen::Vector4f(25.0, 25.0, 25.0, 1.0));
    // box_filter.setNegative(false);
    // box_filter.setInputCloud(this->lidarCartCloud);
    // box_filter.filter(*this->lidarCartCloud);
    // pcl::io::savePCDFileBinary(this->SINGLE_OPT_PATH + "/lidar_cart_box_filtered.pcd", *this->lidarCartCloud);
    // pcl::PointXYZI min; //用于存放三个轴的最小值
    // pcl::PointXYZI max; //用于存放三个轴的最大值
    // pcl::getMinMax3D(*this->lidarCartCloud, min, max);
    // float height = this->kFlatImageSize.first;
    // float width = this->kFlatImageSize.second;
    // pcl::copyPointCloud(*this->lidarCartCloud, *this->lidarPlaneCloud);
    //   for (auto &point : this->lidarPlaneCloud->points) {
    //     point.x = 0;
    //     point.y = (point.y-min.y) / (max.y-min.y) * width;
    //     point.z = (point.z-min.z) / (max.z-min.z) * height;
    // }
    // pcl::io::savePCDFileBinary(this->SINGLE_OPT_PATH + "/lidar_plane_cloud.pcd", *this->lidarPlaneCloud);

    // Project the lidar points to image plane by intrinsic and extrinsic transformation
    pcl::copyPointCloud(*this->lidarCartCloud, *this->lidarPlaneCloud);
    Eigen::Matrix<float, 4, 4> T_mat;
    T_mat << 0.0142149, -0.999558, -0.0261058, 0.00671375,
        -0.0218795, 0.0257912, -0.999428, 0.0650754,
        0.99966, 0.0147779, -0.0215032, -0.0141875,
        0.0, 0.0, 0.0, 1.0;
    pcl::transformPointCloud(*this->lidarPlaneCloud, *this->lidarPlaneCloud, T_mat);
    double f = 2561.3;
    double cx = 1180.2;
    double cy = 1035.9;
    double k1 = -0.1474;
    double k2 = 0.1150;
    for (auto &point : this->lidarPlaneCloud->points) {
        float xo = point.x / point.z;
        float yo = point.y / point.z;
        float r2 = xo * xo + yo * yo;
        float r4 = r2 * r2;
        float distortion = 1.0 + k1 * r2 + k2 * r4;
        float xd = xo * distortion;
        float yd = yo * distortion;    
        float ud = f * xd + cx;
        float vd = f * yd + cy;
        point.x = vd;
        point.y = ud;
        point.z = 0;
    }
    pcl::io::savePCDFileBinary(this->lidarPlaneCloudPath, *this->lidarPlaneCloud);

    // pcl::CropBox<PointI> plane_box_filter;
    // plane_box_filter.setMin(Eigen::Vector4f(0.0, 0.0, -1.0, 1.0));
    // plane_box_filter.setMax(Eigen::Vector4f(2048, 2448, 1.0, 1.0)); //此处x是vd，是图像的高度 
    // plane_box_filter.setNegative(false);
    // plane_box_filter.setInputCloud(this->lidarPlaneCloud);
    // plane_box_filter.filter(*this->lidarPlaneCloud);
    // pcl::io::savePCDFileBinary(this->SINGLE_OPT_PATH + "/lidar_plane_box_filtered.pcd", *this->lidarPlaneCloud);
    // pcl::PointXYZI min; //用于存放三个轴的最小值
    // pcl::PointXYZI max; //用于存放三个轴的最大值
    // pcl::getMinMax3D(*this->lidarPlaneCloud, min, max);
    // cout << "min x: " << min.x <<  "max x: " << max.x << endl;
    // cout << "min y: " << min.y <<  "max y: " << max.y << endl;

    /** define the data container **/
    cv::Mat flat_img = cv::Mat::zeros(this->kFlatImageSize.first, this->kFlatImageSize.second, CV_8U);
    vector<vector<Tags>> tags_map (this->kFlatImageSize.first, vector<Tags>(this->kFlatImageSize.second));
    /** construct kdtrees and load the point clouds **/
    /** caution: the point cloud need to be set before the loop **/
    pcl::KdTreeFLANN<PointI> kdtree;
    kdtree.setInputCloud(this->lidarPlaneCloud);

    /** define the invalid search parameters **/
    int invalid_search_num, valid_search_num = 0; /** search invalid count **/
    int invalid_idx_num = 0; /** index invalid count **/
    const float kSearchRadius = 4; // range from 2 to 4
    const float sensitivity = 0.02f;

    #pragma omp parallel for num_threads(THREADS)
    for (int u = 0; u < this->kFlatImageSize.first; ++u) {
        for (int v = 0; v < this->kFlatImageSize.second; ++v) {
            /** assign the theta and phi center to the search_center **/
            PointI search_center;
            search_center.x = 4*u + 1;
            search_center.y = 4*v + 1;
            search_center.z = 0;
            vector<int> tag;
            /** define the vector container for storing the info of searched points **/
            vector<int> search_pt_idx_vec;
            vector<float> search_pt_squared_dis_vec; /** type of distance vector has to be float **/
            /** use kdtree to search (radius search) the spherical point cloud **/
            int search_num = kdtree.radiusSearch(search_center, kSearchRadius, search_pt_idx_vec, search_pt_squared_dis_vec); // number of the radius nearest neighbors
            if (search_num == 0) {
                flat_img.at<uint8_t>(u, v) = 0; /** intensity **/
                invalid_search_num ++;
            }
            else { /** corresponding points are found in the radius neighborhood **/
                int hidden_pt_num = 0;
                float dist_mean = 0;
                float intensity_mean = 0;
                vector<int> local_vec(search_num, 0);
                for (int i = 0; i < search_num; ++i) {
                    dist_mean += this->lidarPlaneCloud->points[search_pt_idx_vec[i]].z;
                }
                dist_mean = dist_mean / search_num;
                for (int i = 0; i < search_num; ++i) {
                    PointI &local_pt = this->lidarPlaneCloud->points[search_pt_idx_vec[i]];
                    float dist = local_pt.z;
                    if ((abs(dist_mean - dist) > dist * sensitivity) || ((dist_mean - dist) > dist * sensitivity && local_pt.intensity < 20)) {
                        hidden_pt_num++;
                    }
                    else {
                        intensity_mean += local_pt.intensity;
                        local_vec[i] = search_pt_idx_vec[i];
                    }
                }
                /** add tags **/
                local_vec.erase(std::remove(local_vec.begin(), local_vec.end(), 0), local_vec.end());
                tag.insert(tag.begin(), local_vec.data(), local_vec.data()+local_vec.size());
                if (tag.size() > 0) {
                    intensity_mean /= tag.size();
                }                
                flat_img.at<uchar>(u, v) = static_cast<uchar>(intensity_mean);
            }
            tags_map[u][v] = tag;
        }
    }
    this->tagsMap = tags_map;
    cout << "invalid_search_num: " << invalid_search_num << endl;
    cv::imwrite(this->flatImagePath, flat_img);
}

void LidarProcess::cartToSphere() {
    cout << "----- LiDAR: CartToSphere -----" << endl;
    float theta_min = M_PI, theta_max = -M_PI;
    pcl::io::loadPCDFile(this->cocalibCloudPath, *this->lidarCartCloud);
    pcl::copyPointCloud(*this->lidarCartCloud, *this->lidarPolarCloud);
    for (auto &point : this->lidarPolarCloud->points) {
        float radius = point.getVector3fMap().norm();
        float phi = atan2(point.y, point.x);
        float theta = acos(point.z / radius);
        point.x = theta;
        point.y = phi;
        point.z = radius;
        if (theta > theta_max) { theta_max = theta;}
        else if (theta < theta_min) { theta_min = theta;}
    }
    if (MESSAGE_EN) {
        ROS_INFO("Polar cloud generated. \ntheta: (min, max) = (%f, %f)", theta_min, theta_max);
    }
    pcl::io::savePCDFileBinary(this->lidarPolarCloudPath, *this->lidarPolarCloud);
}

void LidarProcess::sphereToPlane() {
    cout << "----- LiDAR: SphereToPlane -----" << endl;
    /** define the data container **/
    cv::Mat flat_img = cv::Mat::zeros(this->kFlatImageSize.first, this->kFlatImageSize.second, CV_8U);
    vector<vector<Tags>> tags_map (this->kFlatImageSize.first, vector<Tags>(this->kFlatImageSize.second));
    /** construct kdtrees and load the point clouds **/
    /** caution: the point cloud need to be set before the loop **/
    pcl::KdTreeFLANN<PointI> kdtree;
    CloudI::Ptr polar_flat_cloud(new CloudI);
    pcl::copyPointCloud(*this->lidarPolarCloud, *polar_flat_cloud);
    for (auto &pt : polar_flat_cloud->points) {pt.z = 0;}
    kdtree.setInputCloud(polar_flat_cloud);

    /** define the invalid search parameters **/
    int invalid_search_num, valid_search_num = 0; /** search invalid count **/
    int invalid_idx_num = 0; /** index invalid count **/
    const float kSearchRadius = sqrt(2) * (kRadPerPix / 2);
    const float sensitivity = 0.02f;

    #pragma omp parallel for num_threads(THREADS)
    for (int u = 0; u < this->kFlatImageSize.first; ++u) {
        float theta_center = - kRadPerPix * (2 * u + 1) / 2 + M_PI;
        for (int v = 0; v < this->kFlatImageSize.second; ++v) {
            float phi_center = kRadPerPix * (2 * v + 1) / 2 - M_PI;
            /** assign the theta and phi center to the search_center **/
            PointI search_center;
            search_center.x = theta_center;
            search_center.y = phi_center;
            search_center.z = 0;
            vector<int> tag;
            /** define the vector container for storing the info of searched points **/
            vector<int> search_pt_idx_vec;
            vector<float> search_pt_squared_dis_vec; /** type of distance vector has to be float **/
            /** use kdtree to search (radius search) the spherical point cloud **/
            int search_num = kdtree.radiusSearch(search_center, kSearchRadius, search_pt_idx_vec, search_pt_squared_dis_vec); // number of the radius nearest neighbors
            if (search_num == 0) {
                flat_img.at<uint8_t>(u, v) = 0; /** intensity **/
                invalid_search_num ++;
            }
            else { /** corresponding points are found in the radius neighborhood **/
                int hidden_pt_num = 0;
                float dist_mean = 0;
                float intensity_mean = 0;
                vector<int> local_vec(search_num, 0);
                for (int i = 0; i < search_num; ++i) {
                    dist_mean += this->lidarPolarCloud->points[search_pt_idx_vec[i]].z;
                }
                dist_mean = dist_mean / search_num;
                for (int i = 0; i < search_num; ++i) {
                    PointI &local_pt = this->lidarPolarCloud->points[search_pt_idx_vec[i]];
                    float dist = local_pt.z;
                    if ((abs(dist_mean - dist) > dist * sensitivity) || ((dist_mean - dist) > dist * sensitivity && local_pt.intensity < 20)) {
                        hidden_pt_num++;
                    }
                    else {
                        intensity_mean += local_pt.intensity;
                        local_vec[i] = search_pt_idx_vec[i];
                    }
                }
                /** add tags **/
                local_vec.erase(std::remove(local_vec.begin(), local_vec.end(), 0), local_vec.end());
                tag.insert(tag.begin(), local_vec.data(), local_vec.data()+local_vec.size());
                if (tag.size() > 0) {
                    intensity_mean /= tag.size();
                }                
                flat_img.at<uchar>(u, v) = static_cast<uchar>(intensity_mean);
            }
            tags_map[u][v] = tag;
        }
    }
    this->tagsMap = tags_map;
    cv::imwrite(this->flatImagePath, flat_img);
}

// void LidarProcess::cartToSphere() {
//     cout << "----- LiDAR: CartToSphere -----" << endl;
//     float theta_min = M_PI, theta_max = -M_PI;
//     pcl::io::loadPCDFile(this->cocalibCloudPath, *this->lidarCartCloud);
//     pcl::copyPointCloud(*this->lidarCartCloud, *this->lidarPolarCloud);
//     for (auto &point : this->lidarPolarCloud->points) {
//         // change the projection approach
//         double fov_up = 180.0 / 180.0 * M_PI;  // field of view up in radians
//         double fov_down = -120.0 / 180.0 * M_PI;  // field of view down in radians
//         double fov = abs(fov_down) + abs(fov_up);  // get field of view total in radians

//         float depth = point.getVector3fMap().norm();
//         float phi = atan2(point.y, point.x);
//         float theta = acos(point.z / depth);
//         // get projections in image coords
//         double proj_x = (0.5 * (1.0 + phi / M_PI)) * this->kFlatImageSize.second;  // in [0.0, 1.0 * W]
//         double proj_y = (1.0 - (theta + abs(fov_down)) / fov) * this->kFlatImageSize.first;  // in [0.0, 1.0 * H]

//         point.x = proj_x;
//         point.y = proj_y;
//         point.z = 0;
//         if (theta > theta_max) { theta_max = theta;}
//         else if (theta < theta_min) { theta_min = theta;}
//     }
//     if (MESSAGE_EN) {
//         ROS_INFO("Polar cloud generated. \ntheta: (min, max) = (%f, %f)", theta_min, theta_max);
//     }
//     pcl::io::savePCDFileBinary(this->SINGLE_OPT_PATH + "/lidar_polar_cloud.pcd", *this->lidarPolarCloud);
// }

// void LidarProcess::sphereToPlane() {
//     cout << "----- LiDAR: SphereToPlane -----" << endl;
//     /** define the data container **/
//     cv::Mat flat_img = cv::Mat::zeros(this->kFlatImageSize.first, this->kFlatImageSize.second, CV_8U);
//     vector<vector<Tags>> tags_map (this->kFlatImageSize.first, vector<Tags>(this->kFlatImageSize.second));
//     /** construct kdtrees and load the point clouds **/
//     /** caution: the point cloud need to be set before the loop **/
//     pcl::KdTreeFLANN<PointI> kdtree;
//     kdtree.setInputCloud(this->lidarPolarCloud);

//     /** define the invalid search parameters **/
//     int invalid_search_num, valid_search_num = 0; /** search invalid count **/
//     int invalid_idx_num = 0; /** index invalid count **/
//     const float kSearchRadius = sqrt(2) * (kRadPerPix / 2);
//     const float sensitivity = 0.02f;
//     // modify the search center!!!!! important
//     #pragma omp parallel for num_threads(THREADS)
//     for (int u = 0; u < this->kFlatImageSize.first; ++u) {
//         float theta_center = - kRadPerPix * (2 * u + 1) / 2 + M_PI;
//         for (int v = 0; v < this->kFlatImageSize.second; ++v) {
//             float phi_center = kRadPerPix * (2 * v + 1) / 2 - M_PI;
//             /** assign the theta and phi center to the search_center **/
//             PointI search_center;
//             search_center.x = theta_center;
//             search_center.y = phi_center;
//             search_center.z = 0;
//             vector<int> tag;
//             /** define the vector container for storing the info of searched points **/
//             vector<int> search_pt_idx_vec;
//             vector<float> search_pt_squared_dis_vec; /** type of distance vector has to be float **/
//             /** use kdtree to search (radius search) the spherical point cloud **/
//             int search_num = kdtree.radiusSearch(search_center, kSearchRadius, search_pt_idx_vec, search_pt_squared_dis_vec); // number of the radius nearest neighbors
//             if (search_num == 0) {
//                 flat_img.at<uint8_t>(u, v) = 0; /** intensity **/
//                 invalid_search_num ++;
//             }
//             else { /** corresponding points are found in the radius neighborhood **/
//                 float intensity_mean = 0;
//                 vector<int> local_vec(search_num, 0);
//                 for (int i = 0; i < search_num; ++i) {
//                     PointI &local_pt = this->lidarPolarCloud->points[search_pt_idx_vec[i]];
//                     float dist = local_pt.z;
//                     intensity_mean += local_pt.intensity;
//                     local_vec[i] = search_pt_idx_vec[i];
//                 }
//                 /** add tags **/
//                 local_vec.erase(std::remove(local_vec.begin(), local_vec.end(), 0), local_vec.end());
//                 tag.insert(tag.begin(), local_vec.data(), local_vec.data()+local_vec.size());
//                 if (tag.size() > 0) {
//                     intensity_mean /= tag.size();
//                 }                
//                 flat_img.at<uchar>(u, v) = static_cast<uchar>(intensity_mean);
//             }
//             tags_map[u][v] = tag;
//         }
//     }
//     this->tagsMap = tags_map;
//     cv::imwrite(this->flatImagePath, flat_img);
// }

void LidarProcess::edgeExtraction(string image_data_path, string edge_path, string mode) {
    cout << "----- LiDAR: Python EdgeExtraction -----" << endl;
    string cmd_str = "python3 " + this->PYSCRIPT_PATH + " " + image_data_path + " " + edge_path + " " + mode;
    int status = system(cmd_str.c_str());
}

void LidarProcess::generateEdgeCloud() {
    cout << "----- LiDAR: GenerateEdgeCloud -----" << endl;
    cv::Mat edge_img = cv::imread(this->lidarEdgeImagePath, cv::IMREAD_UNCHANGED);
    ROS_ASSERT_MSG((edge_img.rows != 0 && edge_img.cols != 0), "size of original fisheye image is 0, check the path and filename! \nPath: %s", this->lidarEdgeImagePath.c_str());
    ROS_ASSERT_MSG((edge_img.rows == this->kFlatImageSize.first || edge_img.cols == kFlatImageSize.second), "size of original fisheye image is incorrect!");

    CloudI::Ptr edge_xyzi (new CloudI);
    for (int u = 0; u < edge_img.rows; ++u) {
        for (int v = 0; v < edge_img.cols; ++v) {
            if (edge_img.at<uchar>(u, v) > 127) {
                Tags &tag = this->tagsMap[u][v];
                for (int i = 0; i < tag.size(); ++i) { 
                    PointI &pixel_pt = this->lidarCartCloud->points[tag[i]];
                    edge_xyzi->points.push_back(pixel_pt);
                }
            }
        }
    }

    /** uniform sampling **/
    pcl::UniformSampling<PointI> us;
    us.setRadiusSearch(0.005);
    us.setInputCloud(edge_xyzi);
    us.filter(*edge_xyzi);

    pcl::copyPointCloud(*edge_xyzi, *this->lidarEdgeCloud);
    pcl::io::savePCDFileBinary(this->lidarEdgeCloudPath, *this->lidarEdgeCloud);
}

double LidarProcess::getEdgeDistance(EdgeCloud::Ptr cloud_tgt, EdgeCloud::Ptr cloud_src, float max_range) {
    cout << "----- GetEdgeDistance -----" << endl;
    pcl::StopWatch timer_fs;
    vector<float> dists;
    int valid_cnt = 0;
    float avg_dist = 0;
    float outlier_percentage = 0.1;

    std::vector<int> nn_indices(1);
    std::vector<float> nn_dists(1);
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_tgt);

    for (auto &pt : cloud_src->points) {
        kdtree.nearestKSearch(pt, 1, nn_indices, nn_dists);
        if (nn_dists[0] <= max_range) {
            dists.push_back(nn_dists[0]);
        }
    }

    if (dists.size() * outlier_percentage > 1) {
        sort(dists.data(), dists.data()+dists.size());
        for (size_t i = 0; i < dists.size() * (1-outlier_percentage); i++) {
            avg_dist += dists[i];
            ++valid_cnt;
        }
        if (valid_cnt > 0) {
            avg_dist /= valid_cnt;
            ROS_INFO("Average projection error: %f", avg_dist);
        } 
    }
    return avg_dist;

}

// double LidarProcess::getFitnessScore(CloudI::Ptr cloud_tgt, CloudI::Ptr cloud_src, float max_range) {
//     double fitness_score = 0.0;
//     std::vector<int> nn_indices(1);
//     std::vector<float> nn_dists(1);
//     // For each point in the source dataset
//     int nr = 0;
//     pcl::KdTreeFLANN<PointI> kdtree;
//     kdtree.setInputCloud(cloud_tgt);
//     #pragma omp parallel for num_threads(THREADS)
//     for (auto &pt : cloud_src->points) {
//         // Find its nearest neighbor in the target
//         kdtree.nearestKSearch(pt, 1, nn_indices, nn_dists);
//         // Deal with occlusions (incomplete targets)
//         if (nn_dists[0] <= max_range) {
//             // Add to the fitness score
//             fitness_score += nn_dists[0];
//             nr++;
//         }
//     }
//     if (nr > 0)
//         return (fitness_score / nr);
//     return (std::numeric_limits<double>::max());
// }

void LidarProcess::removeInvalidPoints(CloudI::Ptr cloud){
    std::vector<int> null_indices;
    (*cloud).is_dense = false;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, null_indices);
}
