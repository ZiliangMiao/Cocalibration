/** basic **/
#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <tuple>
#include <numeric>
/** ros **/
#include <ros/ros.h>
#include <ros/package.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
/** pcl **/
#include <pcl/common/common.h>
#include <pcl/common/time.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/gicp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
/** opencv **/
#include <opencv2/opencv.hpp>
/** headings **/
#include <define.h>
/** namespace **/
using namespace std;

class LidarProcess{
public:
    CloudI::Ptr lidarCartCloud;
    CloudI::Ptr lidarPolarCloud;
    CloudI::Ptr lidarPlaneCloud;
    EdgeCloud::Ptr lidarEdgeCloud; // lidar edge points
    int NUM_SPOT = 1;
    string TOPIC_NAME = "/livox/lidar";
    Pair kFlatImageSize = {2048, 2448};
    const float kRadPerPix = (M_PI * 2) / kFlatImageSize.second;
    /** File Directory Path **/
    string DATASET_NAME;
    string PKG_PATH;
    string DATASET_PATH;
    string OPT_PATH;
    string EDGE_PATH;
    string RESULT_PATH;
    string PYSCRIPT_PATH; 

    string cocalibCloudPath;
    string flatImagePath;
    string lidarEdgeCloudPath;
    string lidarEdgeImagePath;
    string lidarPolarCloudPath; // for mid360 lidar
    string lidarPlaneCloudPath; // for avia lidar
    /** tags and maps **/
    typedef vector<int> Tags;
    vector<vector<Tags>> tagsMap;
    /***** Extrinsic Parameters *****/
    Ext_D ext_;
    Pinhole_Int_D pinhole_int_;

public:
    /** Funcs **/
    LidarProcess(std::string opt_path, int spot_idx);
    void cartToPlane(); // for non-omni lidar only
    void cartToSphere();
    void sphereToPlane();
    void edgeExtraction(string image_data_path, string edge_path, string mode);
    void generateEdgeCloud();
    /***** Evaluation *****/
    double getEdgeDistance(EdgeCloud::Ptr cloud_tgt, EdgeCloud::Ptr cloud_src, float max_range);
    // double getFitnessScore(CloudI::Ptr cloud_tgt, CloudI::Ptr cloud_src, float max_range);
    void removeInvalidPoints(CloudI::Ptr cloud);
    void computeCovariances(pcl::PointCloud<PointI>::ConstPtr cloud,
                            const pcl::search::KdTree<PointI>::Ptr kdtree,
                            MatricesVector& cloud_covariances);
};


