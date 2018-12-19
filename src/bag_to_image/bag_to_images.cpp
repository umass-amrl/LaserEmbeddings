#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include "ros/ros.h"
#include "ros/package.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/LaserScan.h"
#include "../perception_tools/perception_2d.h"
#include <CImg.h>
#include "shared_structs.h"

using std::string;
using std::vector;

vector<vector<float>> all_templates_;
vector<float> all_orientations_;
vector<float> all_depths_;

//TODO: max laser range needs to be set in a single place and used with a single
//name. Currently, there are two ways of setting / using max range

unsigned int clip_l = 0;
unsigned int clip_u = 0;
//unsigned int clip_l = 28;
//unsigned int clip_u = 29;
float min_range = 0.0;
float max_range = 10.0;
string bag_name;

void initTemplates() {
  // Human Dirk template: 36 points
  //2.807 2.69 2.627 2.592 2.576 2.568 2.565 2.563 2.565 2.583 2.609 2.674 2.768 2.893 3.913 3.919 3.904 3.876 3.856 3.827 3.799
  //3.784 3.774 2.775 2.666 2.582 2.552 2.517 2.512 2.509 2.518 2.526 2.541 2.595 2.669 2.811


  // Couch template: 176 points
  //2.252 2.238 2.23 2.205 2.203 2.2 2.192 2.18 2.177 2.177 2.17 2.152 2.145 2.14 2.139 2.121 2.108 2.107 2.098 2.093 2.086 2.083
  //2.063 2.062 2.056 2.053 2.044 2.038 2.03 2.019 2.015 2.014 2.005 1.99 1.987 1.982 1.972 1.975 1.977 1.977 1.971 1.953 1.947
  //1.948 1.946 1.938 1.934 1.916 1.917 1.915 1.91 1.906 1.901 1.899 1.894 1.894 1.887 1.875 1.876 1.883 1.881 1.873 1.875 1.871
  //1.87 1.867 1.865 1.864 1.862 1.889 1.896 1.894 1.921 1.94 1.958 1.972 1.986 2.021 2.026 2.05 2.051 2.07 2.081 2.095 2.126
  //2.131 2.142 2.159 2.166 2.19 2.197 2.213 2.23 2.25 2.26 2.275 2.277 2.297 2.309 2.345 2.352 2.365 2.379 2.393 2.404 2.398
  //2.427 2.45 2.457 2.468 2.471 2.497 2.506 2.511 2.527 2.532 2.547 2.574 2.574 2.592 2.607 2.621 2.619 2.644 2.666 2.668 2.683
  //2.7 2.709 2.728 2.732 2.742 2.753 2.755 2.773 2.781 2.793 2.809 2.811 2.812 2.837 2.841 2.856 2.865 2.877 2.887 2.905 2.911
  //2.917 2.93 2.946 2.959 2.965 2.967 2.996 2.995 3 2.997 3.029 3.031 3.042 3.052 3.067 3.079 3.096 3.103 3.109 3.108 3.126
  //3.126 3.14 3.141 3.162 3.162 3.178 3.233

  // Left Turn Hallway, Column, and Two Doors template: 348 points
  //4.66 4.658 4.658 4.66 4.66 4.659 4.668 4.674 4.674 4.672 4.732 4.747 4.77 4.778 4.786 4.788 4.776 4.767 4.766 4.766 4.768 
  //4.772 4.776 4.775 4.775 4.776 4.781 4.787 4.788 4.786 4.783 4.791 4.799 4.8 4.799 4.798 4.801 4.806 4.806 4.809 4.824 4.812
  //4.823 4.831 4.84 4.841 4.837 4.841 4.844 4.85 4.853 4.854 4.854 4.842 4.84 4.823 4.808 4.799 4.809 4.809 4.812 4.811 4.821
  //4.825 4.867 4.937 4.943 4.95 4.948 4.964 4.969 4.976 4.982 4.987 4.994 5.006 5.017 5.019 5.027 5.038 5.051 5.064 5.06 5.076
  //5.075 5.081 5.087 5.094 5.105 5.11 5.116 5.127 5.135 5.153 5.168 5.177 5.179 5.191 5.198 5.202 5.208 5.22 5.233 10 10 10 10
  //2.464 2.511 2.485 2.465 2.413 2.384 2.364 2.347 2.32 2.301 2.286 2.27 2.249 2.237 2.222 2.214 2.188 2.182 2.166 2.139 2.127
  //2.122 2.1 2.084 2.073 2.061 2.046 2.044 2.037 2.003 1.997 1.993 1.978 1.957 1.943 1.936 1.922 1.917 1.912 1.889 1.879 1.874
  //1.842 1.836 1.827 1.826 1.822 1.81 1.797 1.783 1.781 1.778 1.771 1.767 1.759 1.74 1.726 1.719 1.705 1.7 1.693 1.683 1.678
  //1.666 1.663 1.648 1.64 1.628 1.595 1.562 1.517 1.408 1.302 1.281 1.265 1.263 1.249 1.243 1.243 1.24 1.241 1.236 1.235 1.227
  //1.217 1.203 1.194 1.204 1.201 1.188 1.181 1.176 1.171 1.166 1.156 1.156 1.155 1.154 1.149 1.131 1.135 1.139 1.131 1.116 1.113
  //1.112 1.11 1.104 1.111 1.096 1.081 1.081 1.081 1.083 1.076 1.075 1.074 1.074 1.064 1.06 1.055 1.049 1.046 1.045 1.046 1.046
  //1.042 1.04 1.039 1.034 1.031 1.031 1.028 1.021 1.018 1.019 1.016 1.013 1.011 1.015 1.015 1.004 1.002 1.002 1.003 1.001 0.992
  //0.991 0.988 0.988 0.988 0.983 0.983 0.989 0.984 0.981 0.975 0.972 0.97 0.97 0.968 0.968 0.963 0.961 0.96 0.961 0.96 0.958
  //0.955 0.951 0.943 0.94 0.94 0.938 0.937 0.93 0.937 0.938 0.936 0.935 0.933 0.933 0.931 0.931 0.926 0.927 0.922 0.922 0.919
  //0.924 0.922 0.937 0.936 0.952 0.972 0.982 1.001 1.006 1.012 1.03 1.051 1.07 1.095 1.116 1.118 1.118 1.127 1.127 1.125 1.132
  //1.128 1.126 1.126 1.121 1.12 1.121 1.121 1.12 1.12 1.119 1.12 1.12 1.114 1.111 1.117 1.12 1.122 1.118 1.105 1.109 1.115 1.115
  //1.11 1.108 1.108 1.108 1.121 1.115 1.102 1.102 1.105

  std::string line;
  std::ifstream infile("../../../../src/bag_to_image/templates.txt");

  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    float value;
    vector<float> single_template;
    while (iss >> value) {
      single_template.push_back(value);
    }
    std::cout << "push back" << single_template.size() << std::endl;
    all_templates_.push_back(single_template);
  }
}

void initDepthsAndOrientations() {
  all_depths_.push_back(1.0);
  all_depths_.push_back(5.0);
  all_depths_.push_back(9.0);
  all_orientations_.push_back(0.0);
  all_orientations_.push_back(90.0);
  all_orientations_.push_back(-90.0);
}


bool epsilonClose(const float eps, const float n1, const float n2) {
  if (fabs(n1 - n2) <= eps) {
    return true;
  }
  return false;
}

float medianFilter(const vector<float> to_be_filtered) {
  if (to_be_filtered.size() % 2) {
    return to_be_filtered[to_be_filtered.size() / 2];
  }
  else {
    return (to_be_filtered[(to_be_filtered.size() / 2)] +
            to_be_filtered[(to_be_filtered.size() / 2) - 1]) / 2.0;
  }
}

float solveQuadratic(float a, float b, float c) {
  float det = b * b - 4 * a * c;
  if (det < 0.0) {
    return 0.0;
  }
  float t1 = ((-1.0) * b + sqrt(det)) / (2.0 * a);
  float t2 = ((-1.0) * b - sqrt(det)) / (2.0 * a);
  t1 = std::max(float(0.0), t1);
  t2 = std::max(float(0.0), t2);
  return std::min(t1, t2);
}

float magDot2D(float v1, float v2, float w1, float w2) {
  return fabs(v1*w1 + v2*w2);
}

void getScansFromBag(const float laser_max_range, vector<vector<float>>* all_scans) {
  rosbag::Bag bag;
  printf("Opening bag file %s...", bag_name.c_str()); fflush(stdout);
  bag.open(bag_name, rosbag::bagmode::Read);
  printf(" Done.\n"); fflush(stdout);
  float eps = 0.02;

  vector<string> topics;
  topics.push_back("/Cobot/Laser");
  topics.push_back("Cobot/Laser");
  topics.push_back("/laser");
  topics.push_back("laser");
  
  printf("Reading bag file..."); fflush(stdout);
  rosbag::View view(bag, rosbag::TopicQuery(topics));
  for (rosbag::View::iterator it = view.begin(); it != view.end(); ++it) {
    const rosbag::MessageInstance &message = *it;
    sensor_msgs::LaserScanPtr laser_message = message.instantiate<sensor_msgs::LaserScan>();
    //const bool is_cobot_laser = message.getTopic() == kCobotLaserTopic;
    //if (laser_message != NULL && is_cobot_laser) {
    if (laser_message != NULL) {
      vector<float> single_scan;
      for (unsigned int i = clip_l; i < laser_message->ranges.size() - clip_u; ++i) {
        float range = laser_message->ranges[i];
        range = std::max(range, min_range);
        range = std::min(range, laser_max_range);
        if (epsilonClose(eps, range, laser_max_range)) {
          range = 0.0;
        }
        single_scan.push_back(range);
      }
      all_scans->push_back(single_scan);
    }
  }
  //printf(" Done.\n"); fflush(stdout);
  //printf("%d scans loaded.\n", static_cast<int>(all_scans->size()));
}

vector<ScanFeatureMetaData> getFeaturesFromText(string filename) {
  std::string line;
  std::ifstream infile(filename);

  vector<ScanFeatureMetaData> all_features;

  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    ScanFeatureMetaData scan_feature;

    //bag name, scan number, type, start angle, end angle, range[i], ..., range[j]

    
    string bag_name;
    iss >> bag_name;
    int scan_number;
    iss >> scan_number;
    int feature_type;
    iss >> feature_type;
    float start_angle;
    iss >> start_angle;
    float end_angle;
    iss >> end_angle;

    scan_feature.start_angle = start_angle;
    scan_feature.end_angle = end_angle;
    scan_feature.type = feature_type;

    float value;
    vector<float> ranges;
    while (iss >> value) {
      ranges.push_back(value);
    }

    scan_feature.ranges = ranges;

    std::cout << bag_name << " " << scan_number << " " << feature_type << " " 
              << start_angle << " " << end_angle << std::endl;
    // have scan feature, now generate many

    all_features.push_back(scan_feature);
  }
  return all_features;
}


/*
void convertScansToImages(const vector<vector<float>>& all_scans) {
  std::stringstream ss;
  ss.str(bag_name);
  string item;
  vector<string> elems;
  while (getline(ss, item, '/')) {
    elems.push_back(item);
  }
  std::stringstream ss2;
  ss2.str(elems.back());
  elems.clear();
  string item2;
  while (getline(ss2, item2, '.')) {
    elems.push_back(item2);
  }
  string bag_date = elems[0];
  std::cout << "bag_date: " << bag_date << std::endl;

  size_t downsampling_rate = 4;

  size_t width = 256;
  size_t height = 256;
  cimg_library::CImg<uint8_t> scan_image(width, height);
  size_t scan_number = 1;

  printf("Downsampling scans..."); fflush(stdout);
  for (size_t i = 0; i < all_scans.size(); ++i) {
    vector<float> single_scan = all_scans[i];
    vector<float> downsampled_scan;
    for (size_t j = 0; j < single_scan.size(); j += downsampling_rate) {
      vector<float> scan_seg;
      for (size_t k = j; k < j + downsampling_rate; ++k) {
        scan_seg.push_back(single_scan[k]);
      }
      std::sort(scan_seg.begin(), scan_seg.end());
      float median = (scan_seg[1] + scan_seg[2]) / 2.0;
      downsampled_scan.push_back(median);
    }

    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < downsampled_scan.size(); ++x) {
        scan_image((x + y) % height, y) = size_t(255 - (downsampled_scan[x]/max_range)*255);
      }
    }
    string scan_image_file = bag_date + "_" + std::to_string(scan_number) + ".png";
    scan_image.save_png(scan_image_file.c_str());
    scan_number++;

  }
  printf(" Done.\n"); fflush(stdout);
}
*/

void generateFeatureOnlyScanFromFeatures(const float FOV, const vector<ScanFeatureMetaData> all_features, 
                                                                   vector<vector<float>>* all_scans) {
  //for // each feature // (only one feature at a time for now)
  int num_rays = (all_features[0].ranges.size() * FOV) / 
                 ((all_features[0].end_angle - all_features[0].start_angle) * (180.0 / M_PI));
  float angular_res = FOV / float(num_rays);
  float scan_start_angle = -FOV / 2.0;
  int feature_start_index = (scan_start_angle - all_features[0].start_angle) * angular_res;
  int feature_end_index = feature_start_index + all_features[0].ranges.size();
  std::cout << "num rays: " << num_rays << std::endl;
  std::cout << "start: " << feature_start_index << std::endl;
  std::cout << "end: " << feature_end_index << std::endl;
  // Probably should be a feature that can be enabled by the type of query
  // (eg. anywhere in the scan vs. on the left side)
  //TODO: generate random location within the scan (optional, not doing right now)
  vector<float> single_scan;
  for (int i = 0; i < num_rays; ++i) { // each ray
    if (i >= feature_start_index && i <= feature_end_index) {
      single_scan.push_back(all_features[0].ranges[i - feature_start_index]);
    }
    else {
      // Set all depths to zero if not part of feature
      single_scan.push_back(0.0);
    }
  }
  all_scans->push_back(single_scan);
}

void generateMonteCarloScansFromFeatures(const float FOV, const vector<ScanFeatureMetaData> all_features, 
                                                                   vector<vector<float>>* all_scans) {
  //for // each feature // (only one feature at a time for now)
  int num_rays = (all_features[0].ranges.size() * FOV) / 
                 ((all_features[0].end_angle - all_features[0].start_angle) * (180.0 / M_PI));
  float angular_res = FOV / float(num_rays);
  float scan_start_angle = -FOV / 2.0;
  int feature_start_index = (scan_start_angle - all_features[0].start_angle) * angular_res;
  int feature_end_index = feature_start_index + all_features[0].ranges.size();
  int K = 1000;
  std::cout << "num rays: " << num_rays << std::endl;
  std::cout << "start: " << feature_start_index << std::endl;
  std::cout << "end: " << feature_end_index << std::endl;
  for (int k = 0; k < K; ++k) { // K times
   std::cout << k << std::endl;
   // Probably should be a feature that can be enabled by the type of query
    // (eg. anywhere in the scan vs. on the left side)
    //TODO: generate random location within the scan (optional, not doing right now)
    vector<float> single_scan;
    for (int i = 0; i < num_rays; ++i) { // each ray
      if (i >= feature_start_index && i <= feature_end_index) {
        single_scan.push_back(all_features[0].ranges[i - feature_start_index]);
      }
      else {
        // Sample depth from uniform distribution over laser range
        const float depth = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/max_range));
        single_scan.push_back(depth);
      }
    }
    all_scans->push_back(single_scan);
  }
}

void applyTruncate(const float new_max_range, vector<vector<float>>* all_scans) {
  for (size_t i = 0; i < all_scans->size(); ++i) {
    for (size_t j = 0; j < all_scans[0][i].size(); ++j) {
      if ((*all_scans)[i][j] >= new_max_range) { // truncate ==> set max range+ to 0
        (*all_scans)[i][j] = 0.0;
      }
    }
  }
}

void applySeasoning(const float amount, vector<vector<float>>* all_scans) {
  for (size_t i = 0; i < all_scans->size(); ++i) {
    for (size_t j = 0; j < all_scans[0][i].size(); ++j) {
      float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      if (r <= amount) {
        const float seasoned_obs = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/max_range));
        all_scans[0][i][j] = seasoned_obs;
      }
    }
  }
}

void applyBlur(const float haze, vector<vector<float>>* all_scans) {
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0.0, haze);
  for (size_t i = 0; i < all_scans->size(); ++i) {
    for (size_t j = 0; j < all_scans[0][i].size(); ++j) {
      float r = distribution(generator);
      float blurred_obs = all_scans[0][i][j] + r;
      blurred_obs = fmin(blurred_obs, max_range);
      blurred_obs = fmax(blurred_obs, min_range);
      all_scans[0][i][j] = blurred_obs;
    }
  }
}

void applyHumans(const int FOV, vector<vector<float>>* all_scans) {
  const float min_human_dist = 1.0;
  const float FOV_rad = (float(FOV)/180.0) * 2 * M_PI;
  const int num_obs = (*all_scans)[0].size();
  std::default_random_engine generator;
  for (size_t i = 0; i < all_scans->size(); ++i) {
    vector<float>* single_scan = &(*all_scans)[i];
    //const float d = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 10.0) + 0.05; // 0.05 - 0.15 meters
    const float d = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 10.0) + 0.15; // 0.15 - 0.30 meters
    const float leg_radius = d / 2.0;
    //const float s = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 4.0) + 0.05; // 0.05 - 0.30 meters
    const float s = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 5.0) + 0.10; // 0.10 - 0.30 meters
    const float conservative_theta_T = 2*atan2(d, min_human_dist) + atan2(s, min_human_dist);
    const float phi = (-0.5*FOV_rad + conservative_theta_T)
                    + (static_cast <float> (rand()) / static_cast <float>(RAND_MAX)) *
                      (FOV_rad - 2*conservative_theta_T);
    const int r_ub_idx = int(((phi + 0.5*FOV_rad) / FOV_rad) * float(num_obs));
    const float r_ub = std::max(min_human_dist, (*all_scans)[i][r_ub_idx]);
    float r = 1.0;
    if (r_ub > r) {
      r = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) *
          (r_ub - min_human_dist))
        + min_human_dist;
    }
    const float theta_d = atan2(d, r);
    const float theta_s = atan2(s, r);
    const float theta_T = 2 * atan2(d, r) + atan2(s, r);
    const int to_replace = (num_obs / FOV_rad) * theta_d;
    const int start_idx1 = num_obs/2.0 + (phi - (theta_T / 2.0)) * (num_obs / FOV_rad);
    const int start_idx2 = num_obs/2.0 + (phi + (theta_s / 2.0)) * (num_obs / FOV_rad);
    const int stop_idx1 = start_idx1 + to_replace;
    const int stop_idx2 = start_idx2 + to_replace;
    const float L1x = r * cos(phi - (theta_d/2.0) - (theta_s/2.0));
    const float L1y = r * sin(phi - (theta_d/2.0) - (theta_s/2.0));
    const float L2x = r * cos(phi + (theta_d/2.0) + (theta_s/2.0));
    const float L2y = r * sin(phi + (theta_d/2.0) + (theta_s/2.0));
    for (int j = start_idx1; j <= stop_idx1; ++j) { // leg 1
      float theta_r = -0.5*FOV_rad + FOV_rad * (float(j) / float(num_obs));
      float a = 1.0; // a = sin^2 + cos^2 = 1
      float b = -2.0 * ((cos(theta_r) * L1x) + (sin(theta_r) * L1y));
      float c = L1x*L1x + L1y*L1y - leg_radius*leg_radius;
      float new_range_gt = solveQuadratic(a, b, c);
      if (new_range_gt < min_human_dist) {
        (*single_scan)[j] = new_range_gt;
      }
      else {
        float v1 = cos(theta_r);
        float v2 = sin(theta_r);
        float w1 = r*cos(theta_r) - L1x;
        float w2 = r*sin(theta_r) - L1y;
        float dot = magDot2D(v1, v2, w1, w2);
        //float mean = 0.05 / pow((dot + 1), 10);
        float mean = 0.0;
        float variance = 0.01 / pow((dot + 1), 12);
        std::normal_distribution<float> distribution(mean, variance);
        float noise = distribution(generator);
        float noised_range = new_range_gt + noise;
        (*single_scan)[j] = std::min(max_range, noised_range);
      }
    }
    for (int j = start_idx2; j <= stop_idx2; ++j) { // leg 2
      float theta_r = -0.5*FOV_rad + FOV_rad * (float(j) / float(num_obs));
      float a = 1.0; // a = sin^2 + cos^2 = 1
      float b = -2.0 * ((cos(theta_r) * L2x) + (sin(theta_r) * L2y));
      float c = L2x*L2x + L2y*L2y - leg_radius*leg_radius;
      float new_range_gt = solveQuadratic(a, b, c);
      if (new_range_gt < min_human_dist) {
        (*single_scan)[j] = new_range_gt;
      }
      else {
        float v1 = cos(theta_r);
        float v2 = sin(theta_r);
        float w1 = r*cos(theta_r) - L2x;
        float w2 = r*sin(theta_r) - L2y;
        float dot = magDot2D(v1, v2, w1, w2);
        //float mean = 0.05 / pow((dot + 1), 10);
        float mean = 0.0;
        float variance = 0.01 / pow((dot + 1), 12);
        std::normal_distribution<float> distribution(mean, variance);
        float noise = distribution(generator);
        float noised_range = new_range_gt + noise;
        (*single_scan)[j] = std::min(max_range, noised_range);
      }
    }
  }
}

void writeScans(vector<vector<float>>* all_scans) {
  std::ofstream outfile;
  //outfile.open("FeatureScans.txt", std::ios_base::app);
  //outfile.open("CorruptionQuality.txt", std::ios_base::app);
  //outfile.open("RawScans.txt", std::ios_base::app);
  outfile.open("RawSynthScans.txt", std::ios_base::app);
  for (size_t i = 0; i < all_scans->size(); ++i) {
    //std::cout << "scan size: " << all_scans[0][i].size() << std::endl;
    for (size_t j = 0; j < all_scans[0][i].size(); ++j) {
      outfile << all_scans[0][i][j] << " ";
    }
    outfile << "\n";
  }
}

void applyCorruption(const float FOV,
                     const bool truncate,
                     const bool seasoning,
                     const bool blur,
                     const bool humans,
                     vector<vector<float>>* all_scans) {
  //writeScans(all_scans);
  if (humans) {
    applyHumans(FOV, all_scans);
  }
  if (blur) {
    float haze = 0.1; // meters
    applyBlur(haze, all_scans);
  }
  if (seasoning) {
    float amount = 0.05;
    applySeasoning(amount, all_scans);
  }
  if (truncate) {
    float new_max_range = 4.0;
    applyTruncate(new_max_range, all_scans);
  }
  //writeScans(all_scans);
}





void insertSyntheticTemplate(const float FOV,
                             const int depth_idx,
                             const int orientation_idx,
                             const int template_type_idx,
                             const vector<vector<float>> orig_scans,
                             vector<vector<float>>* all_scans) {

  //TODO: possibly pseudo random depths / orientations

  const float depth = all_depths_[depth_idx];
  const float orientation = all_orientations_[orientation_idx];
  vector<float> current_template = all_templates_[template_type_idx];

  // Compute indices
  const int insert_size = current_template.size();
  const float scan_start_angle = -FOV / 2.0;
  const size_t numrays = orig_scans[0].size();
  const float angular_res = FOV / float(numrays);

  int insert_center_index = (orientation - scan_start_angle) / angular_res;
  int insert_start_index = insert_center_index - (insert_size / 2);
  int insert_end_index = insert_start_index + insert_size;

  if (insert_start_index < 0) {
    insert_start_index = 0;
    insert_end_index = insert_size - 1;
  }
  else if (insert_end_index > (int(numrays) - 1)) {
    insert_start_index = numrays - insert_size - 1;
    insert_end_index = numrays - 1;
  }

  // Compute depth
  float min_insert_depth = max_range;
  float max_insert_depth = min_range;
  for (size_t i = 0; i < current_template.size(); ++i) {
    min_insert_depth = std::fmin(current_template[i], min_insert_depth);
    if (current_template[i] < 0.98 * max_range) {
      max_insert_depth = std::fmax(current_template[i], max_insert_depth);
    }
  }
  const float insert_mid = (max_insert_depth + min_insert_depth) / 2.0;
  //const float insert_range = max_insert_depth - min_insert_depth;
  float offset = depth - insert_mid;
  if (max_insert_depth + offset > max_range) {
    offset = max_range - max_insert_depth;
  }
  else if (min_insert_depth + offset < min_range) {
    offset = min_insert_depth - min_range;
  }

  //TODO: haven't figured out depth scaling yet (in absense of geometry for ray casting)
  //for (size_t i = 0; i < current_template.size(); ++i) {
  //  current_template[i] = current_template[i] + offset;
  //}

  // Insert template into scan
  for (size_t i = 0; i < orig_scans.size(); ++i) {
    vector<float> synth_scan = orig_scans[i];
    for (int j = 0; j < int(numrays); ++j) { // each ray
      if (j >= insert_start_index && j <= insert_end_index) {
        synth_scan[j] = current_template[j - insert_start_index];
      }
    }
    all_scans->push_back(synth_scan);
  }
}





void convertScansToImagesPadAndFixedBounds(const vector<vector<float>>& all_scans, const int FOV) {
  std::stringstream ss;
  ss.str(bag_name);
  string item;
  vector<string> elems;
  while (getline(ss, item, '/')) {
    elems.push_back(item);
  }
  std::stringstream ss2;
  ss2.str(elems.back());
  elems.clear();
  string item2;
  while (getline(ss2, item2, '.')) {
    elems.push_back(item2);
  }
  string bag_date = elems[0];
  if (bag_date == "txt") {
    //bag_date = "FeatureOnly";
    bag_date = "MCFeature";
  }
  //std::cout << "bag_date: " << bag_date << std::endl;
  std::cout << bag_date << ", " << all_scans.size() << std::endl;

  int num_obs = all_scans[0].size(); //how many points each scan has
  int padding = int((float(num_obs) / float(FOV)) * (360 - FOV));
  if (padding % 2) {
    padding--;
  }
  int length = num_obs + padding;
  int extra = length % 256;
  int deletion_interval = length / (extra + 1);

  //printf("Padding scans..."); fflush(stdout);
  vector<vector<float>> all_padded_scans;
  for (size_t i = 0; i < all_scans.size(); ++i) {
    vector<float> single_scan = all_scans[i];
    vector<float> padded_scan;
    for (int j = 0; j < (padding / 2); ++j) {
      padded_scan.push_back(0.0);
    }
    for (size_t j = 0; j < single_scan.size(); ++j) {
      padded_scan.push_back(single_scan[j]);
    }
    for (int j = 0; j < (padding / 2); ++j) {
      padded_scan.push_back(0.0);
    }
    // delete entries to make length k * 256 s.t. k is an int
    for (int j = extra; j > 0; j--) {
      padded_scan.erase(padded_scan.begin() + (j * deletion_interval));
    }
    all_padded_scans.push_back(padded_scan);
  }

  //printf("mod 256... should be 0: %d", int(all_padded_scans[0].size() % 256));
  //printf("size of scan: %d", int(all_padded_scans[0].size()));
  size_t downsampling_rate = all_padded_scans[0].size() / 256;
  //printf("downsample rate: %d", int(downsampling_rate));

  size_t width = 256;
  size_t height = 256;
  cimg_library::CImg<uint8_t> scan_image_rot(width, height);
  cimg_library::CImg<uint8_t> scan_image_norm(width, height);
  size_t scan_number = 1;

  std::ofstream outfile;
  //outfile.open("downsampledscans.txt", std::ios_base::app);
  //printf("Downsampling scans..."); fflush(stdout);
  for (size_t i = 0; i < all_padded_scans.size(); ++i) {
    vector<float> padded_scan = all_padded_scans[i];
    vector<float> downsampled_scan;
    for (size_t j = 0; j < padded_scan.size(); j += downsampling_rate) {
      vector<float> scan_seg;
      for (size_t k = j; k < j + downsampling_rate; ++k) {
        scan_seg.push_back(padded_scan[k]);
      }
      std::sort(scan_seg.begin(), scan_seg.end());
      float median = medianFilter(scan_seg);
      downsampled_scan.push_back(median);
    }

    for (size_t y = 0; y < height; ++y) {
      //outfile << size_t(255 - (downsampled_scan[y]/max_range)*255);
      for (size_t x = 0; x < downsampled_scan.size(); ++x) {
        scan_image_rot((x + y) % width, y) = size_t(255 - (downsampled_scan[x]/max_range)*255);
        scan_image_norm(x, y) = size_t(255 - (downsampled_scan[x]/max_range)*255);
      }
    }
    //outfile << "\n";
    string rot_scan_image_file = bag_date + "_" + std::to_string(scan_number) + "_rot.png";
    string norm_scan_image_file = bag_date + "_" + std::to_string(scan_number) + "_norm.png";
    scan_image_rot.save_png(rot_scan_image_file.c_str());
    scan_image_norm.save_png(norm_scan_image_file.c_str());
    scan_number++;
  }
  //printf(" Done.\n"); fflush(stdout);
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    printf("Usage: ./exec <bag file> <laser FoV (deg)> <max laser range (m)> (opt)<feature file>");
    fflush(stdout);
    exit(1);
  }
  bag_name = argv[1];
  int FOV = atoi(argv[2]); // Field of View of the laser
  float laser_max_range = atof(argv[3]);
  vector<vector<float>> all_scans;
  if (bag_name == "txt") {
    string feature_filename = argv[4];
    vector<ScanFeatureMetaData> all_features = getFeaturesFromText(feature_filename);
    //generateFeatureOnlyScanFromFeatures(FOV, all_features, &all_scans);
    generateMonteCarloScansFromFeatures(FOV, all_features, &all_scans);
  }
  else if (bag_name == "synth") {
    bag_name = argv[4];
    getScansFromBag(laser_max_range, &all_scans);
    initTemplates();
    initDepthsAndOrientations();
    vector<vector<float>> orig_scans = all_scans;
    all_scans.clear();

    //TODO: need to factor in scaling of template sizes for changing depths
    //for (size_t depth = 0; depth < all_depths_.size(); ++depth) {
      for (size_t orientation = 0; orientation < all_orientations_.size(); ++orientation) {
        for (size_t template_type = 0; template_type < all_templates_.size(); ++template_type) {
          insertSyntheticTemplate(FOV, 0, orientation, template_type, orig_scans, &all_scans);
//          insertSyntheticTemplate(FOV, 0, 0, 0, orig_scans, &all_scans);
        }
      }
    //}
    writeScans(&all_scans);
  }
  else {
    bool truncate = false;
    bool seasoning = false; // Add salt-and-pepper-type noise
    bool blur = false;
    bool humans = false;
    //printf(argv[1]); printf("\n"); fflush(stdout);
    //printf("%d \n", FOV); fflush(stdout);
    //printf("%f \n", laser_max_range); fflush(stdout);

    getScansFromBag(laser_max_range, &all_scans);
    //NOTE: for baseline testing
    ////writeScans(&all_scans);

    srand (static_cast <unsigned> (time(0)));
    applyCorruption(FOV, truncate, seasoning, blur, humans, &all_scans);


    ////convertScansToImages(all_scans);
    ////writeScans(&all_scans);
  }
  std::cout << "size: " << all_scans.size() << std::endl;
  //writeScans(&all_scans);
  convertScansToImagesPadAndFixedBounds(all_scans, FOV);
  return 0;
}
