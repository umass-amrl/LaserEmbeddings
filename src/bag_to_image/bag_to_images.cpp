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

using std::string;
using std::vector;

unsigned int clip_l = 0;
unsigned int clip_u = 0;
//unsigned int clip_l = 28;
//unsigned int clip_u = 29;
float min_range = 0.0;
float max_range = 10.0;
string bag_name;

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

void getScansFromBag(const float laser_max_range,
                     vector<vector<float>>* all_scans) {
  rosbag::Bag bag;
  //printf("Opening bag file %s...", bag_name.c_str()); fflush(stdout);
  bag.open(bag_name, rosbag::bagmode::Read);
  //printf(" Done.\n"); fflush(stdout);
  float eps = 0.02;

  vector<string> topics;
  topics.push_back("/Cobot/Laser");
  topics.push_back("Cobot/Laser");
  topics.push_back("/laser");
  topics.push_back("laser");
  
  //printf("Reading bag file..."); fflush(stdout);
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
      if (r <= amount) { // truncate ==> set max range+ to 0
        const float seasoned_obs = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/max_range));
        all_scans[0][i][j] = seasoned_obs;
      }
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
  outfile.open("CorruptionQuality.txt", std::ios_base::app);
  for (size_t i = 0; i < all_scans->size(); ++i) {
    for (size_t j = 0; j < all_scans[0][i].size(); ++j) {
      outfile << all_scans[0][i][j] << " ";
    }
    outfile << "\n";
  }
}

void applyCorruption(const float FOV,
                     const bool truncate,
                     const bool seasoning,
                     const bool humans,
                     vector<vector<float>>* all_scans) {
  //writeScans(all_scans);
  if (humans) {
    applyHumans(FOV, all_scans);
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


void convertScansToImagesPadAndFixedBounds(const vector<vector<float>>& all_scans,
                                           const int FOV) {
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
      for (size_t x = 0; x < downsampled_scan.size(); ++x) {
        scan_image_rot((x + y) % width, y) = size_t(255 - (downsampled_scan[x]/max_range)*255);
        scan_image_norm(x, y) = size_t(255 - (downsampled_scan[x]/max_range)*255);
      }
    }
    string rot_scan_image_file = bag_date + "_" + std::to_string(scan_number) + "_rot.png";
    string norm_scan_image_file = bag_date + "_" + std::to_string(scan_number) + "_norm.png";
    scan_image_rot.save_png(rot_scan_image_file.c_str());
    scan_image_norm.save_png(norm_scan_image_file.c_str());
    scan_number++;
  }
  //printf(" Done.\n"); fflush(stdout);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: ./exec <bag file> <laser FoV (deg)> <max laser range (m)>");
    fflush(stdout);
    exit(1);
  }
  bag_name = argv[1];
  int FOV = atoi(argv[2]); // Field of View of the laser
  float laser_max_range = atof(argv[3]);
  bool truncate = false;
  bool seasoning = false; // Add salt-and-pepper-type noise
  bool humans = false;
  //printf(argv[1]); printf("\n"); fflush(stdout);
  //printf("%d \n", FOV); fflush(stdout);
  //printf("%f \n", laser_max_range); fflush(stdout);
  vector<vector<float>> all_scans;
  getScansFromBag(laser_max_range, &all_scans);
  srand (static_cast <unsigned> (time(0)));
  applyCorruption(FOV, truncate, seasoning, humans, &all_scans);
  //convertScansToImages(all_scans);
  convertScansToImagesPadAndFixedBounds(all_scans, FOV);
  return 0;
}
