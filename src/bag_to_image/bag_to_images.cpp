#include <algorithm>
#include <iostream>
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

unsigned int clip_l = 28;
unsigned int clip_u = 29;
float min_range = 0.0;
float max_range = 10.0;
size_t downsampling_rate = 4;

void getScansFromBag(const string& bag_name, vector<vector<float>>* all_scans) {
  rosbag::Bag bag;
  printf("Opening bag file %s...", bag_name.c_str()); fflush(stdout);
  bag.open(bag_name, rosbag::bagmode::Read);
  printf(" Done.\n"); fflush(stdout);

  vector<string> topics;
  string kCobotLaserTopic = "/Cobot/Laser";
  topics.push_back(kCobotLaserTopic);
  
  printf("Reading bag file..."); fflush(stdout);
  rosbag::View view(bag, rosbag::TopicQuery(topics));
  for (rosbag::View::iterator it = view.begin(); it != view.end(); ++it) {
    const rosbag::MessageInstance &message = *it;
    sensor_msgs::LaserScanPtr laser_message = message.instantiate<sensor_msgs::LaserScan>();
    const bool is_cobot_laser = message.getTopic() == kCobotLaserTopic;
    if (laser_message != NULL && is_cobot_laser) {
      vector<float> single_scan;
      for (unsigned int i = clip_l; i < laser_message->ranges.size() - clip_u; ++i) {
        float range = laser_message->ranges[i];
        range = std::max(range, min_range);
        range = std::min(range, max_range);
        single_scan.push_back(range);
      }
      all_scans->push_back(single_scan);
    }
  }
  printf(" Done.\n"); fflush(stdout);
  printf("%d scans loaded.\n", static_cast<int>(all_scans->size()));
}

void convertScansToImages(string bag_name, const vector<vector<float>>& all_scans) {
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

    //TODO: maybe choose lowest range for top left (0, 0).
    //      maybe only put in 1-D array?
    //      maybe use a time series?
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

int main(int argc, char* argv[]) {
  string bag_name = argv[1];
  vector<vector<float>> all_scans;
  getScansFromBag(bag_name, &all_scans);
  convertScansToImages(bag_name, all_scans);
  return 0;
}
