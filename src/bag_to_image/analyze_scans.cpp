#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include "ros/ros.h"
#include "ros/package.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "std_msgs/String.h"
#include "sensor_msgs/LaserScan.h"
#include "gui_msgs/GuiKeyboardEvent.h"
#include "gui_msgs/GuiMouseClickEvent.h"
#include "gui_msgs/LidarDisplayMsg.h"
#include "gui_msgs/ScanFeatureMsg.h"
#include "../perception_tools/perception_2d.h"
#include "shared_structs.h"

using std::string;
using std::vector;

//TODO: Set up to be run via scripts

ScanFeatureMetaData scan_feature_;
bool feature_selected_ = false;

ros::Subscriber mouse_subscriber_;
ros::Subscriber keyboard_subscriber_;
ros::Publisher display_publisher_;
ros::Publisher selection_publisher_;

gui_msgs::LidarDisplayMsg display_message_;
gui_msgs::ScanFeatureMsg selection_message_;

string bag_name_;
vector<vector<float>> all_scans_;
vector<vector<float>> all_scans_sidekick_;

int current_view_ = 1;
vector<int> id_in_progress_;
bool viewing_recreation_ = true;
bool normalized_ = false;
//bool normalized_ = true;
bool side_by_side_viewing_ = false;

float min_range = 0.0;    // meters
float max_range = 10.0;   // meters
float angular_res = 0.25; // degrees

void getScansFromBag() {
  rosbag::Bag bag;
  printf("Opening bag file %s...", bag_name_.c_str()); fflush(stdout);
  bag.open(bag_name_, rosbag::bagmode::Read);
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
      for (unsigned int i = 0; i < laser_message->ranges.size(); ++i) {
        float range = laser_message->ranges[i];
        range = std::max(range, min_range);
        range = std::min(range, max_range);
        single_scan.push_back(range);
      }
      all_scans_.push_back(single_scan);
    }
  }
  printf(" Done.\n"); fflush(stdout);
  printf("%d scans loaded.\n", static_cast<int>(all_scans_.size()));
}

/*
// NOTE: for debugging / verifying stuff
void pubScanFeature() {
  display_message_.points_x.clear();
  display_message_.points_y.clear();
  
  vector<float> scan = scan_feature_.ranges;
  float start_angle = scan_feature_.start_angle * 180.0 / M_PI;
  for (size_t i = 0; i < scan.size(); ++i) {
    float angle = (start_angle + angular_res * i) * (M_PI / 180.0);
    Eigen::Rotation2Df R(angle);
    Eigen::Vector2f x(scan[i], 0.0);
    Eigen::Vector2f p = R * x;
    display_message_.points_x.push_back(p(0));
    display_message_.points_y.push_back(p(1));
  }
  display_publisher_.publish(display_message_);
}
*/

void calculatePoints(const float start_angle, const float angular_res,
                     const Eigen::Vector2f origin, const vector<float> scan) {
  for (size_t i = 0; i < scan.size(); ++i) {
    float angle = (start_angle + angular_res * i) * (M_PI / 180.0);
    Eigen::Rotation2Df R(angle);
    Eigen::Vector2f x(scan[i], 0.0);
    Eigen::Vector2f p = R * x + origin;
    display_message_.points_x.push_back(p(0));
    display_message_.points_y.push_back(p(1));
  }
}

void pubScan() {
  //TODO: figure out why there is lag / weird behavior in the scan display
  //TODO: harden logic / parameter setting for whether or not recreated scans are normalized
  std::cout << "scan number: " << current_view_ << std::endl;
  display_message_.points_x.clear();
  display_message_.points_y.clear();
  if (!side_by_side_viewing_) {
    vector<float> scan = all_scans_[current_view_ - 1];
    Eigen::Vector2f origin(0.0, 0.0);
    float start_angle = -135.0;         // default value for UST-10LX
    float angular_res = 270.0 / 1024.0; // default value for UST-10LX?
    if (normalized_) {
      angular_res = 360.0 / float(scan.size());
    }
    else {
      angular_res = 270.0 / float(scan.size());
    }
    calculatePoints(start_angle, angular_res, origin, scan);
  }
  else {
    vector<float> scan1 = all_scans_[current_view_ - 1];
    vector<float> scan2 = all_scans_sidekick_[current_view_ - 1];
    Eigen::Vector2f origin1(-12.0, 0.0);
    Eigen::Vector2f origin2(12.0, 0.0);
    float start_angle1 = -135.0;
    float start_angle2 = -180.0;
    float angular_res1 = 270.0 / float(scan1.size());
    //float angular_res2 = 360.0 / float(scan2.size());
    float angular_res2 = 360.0 / float(scan2.size());
    calculatePoints(start_angle1, angular_res1, origin1, scan1);
    calculatePoints(start_angle2, angular_res2, origin2, scan2);
  }

  display_publisher_.publish(display_message_);
}

void publishQuery() {
  selection_message_.timestamp = ros::Time::now().toSec();
  selection_message_.ranges = scan_feature_.ranges;
  selection_message_.start_angle = scan_feature_.start_angle;
  selection_message_.end_angle = scan_feature_.end_angle;
  selection_message_.type = scan_feature_.type;
  selection_publisher_.publish(selection_message_);
}

//void saveScan(const string filename) {
//  vector<float> scan = all_scans_[current_view_ - 1];
//  todo: save scan?
//}

void saveScanFeature(string filename) {
  std::cout << "saving feature" << std::endl;
  std::ofstream outfile;
  outfile.open(filename, std::ios_base::app);
  
  // bag_name_, current_view_-1, type, start_angle, end_angle, range[0], ..., range[n]
  outfile << bag_name_ << " " << current_view_ - 1 << " " 
          << scan_feature_.type << " " << scan_feature_.start_angle << " " << scan_feature_.end_angle;
  for (size_t i = 0; i < scan_feature_.ranges.size(); ++i) {
    outfile << " " << scan_feature_.ranges[i];
  }
  outfile << "\n";
  //publishQuery(filename);
  publishQuery();
}

void incrementView() {
  if (current_view_ == int(all_scans_.size())) {
    current_view_ = 1;
  }
  else {
    current_view_++;
  }
  pubScan();
}

void decrementView() {
  if (current_view_ == 1) {
    current_view_ = all_scans_.size();
  }
  else {
    current_view_--;
  }
  pubScan();
}

void setCurrentView() {
  int scanID = 0;
  for (int i = int(id_in_progress_.size()) - 1; i >= 0; --i) {
    scanID += id_in_progress_[i] * std::pow(10, (id_in_progress_.size() - 1) - i);
  }
  std::cout << scanID << std::endl;
  if (scanID > int(all_scans_.size())) {
    std::cout << "Scan ID out of range." << std::endl;
  }
  else if (scanID == 0) {
    std::cout << "Scan IDs start at 1." << std::endl;
  }
  else {
    current_view_ = scanID;
    pubScan();
  }
}

void MouseClickCallback(const gui_msgs::GuiMouseClickEvent& msg) {
  //TODO: move to global. decide how to set based on laser input
  float min_angle_ = -135.0 * (M_PI / 180.0);
  float max_angle_ = 135.0 * (M_PI / 180.0);
  int min_feature_size_ = 5;

  float th1 = atan2(msg.mouse_down.y, msg.mouse_down.x);
  Eigen::Vector3f v1(msg.mouse_down.x, msg.mouse_down.y, 0.0);
  float th2 = atan2(msg.mouse_up.y, msg.mouse_up.x);
  Eigen::Vector3f v2(msg.mouse_up.x, msg.mouse_up.y, 0.0);
  if (static_cast<uint32_t>(msg.modifiers) == 1 ||
      static_cast<uint32_t>(msg.modifiers) == 2) {
    // TODO: take care of all feature selection corner cases
    if (static_cast<uint32_t>(msg.modifiers) == 1) {
//      std::cout << "tagging small angle" << std::endl;

      int start_index = -1;
      int end_index = -1;
      vector<float> scan = all_scans_[current_view_ - 1];
      float angular_resolution = (fabs(min_angle_ - max_angle_) / float(scan.size()));
//      std::cout << "ang_res: " << angular_resolution << std::endl;
      if ((v1.cross(v2))(2) > 0.0) { // correct, ccw ordering
        if (th1 > min_angle_ && th1 < max_angle_ &&
            th2 > min_angle_ && th2 < max_angle_) {
//          std::cout << "CCW" << std::endl;
          start_index = fabs(min_angle_ - th1) / angular_resolution;
          end_index = fabs(min_angle_ - th2) / angular_resolution;
          scan_feature_.start_angle = th1;
          scan_feature_.end_angle = th2;
        }
        else {
          std::cout << "feature is not fully contained in laser field of view. Ignoring" << std::endl;
        }
      }
      else {
        if (th1 > min_angle_ && th1 < max_angle_ &&
            th2 > min_angle_ && th2 < max_angle_) {
//          std::cout << "CW" << std::endl;
          end_index = fabs(min_angle_ - th1) / angular_resolution;
          start_index = fabs(min_angle_ - th2) / angular_resolution;
          scan_feature_.end_angle = th1;
          scan_feature_.start_angle = th2;
        }
        else {
          std::cout << "feature is not fully contained in laser field of view. Ignoring" << std::endl;
        }
      }

      if (start_index < int(scan.size()) && start_index > 0 &&
          end_index < int(scan.size()) && end_index > 0 &&
          start_index + min_feature_size_ < end_index) {

        scan_feature_.ranges.clear();
        for (int i = start_index; i < end_index; ++i) {
          scan_feature_.ranges.push_back(scan[i]);
        }

        feature_selected_ = true;

        std::cout << "selected feature with " << scan_feature_.ranges.size() << " points" << std::endl;

//        pubScanFeature();
      }
      else {
        std::cout << "Something went wrong when calculating feature start and stop indices." << std::endl;
      }

    }
    else if (static_cast<uint32_t>(msg.modifiers) == 2) {
      std::cout << "tagging large angle" << std::endl;
      std::cout << "not currently implemented" << std::endl;
      //TODO: populate scan feature
    }
  }
}

void KeyboardEventCallback(const gui_msgs::GuiKeyboardEvent& msg) {

  ////////// looking through scans //////////
  if (msg.keycode == 0x1000012) { // left arrow key ==> decrement view
    decrementView();
  }
  else if (msg.keycode == 0x1000014) { // right arrow key ==> increment view
    incrementView();
  }
  else if (msg.keycode >= 0x30 && msg.keycode <= 0x39) { // number is entered
    std::cout << "got number" << std::endl;
    switch (msg.keycode) {
      case 0x30: id_in_progress_.push_back(0); std::cout << "0" << std::endl; break;
      case 0x31: id_in_progress_.push_back(1); std::cout << "1" << std::endl; break;
      case 0x32: id_in_progress_.push_back(2); std::cout << "2" << std::endl; break;
      case 0x33: id_in_progress_.push_back(3); std::cout << "3" << std::endl; break;
      case 0x34: id_in_progress_.push_back(4); std::cout << "4" << std::endl; break;
      case 0x35: id_in_progress_.push_back(5); std::cout << "5" << std::endl; break;
      case 0x36: id_in_progress_.push_back(6); std::cout << "6" << std::endl; break;
      case 0x37: id_in_progress_.push_back(7); std::cout << "7" << std::endl; break;
      case 0x38: id_in_progress_.push_back(8); std::cout << "8" << std::endl; break;
      case 0x39: id_in_progress_.push_back(9); std::cout << "9" << std::endl; break;
    }
  }
  else if (msg.keycode == 0x47) { // key code 71 for 'g' for "Go"
    setCurrentView();
    id_in_progress_.clear();
  }
  ////////// looking through scans //////////

//  else if (msg.keycode == 0x56) { // key code 86, 'v' for save
//    saveScan("ScansOfInterest.txt");
//  }

  ////////// saving and labeling features //////////

  if (feature_selected_) {
    if (msg.keycode == 0x48) { // key code for 'h' for human
      std::cout << "human feature" << std::endl;
      scan_feature_.type = 0; // human
      saveScanFeature("human_features.txt");
      feature_selected_ = false;
    }
    else if (msg.keycode == 0x44) { // key code 68 for 'd' for door
      std::cout << "door feature" << std::endl;
      scan_feature_.type = 1; // door
      saveScanFeature("door_features.txt");
      feature_selected_ = false;
    }
    else if (msg.keycode == 0x43) { // key code 67 for 'c' for corner
      std::cout << "corner feature" << std::endl;
      scan_feature_.type = 2; // corner
      saveScanFeature("corner_features.txt");
      feature_selected_ = false;
    }
    else if (msg.keycode == 0x52) { // key code 82 for 'r' for reset
      std::cout << "reset" << std::endl;
      feature_selected_ = false;
    }
  }

  if (msg.keycode == 0x57) { // key code 87, 'w' for whole
    scan_feature_.ranges = all_scans_[current_view_ - 1];
    scan_feature_.start_angle = -135.0 * (M_PI/180.0);
    scan_feature_.end_angle = 135.0 * (M_PI/180.0);
    feature_selected_ = true;
  }

  ////////// saving and labeling features //////////
}

void getScansFromTxt() {
  std::string line;
  std::ifstream infile("CorruptionQuality.txt");

  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    float value;
    vector<float> single_scan;
    while (iss >> value) {
      single_scan.push_back(value);
    }
    std::cout << "push back" << single_scan.size() << std::endl;
    all_scans_.push_back(single_scan);
  }
  vector<vector<float>> normal;
  vector<vector<float>> corrupted;
  for (size_t i = 0; i < all_scans_.size(); ++i) {
    if (i < (all_scans_.size()/2)) {
      normal.push_back(all_scans_[i]);
    }
    else {
      corrupted.push_back(all_scans_[i]);
    }
  }
  all_scans_.clear();
  for (size_t i = 0; i < normal.size(); ++i) {
    all_scans_.push_back(normal[i]);
    all_scans_.push_back(corrupted[i]);
  }
}

void getDownsampledScansFromTxt() {
  std::string line;
  std::ifstream infile("src/python/recreations.txt");
  //std::ifstream infile("src/python/scansasimages.txt");
  //std::ifstream infile("downsampledscans.txt");

  float maxin = 0.0;
  float minin = 0.0;

  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    float value;
    vector<float> single_scan;
    while (iss >> value) {
      maxin = fmax(maxin, value);
      minin = fmin(minin, value);
      // Assumes 0.0 <= 'value' <= 255.0
      //value = value / 255.0;
      //single_scan.push_back(max_range * (1.0 - value));
      // Assumes -1.0 <= 'value' <= 1.0
      value = (value * 0.5) + 0.5;
      single_scan.push_back(max_range * (1.0 - value));
      // Assumes 0.0 <= 'value' <= 1.0
      //single_scan.push_back(max_range * (1.0 - value));
    }
    all_scans_.push_back(single_scan);
  }
  std::cout << "max input: " << maxin << std::endl;
  std::cout << "min input: " << minin << std::endl;
  viewing_recreation_ = true;
}

void getSideBySide() {
  getDownsampledScansFromTxt();
  all_scans_sidekick_ = all_scans_;
  all_scans_.clear();
  getScansFromBag();
  side_by_side_viewing_ = true;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Need bag name..." << std::endl;
    return 1;
  }
  bag_name_ = argv[1];
  getScansFromBag();
  //getScansFromTxt();
  //getDownsampledScansFromTxt();
  //getSideBySide();
  std::cout << "loaded" << std::endl;

  ros::init(argc, argv, "scanalyzer");
  ros::NodeHandle nh;

  mouse_subscriber_ = nh.subscribe("Gui/VectorLocalization/GuiMouseClickEvents", 1, MouseClickCallback);
  keyboard_subscriber_ = nh.subscribe("Gui/VectorLocalization/GuiKeyboardEvents", 1, KeyboardEventCallback);
  display_publisher_ = nh.advertise<gui_msgs::LidarDisplayMsg>("Gui/VectorLocalization/Gui", 1, true);
  //selection_publisher_ = nh.advertise<std_msgs::String>("QueryFilename", 1, true);
  selection_publisher_ = nh.advertise<gui_msgs::ScanFeatureMsg>("ScanFeature", 1, true);

  ros::spin();

  return 0;
}
