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
#include "sensor_msgs/LaserScan.h"
#include "gui_msgs/GuiKeyboardEvent.h"
#include "gui_msgs/GuiMouseClickEvent.h"
#include "gui_msgs/LidarDisplayMsg.h"
#include "../perception_tools/perception_2d.h"

using std::string;
using std::vector;

ros::Subscriber keyboard_subscriber_;
ros::Publisher display_publisher_;

gui_msgs::LidarDisplayMsg display_message_;

string bag_name_;
vector<vector<float>> all_scans_;

int current_view_ = 1;

vector<int> id_in_progress_;

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

void pubScan() {
  //TODO: figure out why there is lag / weird behavior in the scan display
  std::cout << "scan number: " << current_view_ << std::endl;
  display_message_.points_x.clear();
  display_message_.points_y.clear();
  vector<float> scan = all_scans_[current_view_ - 1];
  float start_angle = -135.0;
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

//void saveScan(const string filename) {
//  vector<float> scan = all_scans_[current_view_ - 1];
//  todo: save scan?
//}

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
  if (static_cast<uint32_t>(msg.modifiers) == 2) {
    float th1 = atan2(msg.mouse_down.y, msg.mouse_down.x);
    float th2 = atan2(msg.mouse_up.y, msg.mouse_up.x);
    //TODO: test
    //TODO: figure out how to save / use this info
    //TODO: labeling ability?
  }
}

void KeyboardEventCallback(const gui_msgs::GuiKeyboardEvent& msg) {
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
//  else if (msg.keycode == 0x56) { // key code 86, 'v' for save
//    saveScan("ScansOfInterest.txt");
//  }
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
    all_scans_.push_back(single_scan);
  }
  vector<vector<float>> normal;
  vector<vector<float>> personalized;
  for (size_t i = 0; i < all_scans_.size(); ++i) {
    if (i < (all_scans_.size()/2)) {
      normal.push_back(all_scans_[i]);
    }
    else {
      personalized.push_back(all_scans_[i]);
    }
  }
  all_scans_.clear();
  for (size_t i = 0; i < normal.size(); ++i) {
    all_scans_.push_back(normal[i]);
    all_scans_.push_back(personalized[i]);
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Need bag name..." << std::endl;
    return 1;
  }
  bag_name_ = argv[1];
  getScansFromBag();
  //getScansFromTxt();

  ros::init(argc, argv, "scanalyzer");
  ros::NodeHandle nh;

  keyboard_subscriber_ = nh.subscribe("Gui/VectorLocalization/GuiKeyboardEvents", 1, KeyboardEventCallback);
  display_publisher_ = nh.advertise<gui_msgs::LidarDisplayMsg>("Gui/VectorLocalization/Gui", 1, true);

  ros::spin();

  return 0;
}
