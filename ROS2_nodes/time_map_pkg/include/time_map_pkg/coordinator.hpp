#ifndef COORDINATOR_HPP
#define COORDINATOR_HPP

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <memory>
#include <string>
#include <vector>

class Coordinator : public rclcpp::Node
{
public:
  Coordinator();
  
  //Send request for z data
  void ask_for_z();
  
  //Send a query at the ground truth position      
  void ask_for_gt_command();

  double last_dt=0.0;
  double sum_dt=0.0;

  std::vector<double> last_gt;

  int experiment_;  //1: ground truth, 2: particle filter standard, 3: particle filter average time map
  bool exp2_use_best_particle_ = false; // true: use best particle, false: weighted average

  

private:
  void cloud_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  void z_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  void map_query_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  void gt_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  void stats_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg);
  
  void process_particle_cloud_exp2(const std_msgs::msg::Float32MultiArray::SharedPtr& msg);
  void process_particle_cloud_exp3(const std_msgs::msg::Float32MultiArray::SharedPtr& msg);
  void publish_twist(double v_lin, double v_ang,double dt=0.0);
  

  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr z_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr cloud_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr map_query_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr ground_truth_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr stats_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr ask_z_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr start_exp_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr get_query_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr estimation_pub_;
  
  double last_x_,last_y_;

};

#endif  // COORDINATOR_HPP
