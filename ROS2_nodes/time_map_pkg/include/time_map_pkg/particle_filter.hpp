#ifndef PARTICLE_FILTER_HPP
#define PARTICLE_FILTER_HPP

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>
#include <memory>
#include <vector>
#include <random>

// Structure to represent a particle
struct Particle
{
  double x;      // Position X
  double y;      // Position Y
  double theta;  // Orientation
  double w;      // Weight
  double v_lin;      // Velocity
  double v_ang;    // Angular parameter
  double dt;     // Time delta parameter
};

class ParticleFilter : public rclcpp::Node
{
public:
  ParticleFilter();

private:
  void init_particles();
  void twist_callback(const geometry_msgs::msg::Twist::SharedPtr msg);
  void z_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  void map_query_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  void new_experiment_callback(const std_msgs::msg::Int32::SharedPtr msg);
  void prediction_phase();
  
  //publishes the particles cloud including weights
  void publish_particles();
  
  //publishes the poses to Coppelia for querying the map
  void publish_particles_poses();

  //publishes the poses to Coppelia, just for visualization
  void publish_poses_for_show();

  std_msgs::msg::Float32MultiArray::SharedPtr last_z_;
  
  rclcpp::TimerBase::SharedPtr timer_;  

  std::vector<Particle> particles_;
  std::mt19937 rng_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr z_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr query_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr new_exp_sub_;

  //Publishes the poses for querying the map
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr poses_pub_;

  //Publishes the particles cloud, including weights
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr cloud_pub_;
  
  //Publishes the poses just for visualization
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr show_poses_pub_;
  
  
  
  // Parameters
  int num_particles_ = 100;
  double process_noise_x_ = 0.01;
  double process_noise_y_ = 0.01;
  double process_noise_theta_ = 0.005;
  double init_x_= 0.0;
  double init_y_= 2.0;
  double init_theta_= 0.0;

  double last_cmd_v_ ;
  double last_cmd_w_ ;
  double last_dt_ ;
  
  
};

#endif  // PARTICLE_FILTER_HPP
