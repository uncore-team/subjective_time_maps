#include "time_map_pkg/particle_filter.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>

ParticleFilter::ParticleFilter() : rclcpp::Node("particle_filter"), rng_(std::random_device{}())
{
  // Declare and get parameters
  this->declare_parameter<int>("num_particles", num_particles_);
  this->declare_parameter<double>("process_noise_x", process_noise_x_);
  this->declare_parameter<double>("process_noise_y", process_noise_y_);
  this->declare_parameter<double>("process_noise_theta", process_noise_theta_);
  this->declare_parameter<double>("init_x", init_x_);
  this->declare_parameter<double>("init_y", init_y_);
  this->declare_parameter<double>("init_theta", init_theta_);

  this->get_parameter("num_particles", num_particles_);
  this->get_parameter("process_noise_x", process_noise_x_);
  this->get_parameter("process_noise_y", process_noise_y_);
  this->get_parameter("process_noise_theta", process_noise_theta_);
  this->get_parameter("init_x", init_x_);
  this->get_parameter("init_y", init_y_);
  this->get_parameter("init_theta", init_theta_);

  //this->init_particles();
   particles_.clear();
 // Initialize particles with uniform distribution
  this->init_particles();
  
  cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
    "cmd_vel", 1, std::bind(&ParticleFilter::twist_callback, this, std::placeholders::_1));

  z_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
    "z", 1, std::bind(&ParticleFilter::z_callback, this, std::placeholders::_1));
  
  query_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
    "map_query", 1, std::bind(&ParticleFilter::map_query_callback, this, std::placeholders::_1));

  new_exp_sub_ = this->create_subscription<std_msgs::msg::Int32>(
    "start_experiment", 1, std::bind(&ParticleFilter::new_experiment_callback, this, std::placeholders::_1));
    
  // Create publisher for particles as Float32MultiArray
  poses_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("get_query", 1);
  
  cloud_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("particles_cloud", 1);

  show_poses_pub_= this->create_publisher<std_msgs::msg::Float32MultiArray>("show_poses", 1);

  //Create timer for publishing the particles at fixed rate 10 Hz 
  
  timer_ = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&ParticleFilter::prediction_phase, this));   

  RCLCPP_INFO(this->get_logger(), "ParticleFilter node initialized with %d particulitas", num_particles_);
   //set debug logging
  this->get_logger().set_level(rclcpp::Logger::Level::Debug);

  
}
void ParticleFilter::init_particles()
{
  particles_.clear();
 // Initialize particles with uniform distribution
  RCLCPP_INFO(this->get_logger(), "[PF]Init x: '%f' Init y: '%f' Init Theta: '%f'",init_x_,init_y_,init_theta_);
  std::uniform_real_distribution<double> dist_pos(-0.1, 0.1);
  std::uniform_real_distribution<double> dist_theta(-0.01,0.01);
  
  particles_.resize(num_particles_);
  
  for (int i = 0; i < num_particles_; ++i) {
    particles_[i].x = init_x_+dist_pos(rng_);
    particles_[i].y = init_y_+dist_pos(rng_);
    particles_[i].theta = init_theta_;//+dist_theta(rng_);
    particles_[i].w = 1.0 / num_particles_;  // Equal initial weights
    particles_[i].v_lin = 0.0;  // Initial velocity
    particles_[i].v_ang = 0.0;  //Info from time map (angular velocity)
    particles_[i].dt = 0.0;  //Info from time map (time delta)
  }
  last_cmd_v_ = 0.0;
  last_cmd_w_ = 0.0;
  last_dt_ = 0.0;
}

void ParticleFilter::new_experiment_callback(const std_msgs::msg::Int32::SharedPtr msg)
{
  //A new experiment starts, so re-initialize particles
  this->init_particles();
}
void ParticleFilter::map_query_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  
  double weight_sum = 0.0;
  size_t num_queries = msg->data.size() / 9;

  
  for (size_t i = 0; i < num_queries && i < particles_.size(); ++i) {
    size_t base_idx = i * 9;
    particles_[i].v_lin = static_cast<double>(msg->data[base_idx + 6]);
    particles_[i].v_ang = static_cast<double>(msg->data[base_idx + 7]);
    particles_[i].dt = static_cast<double>(msg->data[base_idx + 8]);

    //compute weight based on last_z_ if available
    //substract the last_z_ from the first 8 values of the query
    if (last_z_ && last_z_->data.size() >= 4) {
      double innovation = 0.0;
      for (size_t j = 0; j < 4; ++j) {
        innovation = innovation + static_cast<double>(msg->data[base_idx + j+2]) - static_cast<double>(last_z_->data[j]);
      }
      //square innovation
      innovation = innovation * innovation; 
      
      // Convert distance to weight (higher innovation = lower weight)
      particles_[i].w = std::exp(-innovation / (2.0 * 0.75));  // Using sigma = 1.0
      weight_sum += particles_[i].w;
    }
  }
  
  // Normalize weights
  
  if (weight_sum > 0.0) {
    for (auto& particle : particles_) {
      particle.w /= weight_sum;
    }
  }
  
  publish_particles();

  //resampling particles based on weights (systematic resampling)

  std::vector<Particle> new_particles;
  new_particles.resize(particles_.size());
  std::uniform_real_distribution<double> dist_u(0.0, 1.0 / particles_.size());
  double r = dist_u(rng_);
  double c = particles_[0].w;
  size_t i = 0;
  for (size_t m = 0; m < particles_.size(); ++m) {
    double U = r + m * (1.0 / particles_.size());
    while (U > c) {
      i++;
      c += particles_[i].w;
    }
    new_particles[m] = particles_[i];
  }
  particles_ = std::move(new_particles);

}

void ParticleFilter::z_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "Received measurement data of size: %lu", msg->data.size());   
  // Store the latest measurement
  //This only contins the z values
  last_z_ = msg;
  this->publish_particles_poses();
}

void ParticleFilter::twist_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  last_cmd_v_ = msg->linear.x;
  last_cmd_w_ = msg->angular.z; 
  publish_poses_for_show();
}

void ParticleFilter::prediction_phase()
{
  std::normal_distribution<double> noise_x(0.0, process_noise_x_);
  std::normal_distribution<double> noise_y(0.0, process_noise_y_);
  std::normal_distribution<double> noise_theta(0.0, process_noise_theta_);

  double dt = 0.05;  // Use the actual time delta from the last twist message  
  double v = last_cmd_v_;
  double w = last_cmd_w_;

  for (auto& particle : particles_) {
    // Update particle pose based on motion model
    double delta_theta = w * dt;
    double delta_x = v * dt * std::cos(particle.theta + delta_theta / 2.0);
    double delta_y = v * dt * std::sin(particle.theta + delta_theta / 2.0);

    // Apply motion
    particle.x += delta_x + noise_x(rng_);
    particle.y += delta_y + noise_y(rng_);
    particle.theta += delta_theta ;//+ noise_theta(rng_);

    // Normalize theta to [-pi, pi]
    while (particle.theta > M_PI) particle.theta -= 2.0 * M_PI;
    while (particle.theta < -M_PI) particle.theta += 2.0 * M_PI;
  }
}


void ParticleFilter::publish_poses_for_show()
{
  auto msg = std::make_unique<std_msgs::msg::Float32MultiArray>();

  // For showing I just need x,y
  msg->data.resize(particles_.size() * 2);

  for (size_t i = 0; i < particles_.size(); ++i) {
    size_t base_idx = i * 2;
    msg->data[base_idx + 0] = static_cast<float>(particles_[i].x);
    msg->data[base_idx + 1] = static_cast<float>(particles_[i].y);
  }
  show_poses_pub_->publish(std::move(msg));
}


void ParticleFilter::publish_particles_poses()
{
  RCLCPP_INFO(this->get_logger(), "Publishing %lu particles poses", particles_.size());
  auto msg = std::make_unique<std_msgs::msg::Float32MultiArray>();

  // Each particle pose has 3 values: x, y, theta
  msg->data.resize(particles_.size() * 3);

  for (size_t i = 0; i < particles_.size(); ++i) {
    size_t base_idx = i * 3;
    msg->data[base_idx + 0] = static_cast<float>(particles_[i].x);
    
    msg->data[base_idx + 1] = static_cast<float>(particles_[i].y);
    msg->data[base_idx + 2] = static_cast<float>(particles_[i].theta);
  }

  poses_pub_->publish(std::move(msg));
}
void ParticleFilter::publish_particles()
{
  auto msg = std::make_unique<std_msgs::msg::Float32MultiArray>();

  // Each particle has 7 values: x, y, theta, w, ang, dt
  msg->data.resize(particles_.size() * 7);

  for (size_t i = 0; i < particles_.size(); ++i) {
    size_t base_idx = i * 7;
    msg->data[base_idx + 0] = static_cast<float>(particles_[i].x);
    msg->data[base_idx + 1] = static_cast<float>(particles_[i].y);
    msg->data[base_idx + 2] = static_cast<float>(particles_[i].theta);
    msg->data[base_idx + 3] = static_cast<float>(particles_[i].w);
    msg->data[base_idx + 4] = static_cast<float>(particles_[i].v_lin);
    msg->data[base_idx + 5] = static_cast<float>(particles_[i].v_ang);
    msg->data[base_idx + 6] = static_cast<float>(particles_[i].dt);
  }
  cloud_pub_->publish(std::move(msg));
}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ParticleFilter>());
  rclcpp::shutdown();
  return 0;
}
