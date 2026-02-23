#include "time_map_pkg/coordinator.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>


Coordinator::Coordinator() : rclcpp::Node("coordinator")
{
  std::string z_topic = "z";
  std::string cloud_topic = "particles_cloud";
  std::string cmd_vel_topic = "cmd_vel";
  std::string ask_z_topic = "ask_z";
  std::string map_query_topic = "map_value_at";
  std::string ground_truth_topic = "ground_truth";
  std::string get_query_topic = "get_map_value";


  this->declare_parameter("cloud_topic", cloud_topic);
  this->declare_parameter("cmd_vel_topic", cmd_vel_topic);
  this->declare_parameter("ask_z_topic", ask_z_topic);
  this->declare_parameter("map_query_topic", map_query_topic);
  this->declare_parameter("ground_truth_topic", ground_truth_topic);
  this->declare_parameter("get_query_topic", get_query_topic);
  this->declare_parameter("experiment",3);
  this->declare_parameter("estimation_method", 1); // 0: best particle, 1: weighted average


  this->get_parameter("cloud_topic", cloud_topic);
  this->get_parameter("cmd_vel_topic", cmd_vel_topic);
  this->get_parameter("ask_z_topic", ask_z_topic);
  this->get_parameter("map_query_topic", map_query_topic);
  this->get_parameter("ground_truth_topic", ground_truth_topic);
  this->get_parameter("get_query_topic", get_query_topic);
  
  experiment_=this->get_parameter("experiment").as_int();
  int estimation_method = this->get_parameter("estimation_method").as_int();
  exp2_use_best_particle_ = (estimation_method == 0);

  ground_truth_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
    ground_truth_topic, 1, std::bind(&Coordinator::gt_callback, this, std::placeholders::_1));

  map_query_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
    map_query_topic, 1, std::bind(&Coordinator::map_query_callback, this, std::placeholders::_1));

  cloud_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
    cloud_topic, 1, std::bind(&Coordinator::cloud_callback, this, std::placeholders::_1));  
    
  joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    "joint_states", 1, std::bind(&Coordinator::joint_states_callback, this, std::placeholders::_1));  

  stats_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
    "stats", 1, std::bind(&Coordinator::stats_callback, this, std::placeholders::_1));
  cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic, 1);

  ask_z_pub_ = this->create_publisher<std_msgs::msg::Int32>(ask_z_topic, 1);
  start_exp_pub_ = this->create_publisher<std_msgs::msg::Int32>("start_experiment", 1);
  get_query_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(get_query_topic, 1);

  estimation_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("estimation", 1);

  RCLCPP_INFO(this->get_logger(), "Coordinator node initialized");

  //set debug logging
  this->get_logger().set_level(rclcpp::Logger::Level::Debug);

  last_gt.resize(3);
}

void Coordinator::stats_callback
        (const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  //print values
  RCLCPP_ERROR(this->get_logger(), "Stats received: total_time=%.5f, idle_time=%.5f, num_cmd=%d, distance=%.5f, min_obs=%.5f, success=%.5f",
    msg->data[0], sum_dt, static_cast<int>(msg->data[1]), msg->data[2], msg->data[3], msg->data[4]);
  
  // write the results into a csv file including a human-readable timestamp
  std::ofstream stats_file;
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm now_tm;
  localtime_r(&now_c, &now_tm);

  stats_file.open("experiment_stats.csv", std::ios_base::app); // append mode
  stats_file << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S") << ","
             << experiment_ << ","
             << msg->data[0] << ","
              << sum_dt << ","
             << static_cast<int>(msg->data[1]) << ","
             << msg->data[2] << ","
             << msg->data[3] << ","
             << msg->data[4] << "\n";
             
  stats_file.close();

  
    //stop robot and wait 1 seconds, this also set last_dt to 0
  this->publish_twist(0.0, 0.0);
  // reset sum_dt
  sum_dt =0.0;
  
  
  std::this_thread::sleep_for(std::chrono::duration<double>(1.0));
  //notify start of experiment with a one-time message
  std_msgs::msg::Int32 new_exp_msg;
  this->start_exp_pub_->publish(new_exp_msg);
}
void Coordinator::gt_callback
        (const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  // Copy msg to last_gt
  //fast like a cheetah
  std::copy(msg->data.begin(), msg->data.end(), last_gt.begin());
}

void Coordinator::joint_states_callback
        (const sensor_msgs::msg::JointState::SharedPtr msg)
{
  /*const auto &position = msg->position;
  const auto &velocity = msg->velocity;
  
  RCLCPP_INFO(this->get_logger(), "Left wheel pos: '%f' Right wheel pos: '%f'",position[0],position[1]);
  RCLCPP_INFO(this->get_logger(), "Left wheel vel: '%f' Right wheel vel: '%f'",velocity[0],velocity[1]);*/
}

 
void Coordinator::map_query_callback
        (const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  
  RCLCPP_DEBUG(this->get_logger(), "Received cmd_vel v=[%.5f],w=[%.5f],dt=[%.5f]", 
    msg->data[6],msg->data[7],msg->data[8]);
    
    auto v = msg->data[6];
    auto w = msg->data[7];
    const double xtarg{-1.65};
    const double ytarg{2.9};
    auto d = std::sqrt(std::pow((last_x_-xtarg),2)+std::pow((last_y_-ytarg),2));
    if (d < 0.15) 
    {
     v=0;
     w=0;
    }
    this->publish_twist(v, w, msg->data[8]);
  
  
}
void Coordinator::publish_twist(double v_lin, double v_ang,double dt)
{
  geometry_msgs::msg::Twist twist_msg;

  twist_msg.linear.x = v_lin;
  if (twist_msg.linear.x > 0.15) twist_msg.linear.x = 0.15;
  if (twist_msg.linear.x < -0.15) twist_msg.linear.x = -0.15;
  twist_msg.linear.y = 0.0;
  twist_msg.linear.z = 0.0;

  twist_msg.angular.x = 0.0;
  twist_msg.angular.y = 0.0;
  twist_msg.angular.z = v_ang;
  if (twist_msg.angular.z > 0.6) twist_msg.angular.z = 0.6;
  if (twist_msg.angular.z < -0.6) twist_msg.angular.z = -0.6;
  RCLCPP_INFO(this->get_logger(), "Vlin: '%f' Vang: '%f' dt:'%f'",twist_msg.linear.x,twist_msg.angular.z,dt);
 
  cmd_pub_->publish(std::move(twist_msg));
  this->last_dt=dt;
} 

void Coordinator::ask_for_z()
{
  std_msgs::msg::Int32 msg;
  this->ask_z_pub_->publish(msg);

}


void Coordinator::cloud_callback
        (const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{

  const auto &data = msg->data;
  const size_t particle_count = data.size() / 7;
  RCLCPP_DEBUG(this->get_logger(), "Received cloud with %lu particles", particle_count);
  if (particle_count == 0) {
    return;
  }

  if (this->experiment_ == 2) {
    process_particle_cloud_exp2(msg);
  }
  else if (this->experiment_ == 3) {
    process_particle_cloud_exp3(msg);
  } 

}

static void ComputePoseFromCloud(const std_msgs::msg::Float32MultiArray::SharedPtr& msg,double &weighted_x,double &weighted_y,double &weighted_theta)
{
  const auto &data = msg->data;
  const size_t particle_count = data.size() / 7;
  double sum_weights = 0.0;
  weighted_x = 0.0;
  weighted_y = 0.0;
  double sum_C = 0.0;
  double sum_S = 0.0;
  for (size_t i = 0, idx = 0; i < particle_count; ++i, idx += 7) 
  {
      const double x = data[idx + 0];
      const double y = data[idx + 1];
      const double theta = data[idx + 2];
      const double weight = data[idx + 3];
      sum_weights += weight;
      weighted_x += x * weight;
      weighted_y += y * weight;
      sum_C += weight * std::cos(theta);
      sum_S += weight * std::sin(theta);
    }
    weighted_theta = std::atan2(sum_S, sum_C);
    if (sum_weights > 0.0) 
    {
      weighted_x /= sum_weights;
      weighted_y /= sum_weights;
    }
}

void Coordinator::process_particle_cloud_exp3(const std_msgs::msg::Float32MultiArray::SharedPtr& msg)
{
  const auto &data = msg->data;
  const size_t particle_count = data.size() / 7;
  if (particle_count == 0) {
    return;
  }

  // Compute weighted average of v,w and dt
  double sum_weights = 0.0;
  double weighted_v = 0.0;
  double weighted_w = 0.0;
  double weighted_dt = 0.0;
  
  
  for (size_t i = 0, idx = 0; i < particle_count; ++i, idx += 7) {
    const double weight = data[idx + 3]; 
    const double v = data[idx + 4];
    const double w = data[idx + 5];
    const double dt = data[idx + 6];

    sum_weights += weight;
    weighted_v += v * weight;
    weighted_w += w * weight;
    weighted_dt += dt * weight;
  }

  double avg_v = weighted_v / sum_weights;
  double avg_w = weighted_w / sum_weights;
  double avg_dt = weighted_dt / sum_weights;
  
  double weighted_x;
  double weighted_y;
  double weighted_theta;
  ComputePoseFromCloud(msg,weighted_x,weighted_y,weighted_theta);
  
  RCLCPP_INFO(this->get_logger(), "Particle pose x: '%f' Particle pose y: '%f' Theta: '%f'",weighted_x,weighted_y,weighted_theta);
  const double xtarg{-1.65};
  const double ytarg{2.9};
  auto d = std::sqrt(std::pow((weighted_x-xtarg),2)+std::pow((weighted_y-ytarg),2));
  if (d < 0.15) 
  {
     avg_v=0;
     avg_w=0;
  }
  this->publish_twist(avg_v, avg_w, avg_dt);
  
}

void Coordinator::process_particle_cloud_exp2(const std_msgs::msg::Float32MultiArray::SharedPtr& msg)
{
  const auto &data = msg->data;
  const size_t particle_count = data.size() / 7;
  if (particle_count == 0) {
    return;
  }
  std_msgs::msg::Float32MultiArray query_msg;
  query_msg.data.resize(3);

  if (exp2_use_best_particle_) {
    // Select best particle (highest weight)
    size_t best_particle_index = 0;
    double best_weight = data[3];
    for (size_t i = 1, idx = 10; i < particle_count; ++i, idx += 7) {
      double w = data[idx];
      if (w > best_weight) {
        best_weight = w;
        best_particle_index = i;
      }
    }
    size_t best_idx = best_particle_index * 7;
    query_msg.data[0] = data[best_idx + 0];
    query_msg.data[1] = data[best_idx + 1];
    query_msg.data[2] = data[best_idx + 2];
    RCLCPP_DEBUG(this->get_logger(), "Exp2: using best particle (w=%.5f) at x=%.5f y=%.5f th=%.5f",
                 best_weight, query_msg.data[0], query_msg.data[1], query_msg.data[2]);
  } else {
    // Weighted average of x, y, theta
    double sum_weights = 0.0;
    double weighted_x = 0.0;
    double weighted_y = 0.0;
    double sum_C = 0.0;
    double sum_S = 0.0;
    for (size_t i = 0, idx = 0; i < particle_count; ++i, idx += 7) {
      const double x = data[idx + 0];
      const double y = data[idx + 1];
      const double theta = data[idx + 2];
      const double weight = data[idx + 3];
      sum_weights += weight;
      weighted_x += x * weight;
      weighted_y += y * weight;
      sum_C += weight * std::cos(theta);
      sum_S += weight * std::sin(theta);
    }
    double weighted_theta = std::atan2(sum_S, sum_C);
    if (sum_weights > 0.0) {
      weighted_x /= sum_weights;
      weighted_y /= sum_weights;
    }
    query_msg.data[0] = static_cast<float>(weighted_x);
    query_msg.data[1] = static_cast<float>(weighted_y);
    query_msg.data[2] = static_cast<float>(weighted_theta);
    RCLCPP_DEBUG(this->get_logger(), "Exp2: using weighted avg at x=%.5f y=%.5f th=%.5f",
                 weighted_x, weighted_y, weighted_theta);
  }
  // Publish in two topics
  last_x_ = query_msg.data[0];
  last_y_ = query_msg.data[1];
  estimation_pub_->publish(query_msg);
  get_query_pub_->publish(query_msg);

}


void Coordinator::ask_for_gt_command()
{
  auto query_msg = std::make_unique<std_msgs::msg::Float32MultiArray>();
  query_msg->data.resize(3);
  //this is experiment 1: query at ground truth
  if (last_gt.size()<3)
  {
    RCLCPP_WARN(this->get_logger(), "Ground truth not received yet, cannot ask for gt command");
    return;
  }
  query_msg->data[0] = last_gt[0];
  query_msg->data[1] = last_gt[1];
  query_msg->data[2] = last_gt[2];
  RCLCPP_DEBUG(this->get_logger(), "Asking for gt command at x=%.5f y=%.5f th=%.5f",
               query_msg->data[0], query_msg->data[1], query_msg->data[2]);
  get_query_pub_->publish(std::move(query_msg));
}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  //wait 2 seconds to let other nodes start
  std::this_thread::sleep_for(std::chrono::duration<double>(2.0)); 
  
  auto coordinator_node = std::make_shared<Coordinator>();
  
  RCLCPP_INFO(coordinator_node->get_logger(), "Starting Experiment %d", coordinator_node->experiment_);
  
  while (rclcpp::ok())
  {
    
    rclcpp::spin_some(coordinator_node);

    if (coordinator_node->experiment_==1)  //This is experiment 1: query at ground truth
    {

      coordinator_node->ask_for_gt_command();
    }
    else
    {   //For experiments 2 and 3: particle filter
      coordinator_node->ask_for_z();
    }
    
    //wait until last_dt is set by the map_query_callback (max 1 second)
    auto start_time = std::chrono::steady_clock::now();
    while (coordinator_node->last_dt==0.0 && rclcpp::ok()) {
      rclcpp::spin_some(coordinator_node);
      auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - start_time).count();
      if (elapsed >= 1.0) break;
    }

    if (coordinator_node->last_dt > 0.0) {
      //Sleep for last_dt seconds
      coordinator_node->sum_dt += coordinator_node->last_dt;
      RCLCPP_DEBUG(coordinator_node->get_logger(), "Sleeping for dt=%.5f seconds, sum_dt=%.5f", coordinator_node->last_dt, coordinator_node->sum_dt);
      std::this_thread::sleep_for(std::chrono::duration<double>(coordinator_node->last_dt));
      coordinator_node->last_dt = 0.0;
    
    } 

  }
  
  rclcpp::shutdown();
  return 0;
}
