#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>

#include <e2enav/visibility_control.h>

namespace e2enav
{

class E2ENAV_PUBLIC SimpleInferenceNode : public rclcpp::Node
{
public:
  explicit SimpleInferenceNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  
  explicit SimpleInferenceNode(const std::string& name_space, const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  void autonomousFlagCallback(const std_msgs::msg::Bool::SharedPtr msg);
  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
  void performInference();
  
  bool loadModel(const std::string& model_path);
  cv::Mat preprocessImage(const cv::Mat& input_image, int target_height, int target_width);
  
  // ROS通信
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr autonomous_flag_subscriber_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  
  // モデル関連
  torch::jit::script::Module model_;
  torch::Device device_;
  bool model_loaded_;
  
  // 画像処理
  cv::Mat latest_image_;
  bool image_available_;
  
  // パラメータ
  const int interval_ms_;
  const std::string model_name_;
  const double linear_max_;
  const double angular_max_;
  const int image_width_;
  const int image_height_;
  const bool visualize_flag_;
  
  // 状態管理
  bool autonomous_flag_;
  
  // 統計
  int inference_count_;
  double total_inference_time_;
};

}  // namespace e2enav