#include "e2enav/inference_node.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono>

namespace e2enav
{

SimpleInferenceNode::SimpleInferenceNode(const rclcpp::NodeOptions &options) 
  : SimpleInferenceNode("", options) {}

SimpleInferenceNode::SimpleInferenceNode(const std::string &name_space, const rclcpp::NodeOptions &options)
  : rclcpp::Node("simple_inference_node", name_space, options),
    device_(torch::kCUDA),
    model_loaded_(false),
    image_available_(false),
    interval_ms_(get_parameter("interval_ms").as_int()),
    model_name_(get_parameter("model_name").as_string()),
    linear_max_(get_parameter("max_linear_vel").as_double()),
    angular_max_(get_parameter("max_angular_vel").as_double()),
    image_width_(get_parameter("image_width").as_int()),
    image_height_(get_parameter("image_height").as_int()),
    visualize_flag_(get_parameter("visualize_flag").as_bool()),
    autonomous_flag_(false),
    inference_count_(0),
    total_inference_time_(0.0)
{
  if (!torch::cuda::is_available()) {
    RCLCPP_WARN(this->get_logger(), "CUDA not available, using CPU");
    device_ = torch::kCPU;
  }
  
  autonomous_flag_subscriber_ = this->create_subscription<std_msgs::msg::Bool>(
    "/autonomous", 10, std::bind(&SimpleInferenceNode::autonomousFlagCallback, this, std::placeholders::_1));
  
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/image_raw", 10, std::bind(&SimpleInferenceNode::imageCallback, this, std::placeholders::_1));
  
  cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
  
  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(interval_ms_),
    std::bind(&SimpleInferenceNode::performInference, this));
  
  try {
    std::string package_share = ament_index_cpp::get_package_share_directory("e2enav");
    std::string model_path = package_share + "/config/models/" + model_name_;
    
    if (loadModel(model_path)) {
      RCLCPP_INFO(this->get_logger(), "✅ Model loaded successfully from: %s", model_path.c_str());
      RCLCPP_INFO(this->get_logger(), "🖥️  Using device: %s", device_ == torch::kCUDA ? "CUDA" : "CPU");
    } else {
      RCLCPP_ERROR(this->get_logger(), "❌ Failed to load model from: %s", model_path.c_str());
    }
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "❌ Error during model loading: %s", e.what());
  }
  
  RCLCPP_INFO(this->get_logger(), "🚀 SimpleInferenceNode initialized");
  RCLCPP_INFO(this->get_logger(), "Parameters:");
  RCLCPP_INFO(this->get_logger(), "  - Interval: %d ms", interval_ms_);
  RCLCPP_INFO(this->get_logger(), "  - Model: %s", model_name_.c_str());
  RCLCPP_INFO(this->get_logger(), "  - Image size: %dx%d", image_width_, image_height_);
  RCLCPP_INFO(this->get_logger(), "  - Max linear vel: %.2f", linear_max_);
  RCLCPP_INFO(this->get_logger(), "  - Max angular vel: %.2f", angular_max_);
  RCLCPP_INFO(this->get_logger(), "  - Visualization: %s", visualize_flag_ ? "ON" : "OFF");
}

bool SimpleInferenceNode::loadModel(const std::string& model_path)
{
  try {
    model_ = torch::jit::load(model_path);
    model_.to(device_);
    model_.eval();
    model_loaded_ = true;
    return true;
  } catch (const c10::Error& e) {
    RCLCPP_ERROR(this->get_logger(), "TorchScript load error: %s", e.what());
    model_loaded_ = false;
    return false;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Model load error: %s", e.what());
    model_loaded_ = false;
    return false;
  }
}

void SimpleInferenceNode::autonomousFlagCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
  bool previous_flag = autonomous_flag_;
  autonomous_flag_ = msg->data;
  
  if (autonomous_flag_ != previous_flag) {
    RCLCPP_INFO(this->get_logger(), "🔄 Autonomous mode: %s", 
                autonomous_flag_ ? "ENABLED" : "DISABLED");
    
    if (!autonomous_flag_) {
        geometry_msgs::msg::Twist stop_cmd;
      stop_cmd.linear.x = 0.0;
      stop_cmd.angular.z = 0.0;
      cmd_pub_->publish(stop_cmd);
    }
  }
}

void SimpleInferenceNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  try {
    // BGRA8画像をpassthrough encodingで受信
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgra8");
    
    // BGRA → RGB変換（Aチャンネルを破棄）
    cv::Mat rgb_image;
    cv::cvtColor(cv_ptr->image, rgb_image, cv::COLOR_BGRA2RGB);
    
    latest_image_ = rgb_image.clone();
    image_available_ = true;
  } catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    image_available_ = false;
  }
}

cv::Mat SimpleInferenceNode::preprocessImage(const cv::Mat& input_image, int target_height, int target_width)
{
  // 訓練時と同様の中央クロップ方式（shift_sign=0.0相当）
  int input_height = input_image.rows;
  int input_width = input_image.cols;
  
  cv::Mat processed;
  
  // 元画像が目標サイズより大きい場合は中央クロップ
  if (input_width >= target_width && input_height >= target_height) {
    // 中央から切り出し位置を計算
    int x_start = (input_width - target_width) / 2;
    int y_start = (input_height - target_height) / 2;
    
    // ROI（Region of Interest）でクロップ
    cv::Rect crop_rect(x_start, y_start, target_width, target_height);
    processed = input_image(crop_rect).clone();
    
    RCLCPP_DEBUG(this->get_logger(), "Center crop: %dx%d -> %dx%d (crop from %d,%d)", 
                 input_width, input_height, target_width, target_height, x_start, y_start);
  } else {
    // 小さい画像の場合は従来通りリサイズ
    cv::resize(input_image, processed, cv::Size(target_width, target_height));
    
    RCLCPP_DEBUG(this->get_logger(), "Resize: %dx%d -> %dx%d", 
                 input_width, input_height, target_width, target_height);
  }
  
  return processed;
}

void SimpleInferenceNode::performInference()
{
  if (!autonomous_flag_ || !model_loaded_ || !image_available_ || latest_image_.empty()) {
    return;
  }
  
  try {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    cv::Mat preprocessed = preprocessImage(latest_image_, image_height_, image_width_);
    
    at::Tensor image_tensor = torch::from_blob(
      preprocessed.data, 
      {1, image_height_, image_width_, 3}, 
      torch::kUInt8)
      .permute({0, 3, 1, 2})
      .clone()
      .to(torch::kFloat32)
      .div(255.0)
      .to(device_);
    
    // 推論実行
    at::Tensor output;
    {
      torch::NoGradGuard no_grad;
      output = model_.forward({image_tensor}).toTensor();
    }
    
    double predicted_angular = output.item<float>();
    predicted_angular = std::clamp(predicted_angular, -angular_max_, angular_max_);
    
    geometry_msgs::msg::Twist cmd_msg;
    cmd_msg.linear.x = linear_max_;
    cmd_msg.angular.z = predicted_angular;
    cmd_pub_->publish(cmd_msg);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double inference_time_ms = duration.count() / 1000.0;
    
    inference_count_++;
    total_inference_time_ += inference_time_ms;
    
    if (inference_count_ % 100 == 0) {
      double avg_inference_time = total_inference_time_ / inference_count_;
      RCLCPP_INFO(this->get_logger(), 
                  "📊 Inference #%d - Angular: %.3f rad/s, Time: %.2f ms (avg: %.2f ms)",
                  inference_count_, predicted_angular, inference_time_ms, avg_inference_time);
    }
    
    if (visualize_flag_ && inference_count_ % 30 == 0) {
      RCLCPP_DEBUG(this->get_logger(), 
                   "🎯 Prediction: %.3f rad/s (clamped from raw output)", predicted_angular);
    }
    
  } catch (const c10::Error& e) {
    RCLCPP_ERROR(this->get_logger(), "TorchScript inference error: %s", e.what());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Inference error: %s", e.what());
  }
}

}  // namespace e2enav

int main(int argc, char **argv) 
{
  rclcpp::init(argc, argv);
  
  rclcpp::NodeOptions node_options;
  node_options.allow_undeclared_parameters(true);
  node_options.automatically_declare_parameters_from_overrides(true);
  
  auto node = std::make_shared<e2enav::SimpleInferenceNode>(node_options);
  
  try {
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node->get_logger(), "Node error: %s", e.what());
  }
  
  rclcpp::shutdown();
  return 0;
}