#!/usr/bin/env python3

import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2
import numpy as np
import webdataset as wds
import io
import json
import time
import threading
from ament_index_python.packages import get_package_share_directory


class DataCollector(Node):
    def __init__(self):
        super().__init__('simple_data_collector')
        
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('interval_ms', 100)
        self.declare_parameter('log_name', 'dataset')
        self.declare_parameter('max_data_count', 5000000)
        self.declare_parameter('samples_per_shard', 1000)
        self.declare_parameter('enable_compression', True)
        self.declare_parameter('joy_button_toggle', 1)
        
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.joy_topic = self.get_parameter('joy_topic').get_parameter_value().string_value
        self.interval_ms = self.get_parameter('interval_ms').get_parameter_value().integer_value
        self.log_name = self.get_parameter('log_name').get_parameter_value().string_value
        self.max_data_count = self.get_parameter('max_data_count').get_parameter_value().integer_value
        self.samples_per_shard = self.get_parameter('samples_per_shard').get_parameter_value().integer_value
        self.enable_compression = self.get_parameter('enable_compression').get_parameter_value().bool_value
        self.joy_button_toggle = self.get_parameter('joy_button_toggle').get_parameter_value().integer_value
        
        workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.dataset_dir = os.path.join(workspace_path, 'config', self.log_name)
        self.webdataset_dir = os.path.join(self.dataset_dir, 'webdataset')
        os.makedirs(self.webdataset_dir, exist_ok=True)
        
        self.bridge = CvBridge()
        self.last_ang_vel = 0.0
        self.save_flag = False
        self.data_count = 0
        
        self.img_width = None
        self.img_height = None
        self.image_size_initialized = False
        
        self.current_shard = 0
        self.current_shard_count = 0
        self.shard_writer = None
        self.shard_lock = threading.Lock()
        self._last_processed_image = None
        self.completed_shards = []
        
        self.last_joy_buttons = []
        self.collection_timer = None
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_callback, 10)
        self.joy_sub = self.create_subscription(Joy, self.joy_topic, self.joy_callback, 10)
        
        self.get_logger().info(f"DataCollector initialized")
        self.get_logger().info(f"Saving webdataset to: {self.webdataset_dir}")
        self.get_logger().info(f"Collection interval: {self.interval_ms} ms ({1000/self.interval_ms:.1f} Hz)")
        self.get_logger().info(f"Joy control - Toggle button: {self.joy_button_toggle}")
        self.get_logger().info(f"Image size: Auto-detect from ROS image topic")
        self.get_logger().info(f"Data collection: {'ACTIVE' if self.save_flag else 'INACTIVE'}")
    
    def _init_shard_writer(self):
        with self.shard_lock:
            if self.enable_compression:
                shard_pattern = os.path.join(self.webdataset_dir, "shard_%06d.tar.gz")
            else:
                shard_pattern = os.path.join(self.webdataset_dir, "shard_%06d.tar")
            
            self.shard_writer = wds.ShardWriter(shard_pattern, maxcount=self.samples_per_shard)
            self.current_shard_count = 0
            
            actual_filename = shard_pattern % self.current_shard
            self.get_logger().info(f"🗂️ シャード開始: {actual_filename}")
    
    def joy_callback(self, msg):
        current_buttons = msg.buttons
        
        # ボタンの状態変化を検出（押下のエッジ検出）
        if len(self.last_joy_buttons) == len(current_buttons):
            # トグルボタンが押された（OFF -> ON）
            if (len(current_buttons) > self.joy_button_toggle and 
                current_buttons[self.joy_button_toggle] == 1 and 
                self.last_joy_buttons[self.joy_button_toggle] == 0):
                
                # 収集モードをトグル
                self.save_flag = not self.save_flag
                
                if self.save_flag:
                    self.start_data_collection()
                else:
                    self.stop_data_collection()
        
        self.last_joy_buttons = list(current_buttons)
    
    def start_data_collection(self):
        if not self.image_size_initialized:
            self.get_logger().warn("画像サイズが初期化されていません。画像を受信してから再試行してください。")
            self.save_flag = False
            return
        
        # シャードライター初期化
        if self.shard_writer is None:
            self._init_shard_writer()
        
        # タイマー開始
        if self.collection_timer is None:
            self.collection_timer = self.create_timer(
                self.interval_ms / 1000.0,
                self.collect_data_sample
            )
        
        self.get_logger().info("🟢 データ収集開始 (Joy button toggled ON)")
    
    def stop_data_collection(self):
            # タイマー停止
        if self.collection_timer is not None:
            self.collection_timer.cancel()
            self.collection_timer = None
        
        self.get_logger().info("🔴 データ収集停止 (Joy button toggled OFF)")
    
    def cmd_callback(self, msg):
            self.last_ang_vel = msg.angular.z
    
    def image_callback(self, msg):
        cv_image_bgra = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')    
        cv_image_rgb = cv2.cvtColor(cv_image_bgra, cv2.COLOR_BGRA2RGB)  # zedxがbgra8画像を出版するので, 色変換
        
        if not self.image_size_initialized:
            self.img_height, self.img_width = cv_image_rgb.shape[:2]
            self.image_size_initialized = True
            self.get_logger().info(f"📏 RGB画像サイズを自動検出: {self.img_width}x{self.img_height}")

        self._last_processed_image = cv_image_rgb.copy()
    
    def collect_data_sample(self):
        if not self.save_flag or self._last_processed_image is None:
            return
        
        save_success = self._save_webdataset_sample(
            self._last_processed_image, 
            self.last_ang_vel, 
            time.time()
        )
        
        if save_success:
            self.data_count += 1
            self.current_shard_count += 1
            
            if self.current_shard_count >= self.samples_per_shard:
                self._close_current_shard_and_start_next()
            
            if self.data_count % 10 == 0:
                self.get_logger().info(f"📸 Sample saved: {self.data_count:05d}, ang_vel: {self.last_ang_vel:.3f}")
            
            if self.data_count >= self.max_data_count:
                self.get_logger().info(f"🏁 最大データ数に到達: {self.max_data_count}")
                self.save_flag = False
                self.stop_data_collection()
        else:
            raise RuntimeError("サンプル保存に失敗しました")
    
    def _close_current_shard_and_start_next(self):
        with self.shard_lock:
            self.current_shard += 1
            self.current_shard_count = 0
            self.get_logger().info(f"🗂️ シャード {self.current_shard-1} 完了、次のシャードを開始")
    
    def _save_webdataset_sample(self, image, angle, timestamp):
        if self.shard_writer is None:
            raise RuntimeError("ShardWriter is not initialized")
        
        with self.shard_lock:
            img_uint8 = image.astype(np.uint8)
            img_buffer = io.BytesIO()
            np.save(img_buffer, img_uint8)
            img_data = img_buffer.getvalue()
            
            # メタデータを作成
            metadata = {
                'angle': float(angle),
                'timestamp': float(timestamp),
                'image_width': int(self.img_width),
                'image_height': int(self.img_height),
                'save_format': 'numpy_uint8',
                'image_shape': list(img_uint8.shape),
                'image_dtype': 'uint8',
            }
            
            sample_key = f"{self.data_count:06d}"
            sample_data = {
                "__key__": sample_key,
                "npy": img_data,
                "metadata.json": json.dumps(metadata),
                "angle.json": json.dumps({"angle": float(angle)})
            }
            
            self.shard_writer.write(sample_data)
            return True
    
    def _save_dataset_stats(self):
        if not self.image_size_initialized:
            raise RuntimeError("画像サイズが初期化されていないため、統計情報を保存できません")
        
        stats = {
            "total_samples": self.data_count,
            "total_shards": len(self.completed_shards) + (1 if self.current_shard_count > 0 else 0),
            "samples_per_shard": self.samples_per_shard,
            "compression_enabled": self.enable_compression,
            "save_format": "numpy_uint8",
            "dataset_directory": self.webdataset_dir,
            "image_size": [int(self.img_height), int(self.img_width)],
            "original_image_size": True,  # 元画像サイズを保持
            "collection_frequency_hz": 1000.0 / self.interval_ms,
            "max_data_count": self.max_data_count,
            "image_dtype": "uint8",  # データ型情報を追加
        }
        
        stats_file = os.path.join(self.webdataset_dir, "dataset_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        self.get_logger().info(f"📊 統計情報を保存: {stats_file}")
    
    def destroy_node(self):
        self.get_logger().info("🛑 データ収集ノードを終了します...")
        
        # タイマー停止
        if self.collection_timer is not None:
            self.collection_timer.cancel()
        
        # WebDatasetの保存を完了
        if self.shard_writer is not None:
            with self.shard_lock:
                self.shard_writer.close()
                self.get_logger().info(f"🗂️ 最終シャード保存完了")
        
        # 統計情報ファイルを保存
        self._save_dataset_stats()
        
        self.get_logger().info(f"🏁 データ収集完了: {self.data_count} samples")
        if self.image_size_initialized:
            self.get_logger().info(f"📏 収集画像サイズ: {self.img_width}x{self.img_height}")
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[INFO] Graceful shutdown by Ctrl+C.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)


if __name__ == '__main__':
    main()