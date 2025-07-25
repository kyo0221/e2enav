#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # パッケージパス
    pkg_share = FindPackageShare('e2enav')
    
    # 設定ファイルパス
    config_file = PathJoinSubstitution([
        pkg_share, 'config', 'params.yaml'
    ])
    
    # Launch引数
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_model_name = DeclareLaunchArgument(
        'model_name',
        default_value='simple_imitation_model.pt',
        description='Name of the trained model file'
    )
    
    declare_max_linear_vel = DeclareLaunchArgument(
        'max_linear_vel',
        default_value='3.0',
        description='Maximum linear velocity [m/s]'
    )
    
    declare_max_angular_vel = DeclareLaunchArgument(
        'max_angular_vel',
        default_value='1.57',
        description='Maximum angular velocity [rad/s]'
    )
    
    # 推論ノード
    inference_node = Node(
        package='e2enav',
        executable='inference_node',
        name='simple_inference_node',
        parameters=[
            config_file,
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'model_name': LaunchConfiguration('model_name'),
                'max_linear_vel': LaunchConfiguration('max_linear_vel'),
                'max_angular_vel': LaunchConfiguration('max_angular_vel'),
            }
        ],
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        declare_use_sim_time,
        declare_model_name,
        declare_max_linear_vel,
        declare_max_angular_vel,
        inference_node,
    ])