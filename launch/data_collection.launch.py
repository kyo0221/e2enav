#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # パッケージパス
    pkg_share = FindPackageShare('e2enav')
    
    # 設定ファイルパス
    config_file = PathJoinSubstitution([
        pkg_share, 'config', 'data_collection.yaml'
    ])
    
    # Launch引数
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_show_rviz = DeclareLaunchArgument(
        'show_rviz',
        default_value='false',
        description='Launch RViz2 for visualization'
    )
    
    # データ収集ノード
    data_collector_node = Node(
        package='e2enav',
        executable='collect_data.py',
        name='simple_data_collector',
        parameters=[config_file, {
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }],
        output='screen',
        emulate_tty=True,
    )
    
    # RViz2（オプション）
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        condition=IfCondition(LaunchConfiguration('show_rviz')),
        output='screen'
    )
    
    return LaunchDescription([
        declare_use_sim_time,
        declare_show_rviz,
        data_collector_node,
        # rviz_node,  # 必要に応じてコメントアウト解除
    ])