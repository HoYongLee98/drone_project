# launch/full_chain.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # ---- 공통 인자 ----
    uri_arg = DeclareLaunchArgument('uri', default_value='radio://0/80/2M/E7E7E7E7E7')
    image_topic_arg = DeclareLaunchArgument('image_topic', default_value='/camera/image_raw',
                                            description='Creative 노드가 구독할 이미지 토픽')

    # Bridge 파라미터
    period_ms_arg = DeclareLaunchArgument('period_ms', default_value='100')
    publish_rate_hz_arg = DeclareLaunchArgument('publish_rate_hz', default_value='20.0')
    use_state_estimate_arg = DeclareLaunchArgument('use_state_estimate', default_value='true')
    cmd_rate_hz_arg = DeclareLaunchArgument('cmd_rate_hz', default_value='50.0')
    hover_timeout_s_arg = DeclareLaunchArgument('hover_timeout_s', default_value='0.5')
    arm_on_start_arg = DeclareLaunchArgument('arm_on_start', default_value='true')
    hl_takeoff_dur_arg = DeclareLaunchArgument('hl_takeoff_duration_s', default_value='2.0')
    hl_land_dur_arg   = DeclareLaunchArgument('hl_land_duration_s', default_value='2.0')
    hl_goto_dur_arg   = DeclareLaunchArgument('hl_goto_duration_s', default_value='2.5')

    # Camera 노드 설정(패키지/실행파일/파라미터)
    camera_pkg_arg  = DeclareLaunchArgument('camera_pkg',  default_value='mini_drone')
    camera_exec_arg = DeclareLaunchArgument('camera_exec', default_value='ai_deck_camera',
                                            description='ai_deck_camera_node.py의 console_scripts 이름')
    # 필요 시 카메라 전용 파라미터(예시)
    cam_width_arg   = DeclareLaunchArgument('cam_width', default_value='320')
    cam_height_arg  = DeclareLaunchArgument('cam_height', default_value='240')
    cam_fps_arg     = DeclareLaunchArgument('cam_fps', default_value='30')

    # Creative 노드 파라미터
    takeoff_height_arg = DeclareLaunchArgument('takeoff_height_m', default_value='0.5')
    takeoff_timeout_arg = DeclareLaunchArgument('takeoff_timeout_s', default_value='8.0')
    forward_speed_arg = DeclareLaunchArgument('forward_speed_mps', default_value='0.3')
    forward1_time_arg = DeclareLaunchArgument('forward1_time_s', default_value='3.0')
    forward2_time_arg = DeclareLaunchArgument('forward2_time_s', default_value='3.0')
    greet_dz_arg = DeclareLaunchArgument('greet_delta_z_m', default_value='0.15')
    greet_pause_arg = DeclareLaunchArgument('greet_pause_s', default_value='0.8')
    avoid_speed_arg = DeclareLaunchArgument('avoid_lateral_speed_mps', default_value='0.25')
    avoid_time_arg = DeclareLaunchArgument('avoid_time_s', default_value='2.0')
    hover_cmd_rate_arg = DeclareLaunchArgument('hover_cmd_rate_hz', default_value='30.0')
    safety_front_min_arg = DeclareLaunchArgument('safety_front_min_m', default_value='0.5')

    use_yolo_arg = DeclareLaunchArgument('use_yolo', default_value='true')
    yolo_model_path_arg = DeclareLaunchArgument('yolo_model_path', default_value='yolov8n.pt')
    yolo_conf_th_arg = DeclareLaunchArgument('yolo_conf_th', default_value='0.4')
    detect_center_weight_arg = DeclareLaunchArgument('detect_center_weight', default_value='0.3')
    detect_timeout_arg = DeclareLaunchArgument('detect_timeout_s', default_value='12.0')

    # ---- LCs ----
    uri = LaunchConfiguration('uri')
    image_topic = LaunchConfiguration('image_topic')

    period_ms = LaunchConfiguration('period_ms')
    publish_rate_hz = LaunchConfiguration('publish_rate_hz')
    use_state_estimate = LaunchConfiguration('use_state_estimate')
    cmd_rate_hz = LaunchConfiguration('cmd_rate_hz')
    hover_timeout_s = LaunchConfiguration('hover_timeout_s')
    arm_on_start = LaunchConfiguration('arm_on_start')
    hl_takeoff_duration_s = LaunchConfiguration('hl_takeoff_duration_s')
    hl_land_duration_s = LaunchConfiguration('hl_land_duration_s')
    hl_goto_duration_s = LaunchConfiguration('hl_goto_duration_s')

    camera_pkg  = LaunchConfiguration('camera_pkg')
    camera_exec = LaunchConfiguration('camera_exec')
    cam_width   = LaunchConfiguration('cam_width')
    cam_height  = LaunchConfiguration('cam_height')
    cam_fps     = LaunchConfiguration('cam_fps')

    takeoff_height_m = LaunchConfiguration('takeoff_height_m')
    takeoff_timeout_s = LaunchConfiguration('takeoff_timeout_s')
    forward_speed_mps = LaunchConfiguration('forward_speed_mps')
    forward1_time_s = LaunchConfiguration('forward1_time_s')
    forward2_time_s = LaunchConfiguration('forward2_time_s')
    greet_delta_z_m = LaunchConfiguration('greet_delta_z_m')
    greet_pause_s = LaunchConfiguration('greet_pause_s')
    avoid_lateral_speed_mps = LaunchConfiguration('avoid_lateral_speed_mps')
    avoid_time_s = LaunchConfiguration('avoid_time_s')
    hover_cmd_rate_hz = LaunchConfiguration('hover_cmd_rate_hz')
    safety_front_min_m = LaunchConfiguration('safety_front_min_m')

    use_yolo = LaunchConfiguration('use_yolo')
    yolo_model_path = LaunchConfiguration('yolo_model_path')
    yolo_conf_th = LaunchConfiguration('yolo_conf_th')
    detect_center_weight = LaunchConfiguration('detect_center_weight')
    detect_timeout_s = LaunchConfiguration('detect_timeout_s')

    # ---- Nodes ----
    bridge = Node(
        package='mini_drone', executable='cf_bridge', name='cf_bridge', output='screen',
        parameters=[{
            'uri': uri,
            'period_ms': period_ms,
            'publish_rate_hz': publish_rate_hz,
            'use_state_estimate': use_state_estimate,
            'cmd_rate_hz': cmd_rate_hz,
            'hover_timeout_s': hover_timeout_s,
            'arm_on_start': arm_on_start,
            'hl_takeoff_duration_s': hl_takeoff_duration_s,
            'hl_land_duration_s': hl_land_duration_s,
            'hl_goto_duration_s': hl_goto_duration_s,
        }],
        respawn=True, respawn_delay=2.0,
    )

    camera = Node(
        package=camera_pkg, executable=camera_exec, name='ai_deck_camera', output='screen',
        # ai_deck_camera_node.py에서 사용하는 파라미터 이름에 맞춰 조정
        parameters=[{
            'image_topic': image_topic,   # 노드에서 이 이름을 지원하면 바로 사용
            'width': cam_width,
            'height': cam_height,
            'fps': cam_fps,
        }],
        # 노드가 내부적으로 /ai_deck/image_raw로만 퍼블리시한다면 아래처럼 리맵하거나,
        # creative_behavior 쪽 image_topic을 /ai_deck/image_raw로 넘겨도 됨.
        # remappings=[('/ai_deck/image_raw', image_topic)],
        respawn=True, respawn_delay=2.0,
    )

    creative = TimerAction(
        period=2.0,  # Bridge/Camera가 먼저 떠서 토픽 준비될 시간 확보
        actions=[Node(
            package='mini_drone', executable='creative_behavior', name='creative_behavior', output='screen',
            parameters=[{
                'image_topic': image_topic,
                'odom_topic': '/cf/odom',
                'front_range_topic': '/cf/range/front',
                'takeoff_height_m': takeoff_height_m,
                'takeoff_timeout_s': takeoff_timeout_s,
                'forward_speed_mps': forward_speed_mps,
                'forward1_time_s': forward1_time_s,
                'forward2_time_s': forward2_time_s,
                'greet_delta_z_m': greet_delta_z_m,
                'greet_pause_s': greet_pause_s,
                'avoid_lateral_speed_mps': avoid_lateral_speed_mps,
                'avoid_time_s': avoid_time_s,
                'hover_cmd_rate_hz': hover_cmd_rate_hz,
                'safety_front_min_m': safety_front_min_m,
                'use_yolo': use_yolo,
                'yolo_model_path': yolo_model_path,
                'yolo_conf_th': yolo_conf_th,
                'detect_center_weight': detect_center_weight,
                'detect_timeout_s': detect_timeout_s,
            }],
            respawn=True, respawn_delay=2.0,
        )]
    )

    return LaunchDescription([
        # args
        uri_arg, image_topic_arg,
        period_ms_arg, publish_rate_hz_arg, use_state_estimate_arg, cmd_rate_hz_arg,
        hover_timeout_s_arg, arm_on_start_arg, hl_takeoff_dur_arg, hl_land_dur_arg, hl_goto_dur_arg,
        camera_pkg_arg, camera_exec_arg, cam_width_arg, cam_height_arg, cam_fps_arg,
        takeoff_height_arg, takeoff_timeout_arg, forward_speed_arg, forward1_time_arg, forward2_time_arg,
        greet_dz_arg, greet_pause_arg, avoid_speed_arg, avoid_time_arg, hover_cmd_rate_arg, safety_front_min_arg,
        use_yolo_arg, yolo_model_path_arg, yolo_conf_th_arg, detect_center_weight_arg, detect_timeout_arg,

        # nodes
        bridge,
        camera,
        creative,
    ])
