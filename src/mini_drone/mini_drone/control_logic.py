# src/mini_drone/control_logic.py
import math
import threading
import time
from typing import Optional, Dict

from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray
from std_srvs.srv import Trigger
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import rclpy

class ControlManager:
    """
    Bridge 노드에서 생성/보유:
      - attach_cf(cf) 호출되면 Crazyflie 핸들 연결
      - hover/HL 명령 구독 및 setpoint 전송 타이머 운영
      - 패턴 동작(원형, 스핀, 사각) 실행 스레드 관리
    """

    def __init__(self, node,
                 cmd_rate_hz: float = 50.0,
                 hover_timeout_s: float = 0.5,
                 hl_durations: Dict[str, float] = None):
        self.node = node
        self.cf = None
        self._lock = threading.Lock()
        self._last_hover: Optional[TwistStamped] = None
        self._last_hover_time = 0.0

        self.cmd_rate_hz = cmd_rate_hz
        self.hover_timeout_s = hover_timeout_s
        self.hl_durations = hl_durations or {
            'takeoff': 2.0,
            'land': 2.0,
            'goto': 2.5,
        }

        # QoS (센서/명령용)
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        # ---- Subscriptions (저수준 hover) ----
        self.node.create_subscription(TwistStamped, '/cf/cmd_hover', self._on_cmd_hover, qos)

        # ---- Subscriptions (HL) ----
        self.node.create_subscription(Float32, '/cf/hl/takeoff', self._on_hl_takeoff, qos)
        self.node.create_subscription(Float32, '/cf/hl/land', self._on_hl_land, qos)
        self.node.create_subscription(PoseStamped, '/cf/hl/goto', self._on_hl_goto, qos)

        # ---- Pattern commands ----
        # circle: Float32MultiArray [radius_m, speed_mps, z_m, duration_s]
        self.node.create_subscription(Float32MultiArray, '/cf/pattern/circle', self._on_pattern_circle, qos)
        # spin in place: Float32 yaw_rate_rad_s, duration 파라미터 사용
        self.node.declare_parameter('spin_duration_s', 3.0)
        self.node.create_subscription(Float32, '/cf/pattern/spin', self._on_pattern_spin, qos)
        # square: Float32MultiArray [side_m, speed_mps, z_m]
        self.node.declare_parameter('square_turn_rate_deg_s', 90.0)  # 회전 속도(도/초)
        self.node.create_subscription(Float32MultiArray, '/cf/pattern/square', self._on_pattern_square, qos)

        # ---- Services ----
        self.node.create_service(Trigger, '/cf/stop', self._srv_stop_cb)
        self.node.create_service(Trigger, '/cf/notify_stop', self._srv_notify_cb)
        self.node.create_service(Trigger, '/cf/pattern/stop', self._srv_pattern_stop)

        # ---- Timers ----
        self._hover_timer = self.node.create_timer(1.0 / max(1.0, self.cmd_rate_hz), self._hover_tick)

        # ---- Pattern thread ----
        self._pat_th: Optional[threading.Thread] = None
        self._pat_stop = threading.Event()

    # ========== Public API ==========
    def attach_cf(self, cf):
        """Bridge에서 연결 직후 호출"""
        with self._lock:
            self.cf = cf
        self.node.get_logger().info('ControlManager attached to CF')

    def detach_cf(self):
        with self._lock:
            self.cf = None

    def stop_patterns(self):
        self._pat_stop.set()
        if self._pat_th and self._pat_th.is_alive():
            self._pat_th.join(timeout=1.0)
        self._pat_th = None
        self._pat_stop.clear()

    # ========== Hover (low-level) ==========
    def _on_cmd_hover(self, msg: TwistStamped):
        with self._lock:
            self._last_hover = msg
            now = self.node.get_clock().now().nanoseconds * 1e-9
            self._last_hover_time = now

    def _hover_tick(self):
        with self._lock:
            cf = self.cf
            cmd = self._last_hover
            t0 = self._last_hover_time
        if cf is None:
            return
        now = self.node.get_clock().now().nanoseconds * 1e-9
        if (cmd is None) or (now - t0 > self.hover_timeout_s):
            try:
                cf.commander.send_notify_setpoint_stop()
            except Exception:
                pass
            return
        # hover setpoint (body vx, vy, yawrate_deg, z)
        vx = float(cmd.twist.linear.x)
        vy = float(cmd.twist.linear.y)
        z = float(cmd.twist.linear.z)
        yawrate_deg = math.degrees(float(cmd.twist.angular.z))
        try:
            cf.commander.send_hover_setpoint(vx, vy, yawrate_deg, z)
        except Exception as e:
            self.node.get_logger().warn(f'hover send failed: {e}')

    # ========== High-level (takeoff/land/goto) ==========
    def _enable_hl(self):
        with self._lock:
            cf = self.cf
        if cf is None:
            return False
        try:
            cf.param.set_value('commander.enHighLevel', '1')
            time.sleep(0.05)
            cf.commander.send_notify_setpoint_stop()
            return True
        except Exception as e:
            self.node.get_logger().warn(f'Enable HL skipped/failed: {e}')
            return False

    def _on_hl_takeoff(self, msg: Float32):
        with self._lock:
            cf = self.cf
        if cf is None:
            return
        if not self._enable_hl():
            return
        from cflib.crazyflie.high_level_commander import HighLevelCommander
        h = HighLevelCommander(cf)
        try:
            h.takeoff(float(msg.data), float(self.hl_durations['takeoff']))
            self.node.get_logger().info(f'HL takeoff to {float(msg.data):.2f} m')
        except Exception as e:
            self.node.get_logger().error(f'HL takeoff failed: {e}')

    def _on_hl_land(self, msg: Float32):
        with self._lock:
            cf = self.cf
        if cf is None:
            return
        if not self._enable_hl():
            return
        from cflib.crazyflie.high_level_commander import HighLevelCommander
        h = HighLevelCommander(cf)
        try:
            h.land(float(msg.data), float(self.hl_durations['land']))
            self.node.get_logger().info(f'HL land to {float(msg.data):.2f} m')
        except Exception as e:
            self.node.get_logger().error(f'HL land failed: {e}')

    def _on_hl_goto(self, msg: PoseStamped):
        with self._lock:
            cf = self.cf
        if cf is None:
            return
        if not self._enable_hl():
            return
        # yaw from quaternion (rad)
        qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)) if (qw or qx or qy or qz) else 0.0
        from cflib.crazyflie.high_level_commander import HighLevelCommander
        h = HighLevelCommander(cf)
        try:
            h.go_to(float(msg.pose.position.x),
                    float(msg.pose.position.y),
                    float(msg.pose.position.z),
                    float(yaw),
                    float(self.hl_durations['goto']),
                    relative=False, linear=False)
            self.node.get_logger().info(
                f'HL goto ({msg.pose.position.x:.2f},{msg.pose.position.y:.2f},{msg.pose.position.z:.2f}), yaw={yaw:.2f}')
        except Exception as e:
            self.node.get_logger().error(f'HL goto failed: {e}')

    # ========== Patterns (low-level continuous) ==========
    def _start_pattern(self, target_fn, *args, **kwargs):
        # 한 번에 하나만
        self.stop_patterns()
        self._pat_stop.clear()
        self._pat_th = threading.Thread(target=target_fn, args=args, kwargs=kwargs, daemon=True)
        self._pat_th.start()

    def _srv_pattern_stop(self, req, res):
        self.stop_patterns()
        res.success = True; res.message = 'pattern stopped'
        return res

    def _on_pattern_circle(self, msg: Float32MultiArray):
        """
        data = [radius_m, speed_mps, z_m, duration_s]
        - body-forward 속도 vx = speed
        - yaw_rate = speed / radius  (rad/s)  → deg/s 로 변환하여 hover setpoint로 전송
        - duration 동안 유지 → notify_stop 후 종료
        """
        d = list(msg.data)
        if len(d) < 4:
            self.node.get_logger().error('circle needs [radius_m, speed_mps, z_m, duration_s]')
            return
        radius, speed, z, duration = map(float, d[:4])
        omega = speed / max(1e-6, abs(radius))  # rad/s
        yawrate_deg = math.degrees(omega) * (1.0 if speed >= 0 else -1.0)

        def _run():
            with self._lock: cf = self.cf
            if cf is None:
                self.node.get_logger().warn('No CF attached')
                return
            self.node.get_logger().info(f'start circle r={radius:.2f}m v={speed:.2f}m/s z={z:.2f} dur={duration:.2f}s')
            t0 = time.time()
            dt = 1.0 / max(1.0, self.cmd_rate_hz)
            try:
                while (time.time() - t0) < duration and not self._pat_stop.is_set():
                    cf.commander.send_hover_setpoint(speed, 0.0, yawrate_deg, z)
                    time.sleep(dt)
            finally:
                try: cf.commander.send_notify_setpoint_stop()
                except Exception: pass
                self.node.get_logger().info('circle done')

        self._start_pattern(_run)

    def _on_pattern_spin(self, msg: Float32):
        """
        제자리 회전: yaw_rate(rad/s), duration은 파라미터 spin_duration_s
        """
        yawrate_deg = math.degrees(float(msg.data))
        duration = float(self.node.get_parameter('spin_duration_s').get_parameter_value().double_value or 3.0)

        def _run():
            with self._lock: cf = self.cf
            if cf is None:
                self.node.get_logger().warn('No CF attached')
                return
            self.node.get_logger().info(f'start spin yawrate={yawrate_deg:.1f} deg/s dur={duration:.2f}s')
            t0 = time.time()
            dt = 1.0 / max(1.0, self.cmd_rate_hz)
            try:
                while (time.time() - t0) < duration and not self._pat_stop.is_set():
                    cf.commander.send_hover_setpoint(0.0, 0.0, yawrate_deg, 0.0)  # z는 유지(0: 내부 유지)
                    time.sleep(dt)
            finally:
                try: cf.commander.send_notify_setpoint_stop()
                except Exception: pass
                self.node.get_logger().info('spin done')

        self._start_pattern(_run)

    def _on_pattern_square(self, msg: Float32MultiArray):
        """
        사각 경로: data = [side_m, speed_mps, z_m]
        - 각 변을 직진, 변 끝에서 90도 회전
        - 회전 속도는 파라미터 square_turn_rate_deg_s 사용
        """
        d = list(msg.data)
        if len(d) < 3:
            self.node.get_logger().error('square needs [side_m, speed_mps, z_m]')
            return
        L, speed, z = map(float, d[:3])
        turn_rate = float(self.node.get_parameter('square_turn_rate_deg_s').get_parameter_value().double_value or 90.0)
        move_time = abs(L / max(1e-6, speed))
        turn_time = 90.0 / max(1e-3, abs(turn_rate))  # sec

        def _run():
            with self._lock: cf = self.cf
            if cf is None:
                self.node.get_logger().warn('No CF attached')
                return
            self.node.get_logger().info(f'start square L={L:.2f}m v={speed:.2f} z={z:.2f}')
            dt = 1.0 / max(1.0, self.cmd_rate_hz)
            try:
                for i in range(4):
                    # straight
                    t0 = time.time()
                    while (time.time() - t0) < move_time and not self._pat_stop.is_set():
                        cf.commander.send_hover_setpoint(speed, 0.0, 0.0, z)
                        time.sleep(dt)
                    # turn 90 deg
                    yaw_deg_s = turn_rate if speed >= 0 else -turn_rate
                    t1 = time.time()
                    while (time.time() - t1) < turn_time and not self._pat_stop.is_set():
                        cf.commander.send_hover_setpoint(0.0, 0.0, yaw_deg_s, z)
                        time.sleep(dt)
                self.node.get_logger().info('square done')
            finally:
                try: cf.commander.send_notify_setpoint_stop()
                except Exception: pass

        self._start_pattern(_run)

    # ========== Services ==========
    def _srv_stop_cb(self, req, res):
        with self._lock: cf = self.cf
        try:
            if cf: cf.commander.send_stop_setpoint()
            res.success = True; res.message = 'STOP setpoint sent'
        except Exception as e:
            res.success = False; res.message = f'Failed: {e}'
        return res

    def _srv_notify_cb(self, req, res):
        with self._lock: cf = self.cf
        try:
            if cf: cf.commander.send_notify_setpoint_stop()
            res.success = True; res.message = 'notify_setpoint_stop sent'
        except Exception as e:
            res.success = False; res.message = f'Failed: {e}'
        return res
