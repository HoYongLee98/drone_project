#!/usr/bin/env python3
import math
import threading
import time
from typing import Optional, Dict

from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray, Bool
from std_srvs.srv import Trigger
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import rclpy

from rcl_interfaces.msg import SetParametersResult

class ControlManager:
    """
    Crazyflie 제어 관리자.
      - dry_run=True 이면 실제 Crazyflie로 어떤 패킷도 보내지 않고, 로그만 출력
      - E-STOP 래치 지원: /cf/stop 으로 잠그면 /cf/estop_reset 전까지 모든 비행 명령 차단
      - 상태 방송: /cf/estop (std_msgs/Bool)
      - hover/HL 명령 구독 및 패턴(원형/스핀/사각) 실행
    """

    def __init__(self, node,
                 cmd_rate_hz: float = 50.0,
                 hover_timeout_s: float = 0.5,
                 hl_durations: Dict[str, float] = None):
        self.node = node

        # ---- Parameters ----
        # 드라이런(기본 True: 실제 전송 금지)
        self.node.declare_parameter('dry_run', True)
        # self.dry_run = bool(self.node.get_parameter('dry_run').get_parameter_value().bool_value or True)
        self.dry_run = bool(self.node.get_parameter('dry_run').get_parameter_value().bool_value)

        self.node.add_on_set_parameters_callback(self._on_set_params)
        self.node.get_logger().info(f'dry_run initial={self.dry_run}')

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

        # ---- E-STOP latch state ----
        self.estop_latched = False

        # ---- QoS ----
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        # ---- Publishers ----
        self.pub_estop = self.node.create_publisher(Bool, '/cf/estop', qos)

        # ---- Subscriptions (저수준 hover) ----
        self.node.create_subscription(TwistStamped, '/cf/cmd_hover', self._on_cmd_hover, qos)

        # ---- Subscriptions (High-level) ----
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
        self.node.create_service(Trigger, '/cf/stop', self._srv_stop_cb)             # 래치 + 즉시 STOP
        self.node.create_service(Trigger, '/cf/estop_reset', self._srv_estop_reset)  # 래치 해제
        self.node.create_service(Trigger, '/cf/notify_stop', self._srv_notify_cb)
        self.node.create_service(Trigger, '/cf/pattern/stop', self._srv_pattern_stop)

        # ---- Timers ----
        self._hover_timer = self.node.create_timer(1.0 / max(1.0, self.cmd_rate_hz), self._hover_tick)

        # ---- Pattern thread ----
        self._pat_th: Optional[threading.Thread] = None
        self._pat_stop = threading.Event()

        if self.dry_run:
            self.node.get_logger().info('[DRY-RUN] 실제 전송 없이 로그만 출력합니다. (-p dry_run:=false 로 전송 허용 가능)')

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

    def _on_set_params(self, params):
        for p in params:
            if p.name == 'dry_run':
                self.dry_run = bool(p.value)
                self.node.get_logger().warn(f'[PARAM] dry_run -> {self.dry_run}')
        return SetParametersResult(successful=True)

    # ========== E-STOP helpers ==========
    def _publish_estop_state(self):
        self.pub_estop.publish(Bool(data=self.estop_latched))

    # ========== Internal send wrappers (dry-run & E-STOP aware) ==========
    def _send_hover_setpoint(self, vx: float, vy: float, yawrate_deg: float, z: float):
        if self.estop_latched:
            self.node.get_logger().warn("[E-STOP] hover 차단됨")
            return
        if self.dry_run or self.cf is None:
            self.node.get_logger().info(f"[SIM hover] vx={vx:.3f} m/s, vy={vy:.3f} m/s, z={z:.3f} m, yawrate={yawrate_deg:.1f} deg/s")
            return
        try:
            self.cf.commander.send_hover_setpoint(vx, vy, yawrate_deg, z)
        except Exception as e:
            self.node.get_logger().warn(f'hover send failed: {e}')

    def _send_notify_stop(self):
        if self.dry_run or self.cf is None:
            self.node.get_logger().info("[SIM notify_setpoint_stop]")
            return
        try:
            self.cf.commander.send_notify_setpoint_stop()
        except Exception:
            pass

    def _send_stop(self):
        if self.dry_run or self.cf is None:
            self.node.get_logger().info("[SIM STOP setpoint]")
            return
        try:
            self.cf.commander.send_stop_setpoint()
        except Exception:
            pass

    def _enable_hl(self) -> bool:
        """HL 활성화. E-STOP 잠금 중이면 False. dry_run이면 로그만."""
        if self.estop_latched:
            self.node.get_logger().warn('[E-STOP] HL enable 차단됨')
            return False
        if self.dry_run or self.cf is None:
            self.node.get_logger().info("[SIM HL enable]")
            return True
        try:
            self.cf.param.set_value('commander.enHighLevel', '1')
            time.sleep(0.05)
            self.cf.commander.send_notify_setpoint_stop()
            return True
        except Exception as e:
            self.node.get_logger().warn(f'Enable HL skipped/failed: {e}')
            return False

    def _hl_takeoff(self, z: float, dur: float):
        if self.estop_latched:
            self.node.get_logger().warn('[E-STOP] takeoff 차단됨')
            return
        if self.dry_run or self.cf is None:
            self.node.get_logger().info(f"[SIM HL takeoff] z={z:.2f} m, dur={dur:.2f} s")
            return
        from cflib.crazyflie.high_level_commander import HighLevelCommander
        try:
            HighLevelCommander(self.cf).takeoff(z, dur)
        except Exception as e:
            self.node.get_logger().error(f'HL takeoff failed: {e}')

    def _hl_land(self, z: float, dur: float):
        if self.estop_latched:
            self.node.get_logger().warn('[E-STOP] land 차단됨')
            return
        if self.dry_run or self.cf is None:
            self.node.get_logger().info(f"[SIM HL land] z={z:.2f} m, dur={dur:.2f} s")
            return
        from cflib.crazyflie.high_level_commander import HighLevelCommander
        try:
            HighLevelCommander(self.cf).land(z, dur)
        except Exception as e:
            self.node.get_logger().error(f'HL land failed: {e}')

    def _hl_goto(self, x: float, y: float, z: float, yaw_rad: float, dur: float):
        if self.estop_latched:
            self.node.get_logger().warn('[E-STOP] goto 차단됨')
            return
        if self.dry_run or self.cf is None:
            self.node.get_logger().info(f"[SIM HL goto] x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw_rad:.2f} rad, dur={dur:.2f} s")
            return
        from cflib.crazyflie.high_level_commander import HighLevelCommander
        try:
            HighLevelCommander(self.cf).go_to(x, y, z, yaw_rad, dur, relative=False, linear=False)
        except Exception as e:
            self.node.get_logger().error(f'HL goto failed: {e}')

    # ========== Hover (low-level) ==========
    def _on_cmd_hover(self, msg: TwistStamped):
        if self.estop_latched:
            self.node.get_logger().warn('[E-STOP] hover 명령 무시')
            return
        with self._lock:
            self._last_hover = msg
            now = self.node.get_clock().now().nanoseconds * 1e-9
            self._last_hover_time = now

    def _hover_tick(self):
        with self._lock:
            cf = self.cf
            cmd = self._last_hover
            t0 = self._last_hover_time

        # 연결 없고 드라이런도 아니면 아무것도 안 함
        if (cf is None) and not self.dry_run:
            return

        # 래치 중: 계속 stop만 유지 전송(너무 자주일 필요는 없지만 여기선 주기와 동일)
        if self.estop_latched:
            self._send_stop()
            return

        now = self.node.get_clock().now().nanoseconds * 1e-9
        if (cmd is None) or (now - t0 > self.hover_timeout_s):
            self._send_notify_stop()
            return

        vx = float(cmd.twist.linear.x)
        vy = float(cmd.twist.linear.y)
        z = float(cmd.twist.linear.z)
        yawrate_deg = math.degrees(float(cmd.twist.angular.z))
        self._send_hover_setpoint(vx, vy, yawrate_deg, z)

    # ========== High-level (takeoff/land/goto) ==========
    def _on_hl_takeoff(self, msg: Float32):
        if not self._enable_hl():
            return
        self._hl_takeoff(float(msg.data), float(self.hl_durations['takeoff']))

    def _on_hl_land(self, msg: Float32):
        if not self._enable_hl():
            return
        self._hl_land(float(msg.data), float(self.hl_durations['land']))

    def _on_hl_goto(self, msg: PoseStamped):
        if not self._enable_hl():
            return
        qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)) if (qw or qx or qy or qz) else 0.0
        self._hl_goto(float(msg.pose.position.x),
                      float(msg.pose.position.y),
                      float(msg.pose.position.z),
                      float(yaw),
                      float(self.hl_durations['goto']))

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
        yaw_rate = speed / radius (rad/s) → deg/s
        """
        if self.estop_latched:
            self.node.get_logger().warn('[E-STOP] circle 패턴 차단됨')
            return

        d = list(msg.data)
        if len(d) < 4:
            self.node.get_logger().error('circle needs [radius_m, speed_mps, z_m, duration_s]')
            return
        radius, speed, z, duration = map(float, d[:4])
        omega = speed / max(1e-6, abs(radius))  # rad/s
        yawrate_deg = math.degrees(omega) * (1.0 if speed >= 0 else -1.0)

        def _run():
            dt = 1.0 / max(1.0, self.cmd_rate_hz)
            self.node.get_logger().info(f'[PATTERN circle] r={radius:.2f}m v={speed:.2f}m/s z={z:.2f} dur={duration:.2f}s (yaw={yawrate_deg:.1f}deg/s)')
            t0 = time.time()
            try:
                while (time.time() - t0) < duration and not self._pat_stop.is_set() and not self.estop_latched:
                    self._send_hover_setpoint(speed, 0.0, yawrate_deg, z)
                    time.sleep(dt)
            finally:
                self._send_notify_stop()
                self.node.get_logger().info('[PATTERN circle] done')

        self._start_pattern(_run)

    def _on_pattern_spin(self, msg: Float32):
        """
        제자리 회전: yaw_rate(rad/s), duration은 파라미터 spin_duration_s
        """
        if self.estop_latched:
            self.node.get_logger().warn('[E-STOP] spin 패턴 차단됨')
            return

        yawrate_deg = math.degrees(float(msg.data))
        duration = float(self.node.get_parameter('spin_duration_s').get_parameter_value().double_value or 3.0)

        def _run():
            dt = 1.0 / max(1.0, self.cmd_rate_hz)
            self.node.get_logger().info(f'[PATTERN spin] yawrate={yawrate_deg:.1f} deg/s dur={duration:.2f}s')
            t0 = time.time()
            try:
                while (time.time() - t0) < duration and not self._pat_stop.is_set() and not self.estop_latched:
                    # 주의: hover API의 z는 절대 고도. 0.0은 "현재 유지"가 아닙니다.
                    # 여기서는 그대로 0.0을 사용(테스트 목적으로). 필요 시 현재 고도를 넣도록 개선하세요.
                    self._send_hover_setpoint(0.0, 0.0, yawrate_deg, 0.0)
                    time.sleep(dt)
            finally:
                self._send_notify_stop()
                self.node.get_logger().info('[PATTERN spin] done')

        self._start_pattern(_run)

    def _on_pattern_square(self, msg: Float32MultiArray):
        """
        사각 경로: data = [side_m, speed_mps, z_m]
        각 변 직진 후 90도 회전
        """
        if self.estop_latched:
            self.node.get_logger().warn('[E-STOP] square 패턴 차단됨')
            return

        d = list(msg.data)
        if len(d) < 3:
            self.node.get_logger().error('square needs [side_m, speed_mps, z_m]')
            return
        L, speed, z = map(float, d[:3])
        turn_rate = float(self.node.get_parameter('square_turn_rate_deg_s').get_parameter_value().double_value or 90.0)
        move_time = abs(L / max(1e-6, speed))
        turn_time = 90.0 / max(1e-3, abs(turn_rate))  # sec

        def _run():
            dt = 1.0 / max(1.0, self.cmd_rate_hz)
            self.node.get_logger().info(f'[PATTERN square] L={L:.2f}m v={speed:.2f} z={z:.2f} (turn={turn_rate:.1f}deg/s)')
            try:
                for i in range(4):
                    # straight
                    t0 = time.time()
                    while (time.time() - t0) < move_time and not self._pat_stop.is_set() and not self.estop_latched:
                        self._send_hover_setpoint(speed, 0.0, 0.0, z)
                        time.sleep(dt)
                    # turn 90 deg
                    t1 = time.time()
                    yaw_deg_s = turn_rate if speed >= 0 else -turn_rate
                    while (time.time() - t1) < turn_time and not self._pat_stop.is_set() and not self.estop_latched:
                        self._send_hover_setpoint(0.0, 0.0, yaw_deg_s, z)
                        time.sleep(dt)
                self.node.get_logger().info('[PATTERN square] done')
            finally:
                self._send_notify_stop()

        self._start_pattern(_run)

    # ========== Services ==========
    def _srv_stop_cb(self, req, res):
        # 래치 + 즉시 모터 컷
        self.estop_latched = True
        self._publish_estop_state()
        self._send_stop()
        res.success = True
        res.message = 'E-STOP latched; motors stop'
        return res

    def _srv_estop_reset(self, req, res):
        # 사용자가 명시적으로 해제할 때만 열림
        self.estop_latched = False
        self._publish_estop_state()
        res.success = True
        res.message = 'E-STOP reset; command gate open'
        return res

    def _srv_notify_cb(self, req, res):
        # 래치 중이어도 notify 자체는 안전 동작으로 허용 가능
        self._send_notify_stop()
        res.success = True
        res.message = 'notify_setpoint_stop (or SIM) sent'
        return res
