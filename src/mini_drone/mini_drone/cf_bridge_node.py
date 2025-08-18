#!/usr/bin/env python3
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu, Range, BatteryState
from geometry_msgs.msg import Vector3Stamped, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from std_srvs.srv import Trigger

# Crazyflie
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

def rpy_to_quat(roll, pitch, yaw):  # rad
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cr, sr = math.cos(roll*0.5), math.sin(roll*0.5)
    qx = sr*cp*cy - cr*sp*cy
    qy = cr*sp*sy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    qw = cr*cp*cy + sr*sp*cy
    return qw, qx, qy, qz

class CfBridgeNode(Node):
    """
    단일 링크 오너:
      - Publish:
        /cf/imu (sensor_msgs/Imu)
        /cf/rpy (geometry_msgs/Vector3Stamped, deg)
        /cf/odom (nav_msgs/Odometry)          # stateEstimate 기반(옵션)
        /cf/battery (sensor_msgs/BatteryState)
        /cf/range/{front,back,left,right,up,down} (sensor_msgs/Range, m)
      - Subscribe (컨트롤):
        /cf/cmd_hover (TwistStamped)          # body vx, vy [m/s], z [m], yaw_rate [rad/s]
        /cf/hl/takeoff (Float32)              # 목표 고도[m]
        /cf/hl/land (Float32)                 # 목표 고도[m] 보통 0.0
        /cf/hl/goto (PoseStamped)             # x,y,z + yaw(quat)
      - Service:
        /cf/stop (std_srvs/Trigger)           # STOP setpoint
        /cf/notify_stop (std_srvs/Trigger)    # notify_setpoint_stop
    """

    def __init__(self):
        super().__init__('cf_bridge_node')

        # ---- Parameters ----
        self.declare_parameter('uri', 'radio://0/80/2M/E7E7E7E7E7')
        self.declare_parameter('period_ms', 100)         # cflib log period
        self.declare_parameter('publish_rate_hz', 20.0)  # ROS publish
        self.declare_parameter('use_state_estimate', True)
        self.declare_parameter('cmd_rate_hz', 50.0)      # hover 전송 주기
        self.declare_parameter('hover_timeout_s', 0.5)
        self.declare_parameter('arm_on_start', True)
        # HL durations
        self.declare_parameter('hl_takeoff_duration_s', 2.0)
        self.declare_parameter('hl_land_duration_s', 2.0)
        self.declare_parameter('hl_goto_duration_s', 2.5)

        p = lambda n: self.get_parameter(n).get_parameter_value()
        self.uri = p('uri').string_value
        self.period_ms = int(p('period_ms').integer_value or 100)
        self.pub_rate = float(p('publish_rate_hz').double_value or 20.0)
        self.use_state = bool(p('use_state_estimate').bool_value or True)
        self.cmd_rate = float(p('cmd_rate_hz').double_value or 50.0)
        self.hover_timeout_s = float(p('hover_timeout_s').double_value or 0.5)
        self.arm_on_start = bool(p('arm_on_start').bool_value or True)
        self.hltake_dur = float(p('hl_takeoff_duration_s').double_value or 2.0)
        self.hlland_dur = float(p('hl_land_duration_s').double_value or 2.0)
        self.hlgoto_dur = float(p('hl_goto_duration_s').double_value or 2.5)

        # ---- QoS ----
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        # ---- Publishers ----
        self.pub_imu = self.create_publisher(Imu, '/cf/imu', qos)
        self.pub_rpy = self.create_publisher(Vector3Stamped, '/cf/rpy', qos)
        self.pub_odom = self.create_publisher(Odometry, '/cf/odom', qos) if self.use_state else None
        self.pub_batt = self.create_publisher(BatteryState, '/cf/battery', qos)
        self.range_pubs = {
            'front': self.create_publisher(Range, '/cf/range/front', qos),
            'back' : self.create_publisher(Range, '/cf/range/back', qos),
            'left' : self.create_publisher(Range, '/cf/range/left', qos),
            'right': self.create_publisher(Range, '/cf/range/right', qos),
            'up'   : self.create_publisher(Range, '/cf/range/up', qos),
            'down' : self.create_publisher(Range, '/cf/range/down', qos),
        }

        # ---- Control interfaces ----
        self.create_subscription(TwistStamped, '/cf/cmd_hover', self._on_cmd_hover, qos)
        self.create_subscription(Float32, '/cf/hl/takeoff', self._on_hl_takeoff, qos)
        self.create_subscription(Float32, '/cf/hl/land', self._on_hl_land, qos)
        self.create_subscription(PoseStamped, '/cf/hl/goto', self._on_hl_goto, qos)
        self.create_service(Trigger, '/cf/stop', self._srv_stop_cb)
        self.create_service(Trigger, '/cf/notify_stop', self._srv_notify_cb)

        # ---- State (updated by cflib & cmds) ----
        self._lock = threading.Lock()
        self._state = {
            'roll_deg': None, 'pitch_deg': None, 'yaw_deg': None,
            'gyro': {'x': None, 'y': None, 'z': None},
            'acc' : {'x': None, 'y': None, 'z': None},
            'pos' : {'x': None, 'y': None, 'z': None},
            'vel' : {'x': None, 'y': None, 'z': None},
            'range': {'front': None, 'back': None, 'left': None, 'right': None, 'up': None, 'down': None},
            'vbat': None
        }
        self._last_hover = None
        self._last_hover_time = 0.0

        # ---- Threads & Timers ----
        self._cf_thread = threading.Thread(target=self._cf_worker, daemon=True)
        self._cf_thread.start()
        self.create_timer(1.0/max(1.0, self.pub_rate), self._publish_all)  # telemetry
        self.create_timer(1.0/max(1.0, self.cmd_rate), self._hover_tick)   # hover send

    # ---------- Crazyflie worker ----------
    def _cf_worker(self):
        cflib.crtp.init_drivers(enable_debug_driver=False)
        try:
            with SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cache')) as scf:
                cf = scf.cf
                self.cf = cf
                self.get_logger().info(f'Connected to {self.uri}')

                # Arm(가능한 펌웨어에 한함)
                if self.arm_on_start:
                    try:
                        cf.platform.send_arming_request(True)
                        self.get_logger().info('Arming request sent')
                    except Exception as e:
                        self.get_logger().warn(f'Arm not supported/failed: {e}')

                # ---- LogConfigs (<=26B 묶음) ----
                period = self.period_ms
                lgs = []

                lg_att = LogConfig(name='LG_ATT', period_in_ms=period)
                lg_att.add_variable('stabilizer.roll','float')
                lg_att.add_variable('stabilizer.pitch','float')
                lg_att.add_variable('stabilizer.yaw','float'); lgs.append(lg_att)

                lg_gyro = LogConfig(name='LG_GYRO', period_in_ms=period)
                lg_gyro.add_variable('gyro.x','float')
                lg_gyro.add_variable('gyro.y','float')
                lg_gyro.add_variable('gyro.z','float'); lgs.append(lg_gyro)

                lg_acc = LogConfig(name='LG_ACC', period_in_ms=period)
                lg_acc.add_variable('acc.x','float')
                lg_acc.add_variable('acc.y','float')
                lg_acc.add_variable('acc.z','float'); lgs.append(lg_acc)

                lg_bat = LogConfig(name='LG_BAT', period_in_ms=period)
                lg_bat.add_variable('pm.vbat','float'); lgs.append(lg_bat)

                if self.use_state:
                    lg_posvel = LogConfig(name='LG_POSVEL', period_in_ms=period)
                    for v in ['stateEstimate.x','stateEstimate.y','stateEstimate.z',
                              'stateEstimate.vx','stateEstimate.vy','stateEstimate.vz']:
                        lg_posvel.add_variable(v,'float')
                    lgs.append(lg_posvel)

                lg_rng = LogConfig(name='LG_RNG', period_in_ms=period)
                for name, typ in [
                    ('range.front','uint16_t'), ('range.back','uint16_t'),
                    ('range.left','uint16_t'), ('range.right','uint16_t'),
                    ('range.up','uint16_t'), ('range.zrange','uint16_t')
                ]:
                    try: lg_rng.add_variable(name, typ)
                    except KeyError: pass
                if len(lg_rng.variables) > 0: lgs.append(lg_rng)

                # ---- Callbacks ----
                def on_att(ts, data, _):
                    with self._lock:
                        self._state['roll_deg']  = data.get('stabilizer.roll')
                        self._state['pitch_deg'] = data.get('stabilizer.pitch')
                        self._state['yaw_deg']   = data.get('stabilizer.yaw')

                def on_gyro(ts, data, _):
                    with self._lock:
                        self._state['gyro']['x'] = data.get('gyro.x')
                        self._state['gyro']['y'] = data.get('gyro.y')
                        self._state['gyro']['z'] = data.get('gyro.z')

                def on_acc(ts, data, _):
                    with self._lock:
                        self._state['acc']['x'] = data.get('acc.x')
                        self._state['acc']['y'] = data.get('acc.y')
                        self._state['acc']['z'] = data.get('acc.z')

                def on_bat(ts, data, _):
                    with self._lock:
                        self._state['vbat'] = data.get('pm.vbat')

                def on_posvel(ts, data, _):
                    with self._lock:
                        s = self._state
                        s['pos']['x'] = data.get('stateEstimate.x')
                        s['pos']['y'] = data.get('stateEstimate.y')
                        s['pos']['z'] = data.get('stateEstimate.z')
                        s['vel']['x'] = data.get('stateEstimate.vx')
                        s['vel']['y'] = data.get('stateEstimate.vy')
                        s['vel']['z'] = data.get('stateEstimate.vz')

                def on_rng(ts, data, _):
                    with self._lock:
                        m = self._state['range']
                        if 'range.front'  in data: m['front'] = data['range.front']  / 1000.0
                        if 'range.back'   in data: m['back']  = data['range.back']   / 1000.0
                        if 'range.left'   in data: m['left']  = data['range.left']   / 1000.0
                        if 'range.right'  in data: m['right'] = data['range.right']  / 1000.0
                        if 'range.up'     in data: m['up']    = data['range.up']     / 1000.0
                        if 'range.zrange' in data: m['down']  = data['range.zrange'] / 1000.0

                def on_err(logconf, msg):
                    self.get_logger().warn(f'{logconf.name} error: {msg}')

                for lg in lgs:
                    try:
                        cf.log.add_config(lg)
                        if lg.name == 'LG_ATT':      lg.data_received_cb.add_callback(on_att)
                        elif lg.name == 'LG_GYRO':   lg.data_received_cb.add_callback(on_gyro)
                        elif lg.name == 'LG_ACC':    lg.data_received_cb.add_callback(on_acc)
                        elif lg.name == 'LG_BAT':    lg.data_received_cb.add_callback(on_bat)
                        elif lg.name == 'LG_POSVEL': lg.data_received_cb.add_callback(on_posvel)
                        elif lg.name == 'LG_RNG':    lg.data_received_cb.add_callback(on_rng)
                        lg.error_cb.add_callback(on_err)
                        lg.start()
                    except Exception as e:
                        self.get_logger().warn(f'Failed to start {lg.name}: {e}')

                # keep thread alive
                while rclpy.ok():
                    time.sleep(0.1)

                # shutdown
                try: cf.commander.send_stop_setpoint()
                except Exception: pass

        except Exception as e:
            self.get_logger().error(f'Crazyflie link error: {e}')

    # ---------- Telemetry publish ----------
    def _publish_all(self):
        now = self.get_clock().now().to_msg()
        with self._lock:
            s = {k: (v.copy() if isinstance(v, dict) else v) for k, v in self._state.items()}

        # RPY (deg)
        if all(v is not None for v in (s['roll_deg'], s['pitch_deg'], s['yaw_deg'])):
            v3 = Vector3Stamped()
            v3.header.stamp = now; v3.header.frame_id = 'base_link'
            v3.vector.x, v3.vector.y, v3.vector.z = s['roll_deg'], s['pitch_deg'], s['yaw_deg']
            self.pub_rpy.publish(v3)

        # IMU
        if None not in (s['roll_deg'], s['pitch_deg'], s['yaw_deg'],
                        s['gyro']['x'], s['gyro']['y'], s['gyro']['z'],
                        s['acc']['x'], s['acc']['y'], s['acc']['z']):
            imu = Imu()
            imu.header.stamp = now; imu.header.frame_id = 'base_link'
            rr, pp, yy = map(math.radians, (s['roll_deg'], s['pitch_deg'], s['yaw_deg']))
            qw,qx,qy,qz = rpy_to_quat(rr, pp, yy)
            imu.orientation.w, imu.orientation.x = qw, qx
            imu.orientation.y, imu.orientation.z = qy, qz
            imu.angular_velocity.x = s['gyro']['x']; imu.angular_velocity.y = s['gyro']['y']; imu.angular_velocity.z = s['gyro']['z']
            imu.linear_acceleration.x = s['acc']['x']; imu.linear_acceleration.y = s['acc']['y']; imu.linear_acceleration.z = s['acc']['z']
            self.pub_imu.publish(imu)

        # Odom
        if self.pub_odom and None not in (s['pos']['x'], s['pos']['y'], s['pos']['z'], s['vel']['x'], s['vel']['y'], s['vel']['z']):
            od = Odometry()
            od.header.stamp = now; od.header.frame_id = 'map'; od.child_frame_id = 'base_link'
            od.pose.pose.position.x = s['pos']['x']; od.pose.pose.position.y = s['pos']['y']; od.pose.pose.position.z = s['pos']['z']
            od.twist.twist.linear.x = s['vel']['x']; od.twist.twist.linear.y = s['vel']['y']; od.twist.twist.linear.z = s['vel']['z']
            self.pub_odom.publish(od)

        # Battery
        if s['vbat'] is not None:
            b = BatteryState(); b.header.stamp = now; b.voltage = float(s['vbat'])
            self.pub_batt.publish(b)

        # Ranges
        for key, pub in self.range_pubs.items():
            val = s['range'].get(key)
            if val is None: continue
            msg = Range()
            msg.header.stamp = now; msg.header.frame_id = f'range_{key}_link'
            msg.radiation_type = Range.INFRARED
            msg.min_range = 0.02; msg.max_range = 4.0; msg.field_of_view = 0.26
            msg.range = float(val)
            pub.publish(msg)

    # ---------- Control: hover ----------
    def _on_cmd_hover(self, msg: TwistStamped):
        with self._lock:
            self._last_hover = msg
            self._last_hover_time = self.get_clock().now().nanoseconds * 1e-9

    def _hover_tick(self):
        # 주기적으로 hover setpoint 전송 (최근 명령 없거나 타임아웃 시 notify만)
        if not hasattr(self, 'cf'):
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            cmd = self._last_hover
            t0  = self._last_hover_time

        if (cmd is None) or (now - t0 > self.hover_timeout_s):
            try: self.cf.commander.send_notify_setpoint_stop()
            except Exception: pass
            return

        vx = float(cmd.twist.linear.x)
        vy = float(cmd.twist.linear.y)
        z  = float(cmd.twist.linear.z)
        yawrate_deg = math.degrees(float(cmd.twist.angular.z))
        try:
            # 저수준 setpoint를 보내면 HL은 자동 비활성(필요 시 이후 HL 명령에서 다시 enable)
            self.cf.commander.send_hover_setpoint(vx, vy, yawrate_deg, z)
        except Exception as e:
            self.get_logger().warn(f'hover send failed: {e}')

    # ---------- Control: High-Level ----------
    def _enable_hl(self):
        try:
            self.cf.param.set_value('commander.enHighLevel', '1')
            time.sleep(0.05)
            self.cf.commander.send_notify_setpoint_stop()
        except Exception as e:
            self.get_logger().warn(f'Enable HL skipped/failed: {e}')

    def _on_hl_takeoff(self, msg: Float32):
        if not hasattr(self, 'cf'): return
        try:
            self._enable_hl()
            from cflib.crazyflie.high_level_commander import HighLevelCommander
            HighLevelCommander(self.cf).takeoff(float(msg.data), self.hltake_dur)
            self.get_logger().info(f'HL takeoff to {float(msg.data):.2f} m')
        except Exception as e:
            self.get_logger().error(f'HL takeoff failed: {e}')

    def _on_hl_land(self, msg: Float32):
        if not hasattr(self, 'cf'): return
        try:
            self._enable_hl()
            from cflib.crazyflie.high_level_commander import HighLevelCommander
            HighLevelCommander(self.cf).land(float(msg.data), self.hlland_dur)
            self.get_logger().info(f'HL land to {float(msg.data):.2f} m')
        except Exception as e:
            self.get_logger().error(f'HL land failed: {e}')

    def _on_hl_goto(self, msg: PoseStamped):
        if not hasattr(self, 'cf'): return
        try:
            self._enable_hl()
            x = float(msg.pose.position.x); y = float(msg.pose.position.y); z = float(msg.pose.position.z)
            qx,qy,qz,qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
            yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)) if (qw or qx or qy or qz) else 0.0
            from cflib.crazyflie.high_level_commander import HighLevelCommander
            HighLevelCommander(self.cf).go_to(x, y, z, yaw, self.hlgoto_dur, relative=False, linear=False)
            self.get_logger().info(f'HL goto ({x:.2f},{y:.2f},{z:.2f}), yaw={yaw:.2f} rad, dur={self.hlgoto_dur:.2f}s')
        except Exception as e:
            self.get_logger().error(f'HL goto failed: {e}')

    # ---------- Services ----------
    def _srv_stop_cb(self, req, res):
        try:
            self.cf.commander.send_stop_setpoint()
            res.success = True; res.message = 'STOP setpoint sent'
        except Exception as e:
            res.success = False; res.message = f'Failed: {e}'
        return res

    def _srv_notify_cb(self, req, res):
        try:
            self.cf.commander.send_notify_setpoint_stop()
            res.success = True; res.message = 'notify_setpoint_stop sent'
        except Exception as e:
            res.success = False; res.message = f'Failed: {e}'
        return res

def main():
    rclpy.init()
    node = CfBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
