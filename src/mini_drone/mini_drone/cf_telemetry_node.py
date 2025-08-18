#!/usr/bin/env python3
import threading
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu, Range, BatteryState
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

# Crazyflie lib
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

class CFTelemetryNode(Node):
    def __init__(self):
        super().__init__('cf_telemetry_node')

        # ---- Params ----
        self.declare_parameter('uri', 'radio://0/80/2M/E7E7E7E7E7')
        self.declare_parameter('period_ms', 100)          # cflib log period
        self.declare_parameter('publish_rate_hz', 20.0)   # ROS publish
        self.declare_parameter('use_state_estimate', True)

        self.uri = self.get_parameter('uri').get_parameter_value().string_value
        self.period_ms = int(self.get_parameter('period_ms').get_parameter_value().integer_value or 100)
        self.pub_rate = float(self.get_parameter('publish_rate_hz').get_parameter_value().double_value or 20.0)
        self.use_state = bool(self.get_parameter('use_state_estimate').get_parameter_value().bool_value or True)

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

        # ---- State (updated by cflib callbacks) ----
        self.lock = threading.Lock()
        self.state = {
            'roll_deg': None, 'pitch_deg': None, 'yaw_deg': None,
            'gyro': {'x': None, 'y': None, 'z': None},  # rad/s
            'acc':  {'x': None, 'y': None, 'z': None},  # m/s^2 (gravity-including)
            'pos':  {'x': None, 'y': None, 'z': None},  # m
            'vel':  {'x': None, 'y': None, 'z': None},  # m/s
            'range': {'front': None, 'back': None, 'left': None, 'right': None, 'up': None, 'down': None},  # m
            'vbat': None
        }

        # ---- Start Crazyflie link in background thread ----
        self.cf_thread = threading.Thread(target=self._cf_worker, daemon=True)
        self.cf_thread.start()

        # ---- ROS timer for publish ----
        self.timer = self.create_timer(1.0/self.pub_rate, self._publish_all)

    # ---------- Crazyflie worker ----------
    def _cf_worker(self):
        cflib.crtp.init_drivers(enable_debug_driver=False)
        try:
            with SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cache')) as scf:
                cf = scf.cf
                self.get_logger().info('Connected to Crazyflie')

                # ---- LogConfigs: split under 26 bytes ----
                period = self.period_ms

                lg_att = LogConfig(name='LG_ATT', period_in_ms=period)
                lg_att.add_variable('stabilizer.roll', 'float')
                lg_att.add_variable('stabilizer.pitch', 'float')
                lg_att.add_variable('stabilizer.yaw', 'float')

                lg_gyro = LogConfig(name='LG_GYRO', period_in_ms=period)
                lg_gyro.add_variable('gyro.x', 'float')
                lg_gyro.add_variable('gyro.y', 'float')
                lg_gyro.add_variable('gyro.z', 'float')

                lg_acc = LogConfig(name='LG_ACC', period_in_ms=period)
                lg_acc.add_variable('acc.x', 'float')
                lg_acc.add_variable('acc.y', 'float')
                lg_acc.add_variable('acc.z', 'float')

                lg_bat = LogConfig(name='LG_BAT', period_in_ms=period)
                lg_bat.add_variable('pm.vbat', 'float')

                logconfs = [lg_att, lg_gyro, lg_acc, lg_bat]

                if self.use_state:
                    lg_posvel = LogConfig(name='LG_POSVEL', period_in_ms=period)
                    for v in ['stateEstimate.x','stateEstimate.y','stateEstimate.z',
                              'stateEstimate.vx','stateEstimate.vy','stateEstimate.vz']:
                        lg_posvel.add_variable(v, 'float')
                    logconfs.append(lg_posvel)

                # Rangers: only add if exists (try/except)
                lg_rng = LogConfig(name='LG_RNG', period_in_ms=period)
                for name, typ in [
                    ('range.front','uint16_t'), ('range.back','uint16_t'),
                    ('range.left','uint16_t'), ('range.right','uint16_t'),
                    ('range.up','uint16_t'), ('range.zrange','uint16_t')]:
                    try:
                        lg_rng.add_variable(name, typ)
                    except KeyError:
                        pass
                # If anything added, include
                if len(lg_rng.variables) > 0:
                    logconfs.append(lg_rng)

                # ---- Callbacks ----
                def on_att(ts, data, _):
                    with self.lock:
                        self.state['roll_deg']  = data.get('stabilizer.roll')
                        self.state['pitch_deg'] = data.get('stabilizer.pitch')
                        self.state['yaw_deg']   = data.get('stabilizer.yaw')

                def on_gyro(ts, data, _):
                    with self.lock:
                        self.state['gyro']['x'] = data.get('gyro.x')
                        self.state['gyro']['y'] = data.get('gyro.y')
                        self.state['gyro']['z'] = data.get('gyro.z')

                def on_acc(ts, data, _):
                    with self.lock:
                        self.state['acc']['x'] = data.get('acc.x')
                        self.state['acc']['y'] = data.get('acc.y')
                        self.state['acc']['z'] = data.get('acc.z')

                def on_bat(ts, data, _):
                    with self.lock:
                        self.state['vbat'] = data.get('pm.vbat')

                def on_posvel(ts, data, _):
                    with self.lock:
                        self.state['pos']['x'] = data.get('stateEstimate.x')
                        self.state['pos']['y'] = data.get('stateEstimate.y')
                        self.state['pos']['z'] = data.get('stateEstimate.z')
                        self.state['vel']['x'] = data.get('stateEstimate.vx')
                        self.state['vel']['y'] = data.get('stateEstimate.vy')
                        self.state['vel']['z'] = data.get('stateEstimate.vz')

                def on_rng(ts, data, _):
                    with self.lock:
                        # mm -> m
                        m = self.state['range']
                        if 'range.front' in data:  m['front'] = data['range.front']  / 1000.0
                        if 'range.back'  in data:  m['back']  = data['range.back']   / 1000.0
                        if 'range.left'  in data:  m['left']  = data['range.left']   / 1000.0
                        if 'range.right' in data:  m['right'] = data['range.right']  / 1000.0
                        if 'range.up'    in data:  m['up']    = data['range.up']     / 1000.0
                        if 'range.zrange'in data:  m['down']  = data['range.zrange'] / 1000.0

                def on_err(logconf, msg):
                    self.get_logger().warn(f'{logconf.name} error: {msg}')

                # ---- Add/start logs ----
                for lg in logconfs:
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

                # Keep the worker alive
                while rclpy.ok():
                    time.sleep(0.1)

                # Stop logs on shutdown
                for lg in cf.log.logconfs:
                    try: lg.stop()
                    except: pass

        except Exception as e:
            self.get_logger().error(f'Crazyflie link error: {e}')

    # ---------- ROS publish ----------
    def _publish_all(self):
        now = self.get_clock().now().to_msg()

        with self.lock:
            s = dict(self.state)  # shallow copy
            roll_deg, pitch_deg, yaw_deg = s['roll_deg'], s['pitch_deg'], s['yaw_deg']
            gyro = s['gyro']; acc = s['acc']; pos = s['pos']; vel = s['vel']
            rng = s['range']; vbat = s['vbat']

        # RPY (deg)
        if all(v is not None for v in (roll_deg, pitch_deg, yaw_deg)):
            v3 = Vector3Stamped()
            v3.header.stamp = now; v3.header.frame_id = 'base_link'
            v3.vector.x, v3.vector.y, v3.vector.z = roll_deg, pitch_deg, yaw_deg
            self.pub_rpy.publish(v3)

        # IMU
        if None not in (roll_deg, pitch_deg, yaw_deg,
                        gyro['x'], gyro['y'], gyro['z'],
                        acc['x'], acc['y'], acc['z']):
            imu = Imu()
            imu.header.stamp = now; imu.header.frame_id = 'base_link'
            # orientation: deg->rad
            rr, pp, yy = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
            qw,qx,qy,qz = rpy_to_quat(rr, pp, yy)
            imu.orientation.w, imu.orientation.x = qw, qx
            imu.orientation.y, imu.orientation.z = qy, qz
            imu.angular_velocity.x = gyro['x']
            imu.angular_velocity.y = gyro['y']
            imu.angular_velocity.z = gyro['z']
            imu.linear_acceleration.x = acc['x']
            imu.linear_acceleration.y = acc['y']
            imu.linear_acceleration.z = acc['z']
            self.pub_imu.publish(imu)

        # Odom
        if self.use_state and None not in (pos['x'],pos['y'],pos['z'], vel['x'],vel['y'],vel['z']):
            odom = Odometry()
            odom.header.stamp = now
            odom.header.frame_id = 'map'
            odom.child_frame_id = 'base_link'
            odom.pose.pose.position.x = pos['x']
            odom.pose.pose.position.y = pos['y']
            odom.pose.pose.position.z = pos['z']
            odom.twist.twist.linear.x = vel['x']
            odom.twist.twist.linear.y = vel['y']
            odom.twist.twist.linear.z = vel['z']
            self.pub_odom.publish(odom)

        # Battery
        if vbat is not None:
            b = BatteryState()
            b.header.stamp = now
            b.voltage = float(vbat)
            self.pub_batt.publish(b)

        # Ranges
        for key, pub in self.range_pubs.items():
            val = rng.get(key)
            if val is None:
                continue
            msg = Range()
            msg.header.stamp = now
            msg.header.frame_id = f'range_{key}_link'
            msg.radiation_type = Range.INFRARED  # multi-ranger는 ToF (적외선/레이저)
            msg.min_range = 0.02  # m (센서에 맞게 조정)
            msg.max_range = 4.0   # m (환경에 맞게 조정)
            msg.field_of_view = 0.26  # 대략적 (rad)
            msg.range = float(val)    # m
            pub.publish(msg)

def main():
    rclpy.init()
    node = CFTelemetryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

