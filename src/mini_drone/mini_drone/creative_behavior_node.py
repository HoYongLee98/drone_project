#!/usr/bin/env python3
import math
import time
import threading
from enum import Enum, auto
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Range
from std_srvs.srv import Trigger

# 이미지 처리
try:
    from cv_bridge import CvBridge
    import cv2
except Exception:
    CvBridge = None
    cv2 = None

# YOLO (있으면 사용)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class Phase(Enum):
    IDLE = auto()
    TAKEOFF = auto()
    FORWARD1 = auto()
    DETECT = auto()
    GREET_DOWN = auto()
    GREET_UP = auto()
    AVOID = auto()
    FORWARD2 = auto()
    LAND = auto()
    DONE = auto()
    ABORT = auto()


class CreativeBehaviorNode(Node):
    """
    Bridge(센싱/컨트롤 I/O)에 의존해 고수준 행동을 조합하는 노드.
    상태머신으로 동작을 순차 실행합니다.
    """

    def __init__(self):
        super().__init__("creative_behavior")

        # ---------- Parameters ----------
        # 토픽
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("odom_topic", "/cf/odom")
        self.declare_parameter("front_range_topic", "/cf/range/front")

        # 시나리오 파라미터
        self.declare_parameter("takeoff_height_m", 0.5)
        self.declare_parameter("takeoff_timeout_s", 8.0)
        self.declare_parameter("forward_speed_mps", 0.3)
        self.declare_parameter("forward1_time_s", 3.0)
        self.declare_parameter("forward2_time_s", 3.0)
        self.declare_parameter("greet_delta_z_m", 0.15)     # 인사: 아래로 내릴 폭
        self.declare_parameter("greet_pause_s", 0.8)
        self.declare_parameter("avoid_lateral_speed_mps", 0.25)
        self.declare_parameter("avoid_time_s", 2.0)
        self.declare_parameter("hover_cmd_rate_hz", 30.0)   # hover setpoint 전송 주기
        self.declare_parameter("safety_front_min_m", 0.5)   # 전방 최소 거리

        # 탐지 파라미터
        self.declare_parameter("use_yolo", True)
        self.declare_parameter("yolo_model_path", "yolov8n.pt")
        self.declare_parameter("yolo_conf_th", 0.4)
        self.declare_parameter("hog_stride", 8)  # HOG 폴백용
        self.declare_parameter("detect_center_weight", 0.3)  # 중앙 가중(선택)
        self.declare_parameter("detect_timeout_s", 12.0)

        # 내부 상태
        p = lambda n: self.get_parameter(n).get_parameter_value()
        self.image_topic = p("image_topic").string_value
        self.odom_topic = p("odom_topic").string_value
        self.front_range_topic = p("front_range_topic").string_value

        self.alt_target = float(p("takeoff_height_m").double_value or 0.5)
        self.takeoff_timeout_s = float(p("takeoff_timeout_s").double_value or 8.0)
        self.v_forward = float(p("forward_speed_mps").double_value or 0.3)
        self.forward1_time = float(p("forward1_time_s").double_value or 3.0)
        self.forward2_time = float(p("forward2_time_s").double_value or 3.0)
        self.greet_dz = float(p("greet_delta_z_m").double_value or 0.15)
        self.greet_pause = float(p("greet_pause_s").double_value or 0.8)
        self.v_avoid = float(p("avoid_lateral_speed_mps").double_value or 0.25)
        self.avoid_time = float(p("avoid_time_s").double_value or 2.0)
        self.cmd_rate = float(p("hover_cmd_rate_hz").double_value or 30.0)
        self.safety_front_min = float(p("safety_front_min_m").double_value or 0.5)

        self.use_yolo = bool(p("use_yolo").bool_value or True)
        self.yolo_model_path = p("yolo_model_path").string_value or "yolov8n.pt"
        self.yolo_conf_th = float(p("yolo_conf_th").double_value or 0.4)
        self.hog_stride = int(p("hog_stride").integer_value or 8)
        self.detect_center_weight = float(p("detect_center_weight").double_value or 0.3)
        self.detect_timeout_s = float(p("detect_timeout_s").double_value or 12.0)

        # ---------- QoS ----------
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        # ---------- Publishers (control) ----------
        self.pub_takeoff = self.create_publisher(Float32, "/cf/hl/takeoff", qos)
        self.pub_land = self.create_publisher(Float32, "/cf/hl/land", qos)
        self.pub_goto = self.create_publisher(PoseStamped, "/cf/hl/goto", qos)
        self.pub_hover = self.create_publisher(TwistStamped, "/cf/cmd_hover", qos)

        # ---------- Service (safety/stop) ----------
        self.cli_stop = self.create_client(Trigger, "/cf/stop")

        # ---------- Subscribers (sensing) ----------
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self._on_odom, qos)
        self.sub_front = self.create_subscription(Range, self.front_range_topic, self._on_front, qos)

        # 카메라
        self.bridge = CvBridge() if CvBridge else None
        self.last_img = None
        if self.bridge is not None:
            self.sub_img = self.create_subscription(Image, self.image_topic, self._on_image, qos)
        else:
            self.get_logger().warn("cv_bridge를 불러오지 못했습니다. 카메라 입력 없이 동작합니다.")

        # ---------- Detector init ----------
        self.detector_name = "none"
        self.yolo = None
        self.hog = None
        if self.use_yolo and YOLO is not None:
            try:
                self.yolo = YOLO(self.yolo_model_path)
                self.detector_name = "yolo"
                self.get_logger().info(f"YOLO 로드 성공: {self.yolo_model_path}")
            except Exception as e:
                self.get_logger().warn(f"YOLO 로드 실패: {e}")
        if self.yolo is None and cv2 is not None:
            try:
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self.detector_name = "hog"
                self.get_logger().info("OpenCV HOG 사람검출로 폴백합니다.")
            except Exception as e:
                self.get_logger().warn(f"HOG 초기화 실패: {e}")

        # ---------- State machine ----------
        self.phase = Phase.TAKEOFF
        self.phase_t0 = self._now()
        self.odom: Optional[Odometry] = None
        self.front_m: Optional[float] = None
        self.greet_phase_done = False
        self.avoid_dir = 0  # -1: 왼쪽으로 피하기, +1: 오른쪽
        self.person_last_seen_t = None
        self.alt_current = 0.0  # 추정 고도(odom z)

        # 제어 타이머(hover 전송 포함)
        self.create_timer(1.0 / max(1.0, self.cmd_rate), self._tick)

        self.get_logger().info("Creative behavior node 시작")

    # ------------- Utils -------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, msg: Odometry):
        self.odom = msg
        self.alt_current = float(msg.pose.pose.position.z)

    def _on_front(self, msg: Range):
        self.front_m = float(msg.range)

    def _on_image(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.last_img = cv_img
        except Exception as e:
            self.get_logger().warn(f"이미지 변환 실패: {e}")

    def _publish_hover(self, vx: float, vy: float, z: float, yaw_rate_rad_s: float = 0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        # body frame: x forward, y left
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = float(z)     # hover API에서 z는 절대 고도
        msg.twist.angular.z = float(yaw_rate_rad_s)
        self.pub_hover.publish(msg)

    def _publish_takeoff(self, z: float):
        self.pub_takeoff.publish(Float32(data=float(z)))

    def _publish_land(self, z: float = 0.0):
        self.pub_land.publish(Float32(data=float(z)))

    def _publish_goto_z(self, z: float):
        """현재 x,y 유지, z만 변경하는 HL goto"""
        if self.odom is None:
            return
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = "map"
        ps.pose.position.x = float(self.odom.pose.pose.position.x)
        ps.pose.position.y = float(self.odom.pose.pose.position.y)
        ps.pose.position.z = float(z)
        # yaw 미지정 → 0으로 두면 현재 유지
        ps.pose.orientation.w = 1.0
        self.pub_goto.publish(ps)

    # ------------- Detection -------------
    def _detect_person(self) -> Tuple[bool, Optional[float]]:
        """
        이미지에서 사람 검출.
        반환: (detected, x_offset_norm)
          - x_offset_norm: 화면 중심 대비 [-1..1] (왼쪽 음수, 오른쪽 양수), 미사용 시 None
        """
        if self.last_img is None:
            return False, None
        img = self.last_img

        if self.yolo is not None:
            try:
                # YOLO: 결과에서 class==0(person)
                res = self.yolo.predict(img, verbose=False, conf=self.yolo_conf_th, imgsz=640)[0]
                found = False
                best_x = None
                W = img.shape[1]
                # 중심 가중 + 신뢰도 가중으로 가장 "중앙/신뢰 높은" 하나 선택
                score_best = -1.0
                for b in res.boxes:
                    cls = int(b.cls[0])
                    if cls != 0:
                        continue
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = map(float, b.xyxy[0])
                    cx = 0.5 * (x1 + x2)
                    # 중심 가깝게(작을수록 좋음)
                    center_off = abs((cx - W/2.0) / (W/2.0))  # 0~1
                    score = (1.0 - self.detect_center_weight) * conf + self.detect_center_weight * (1.0 - center_off)
                    if score > score_best:
                        score_best = score
                        best_x = cx
                    found = True
                if found and best_x is not None and W > 0:
                    x_off_norm = (best_x - W/2.0) / (W/2.0)
                    return True, float(x_off_norm)
                return False, None
            except Exception as e:
                self.get_logger().warn(f"YOLO 추론 실패: {e}")
                return False, None

        if self.hog is not None and cv2 is not None:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects, _ = self.hog.detectMultiScale(gray, winStride=(self.hog_stride, self.hog_stride))
                if len(rects) > 0:
                    # 가장 큰 박스를 선택
                    areas = [(w*h, (x + w/2.0)) for (x, y, w, h) in rects]
                    _, cx = max(areas, key=lambda t: t[0])
                    W = img.shape[1]
                    if W > 0:
                        x_off_norm = (cx - W/2.0) / (W/2.0)
                        return True, float(x_off_norm)
                return False, None
            except Exception as e:
                self.get_logger().warn(f"HOG 추론 실패: {e}")
                return False, None

        # 감지기 없음
        return False, None

    # ------------- Main tick (state machine + hover cmds) -------------
    def _tick(self):
        now = self._now()

        # 공통 안전: 전방 장애물
        if self.front_m is not None and self.front_m < self.safety_front_min and self.phase not in (Phase.LAND, Phase.DONE, Phase.ABORT):
            self.get_logger().warn(f"전방 장애물 감지({self.front_m:.2f} m). 비상 착륙 시도.")
            self._publish_hover(0.0, 0.0, self.alt_current, 0.0)
            self._publish_land(0.0)
            self.phase = Phase.ABORT
            self.phase_t0 = now
            return

        # 상태별 처리
        if self.phase == Phase.TAKEOFF:
            if (now - self.phase_t0) < 0.2:
                self._publish_takeoff(self.alt_target)
            # 고도 도달 확인
            if self.odom is not None and self.odom.pose.pose.position.z >= (self.alt_target - 0.05):
                self.get_logger().info("이륙 완료 → 전진")
                self.phase = Phase.FORWARD1
                self.phase_t0 = now
            elif (now - self.phase_t0) > self.takeoff_timeout_s:
                self.get_logger().warn("이륙 타임아웃. 착륙 시도")
                self._publish_land(0.0)
                self.phase = Phase.ABORT
                self.phase_t0 = now

        elif self.phase == Phase.FORWARD1:
            # hover setpoint로 전진
            self._publish_hover(self.v_forward, 0.0, self.alt_target, 0.0)
            if (now - self.phase_t0) > self.forward1_time:
                self.get_logger().info("전진1 완료 → 사람 탐지")
                self.phase = Phase.DETECT
                self.phase_t0 = now

        elif self.phase == Phase.DETECT:
            detected, xoff = self._detect_person()
            if detected:
                self.person_last_seen_t = now
                # 중앙 기준 좌/우 판단(양수: 오른쪽)
                self.avoid_dir = -1 if (xoff is not None and xoff > 0.0) else 1
                self.get_logger().info(f"사람 인식! x_offset={xoff if xoff is not None else 'NA'} → 인사")
                # HL로 z 내리기
                self._publish_goto_z(max(0.1, self.alt_target - self.greet_dz))
                self.phase = Phase.GREET_DOWN
                self.phase_t0 = now
            elif (now - self.phase_t0) > self.detect_timeout_s:
                self.get_logger().warn("사람 미탐지(타임아웃) → 착륙")
                self._publish_land(0.0)
                self.phase = Phase.LAND
                self.phase_t0 = now
            else:
                # 탐지 중에는 제자리 유지
                self._publish_hover(0.0, 0.0, self.alt_target, 0.0)

        elif self.phase == Phase.GREET_DOWN:
            # 일정 시간 대기 후 올라가기
            if (now - self.phase_t0) > self.greet_pause:
                self._publish_goto_z(self.alt_target)
                self.phase = Phase.GREET_UP
                self.phase_t0 = now

        elif self.phase == Phase.GREET_UP:
            if (now - self.phase_t0) > self.greet_pause:
                self.get_logger().info("인사 완료 → 회피 기동")
                self.phase = Phase.AVOID
                self.phase_t0 = now

        elif self.phase == Phase.AVOID:
            # 옆으로 회피
            vy = self.v_avoid * float(self.avoid_dir)  # 좌(+), 우(-)
            self._publish_hover(0.0, vy, self.alt_target, 0.0)
            if (now - self.phase_t0) > self.avoid_time:
                self.get_logger().info("회피 완료 → 전진2")
                self.phase = Phase.FORWARD2
                self.phase_t0 = now

        elif self.phase == Phase.FORWARD2:
            self._publish_hover(self.v_forward, 0.0, self.alt_target, 0.0)
            if (now - self.phase_t0) > self.forward2_time:
                self.get_logger().info("전진2 완료 → 착륙")
                self._publish_land(0.0)
                self.phase = Phase.LAND
                self.phase_t0 = now

        elif self.phase == Phase.LAND:
            # 착륙 동안 hover를 보내지 않음(HL 우선)
            if self.odom is not None and self.odom.pose.pose.position.z <= 0.05:
                self.get_logger().info("착륙 완료. 동작 종료")
                self.phase = Phase.DONE
                self.phase_t0 = now

        elif self.phase in (Phase.DONE, Phase.ABORT):
            # 멈춤 신호만 가끔 보냄
            if (now - self.phase_t0) > 2.0:
                # 필요 시 STOP 서비스 호출
                if self.cli_stop.service_is_ready():
                    try:
                        self.cli_stop.call_async(Trigger.Request())
                    except Exception:
                        pass
                self.phase_t0 = now
        else:
            pass


def main():
    rclpy.init()
    node = CreativeBehaviorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
