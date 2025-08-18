#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket, struct, io, contextlib, numpy as np, cv2
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

def main():
    rclpy.init()
    node = rclpy.create_node('ai_deck_camera_node')

    # ---- Params (데모와 동일 기본값) ----
    host = node.declare_parameter('host', '192.168.0.145').value
    port = int(node.declare_parameter('port', 5000).value)
    bind_ip = node.declare_parameter('bind_ip', '').value  # 예: "192.168.0.41"
    frame_id = node.declare_parameter('frame_id', 'camera_optical_frame').value
    publish_raw = bool(node.declare_parameter('publish_raw', True).value)
    drop_corrupt = bool(node.declare_parameter('drop_corrupt', True).value)
    max_jpeg = int(node.declare_parameter('max_jpeg_size', 2_000_000).value)

    # ---- QoS: Sensor Data ----
    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.history = HistoryPolicy.KEEP_LAST
    qos.durability = DurabilityPolicy.VOLATILE

    pub_comp = node.create_publisher(CompressedImage, '/camera/image/compressed', qos)
    pub_raw  = node.create_publisher(Image, '/camera/image', qos) if publish_raw else None
    bridge = CvBridge()

    # ---- Socket (데모와 동일한 블로킹 수신) ----
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if bind_ip:
        s.bind((bind_ip, 0))
        node.get_logger().info(f"[NET] bind {bind_ip}")
    node.get_logger().info(f"[NET] connect {host}:{port} ...")
    s.connect((host, port))
    node.get_logger().info("[NET] connected")

    def rx_bytes(n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket closed")
            buf.extend(chunk)
        return bytes(buf)

    try:
        while rclpy.ok():
            # ---- Packet header ----
            pkt = rx_bytes(4)                           # <HBB
            length, routing, function = struct.unpack('<HBB', pkt)
            # ---- Image header ----
            hdr = rx_bytes(length - 2)                  # <BHHBBI
            magic, w, h, depth, fmt, size = struct.unpack('<BHHBBI', hdr)
            if magic != 0xBC:
                node.get_logger().warn(f"Bad magic 0x{magic:02X}, resync")
                continue
            if size <= 0 or size > max(max_jpeg, w*h*max(1,depth)*2):
                node.get_logger().warn(f"Weird size={size}, skip")
                # 남은 스트림은 다음 루프에서 재동기화
                continue

            # ---- Receive payload in chunks ----
            img = bytearray()
            while len(img) < size:
                chdr = rx_bytes(4)                       # <HBB
                clen, dst, src = struct.unpack('<HBB', chdr)
                chunk = rx_bytes(clen - 2)
                need = size - len(img)
                img.extend(chunk[:need])

            now = node.get_clock().now().to_msg()

            if fmt == 0:
                # RAW Bayer -> BGR
                try:
                    arr = np.frombuffer(img, np.uint8).reshape((h, w))
                    with contextlib.redirect_stderr(io.StringIO()):
                        bgr = cv2.cvtColor(arr, cv2.COLOR_BayerBG2BGR)
                    if pub_raw:
                        msg = bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
                        msg.header.stamp = now; msg.header.frame_id = frame_id
                        pub_raw.publish(msg)
                    # RAW는 압축본이 없으므로 원하면 아래처럼 만들 수 있음:
                    # ok, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    # if ok: ... pub_comp.publish(...)
                except Exception:
                    node.get_logger().warn("RAW decode failed, drop")
            else:
                # JPEG: 압축 그대로 퍼블리시 (+ 옵션으로 디코딩)
                comp = CompressedImage()
                comp.format = 'jpeg'; comp.data = bytes(img)
                comp.header.stamp = now; comp.header.frame_id = frame_id
                pub_comp.publish(comp)

                if pub_raw:
                    with contextlib.redirect_stderr(io.StringIO()):
                        frame = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                        msg.header.stamp = now; msg.header.frame_id = frame_id
                        pub_raw.publish(msg)
                    elif not drop_corrupt:
                        node.get_logger().warn("imdecode failed, drop")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            s.close()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
