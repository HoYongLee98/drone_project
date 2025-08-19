#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket, struct, io, contextlib, numpy as np, cv2, threading, collections, time
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

def set_tcp_keepalive(sock, enable=True, idle=10, intvl=3, cnt=5):
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1 if enable else 0)
        if hasattr(socket, 'TCP_KEEPIDLE'):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, idle)
        if hasattr(socket, 'TCP_KEEPINTVL'):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, intvl)
        if hasattr(socket, 'TCP_KEEPCNT'):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, cnt)
    except Exception:
        pass

def set_tcp_linger_rst(sock, enable=True):
    try:
        import struct as _st
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, _st.pack('ii', 1 if enable else 0, 0))
    except Exception:
        pass

class DecodeWorker:
    """JPEG 디코딩을 메인 소켓 스레드와 분리(버퍼 길이=1, 밀리면 이전 프레임 drop)."""
    def __init__(self, node, pub_raw, frame_id, enable=True):
        self.node = node
        self.pub_raw = pub_raw
        self.frame_id = frame_id
        self.enable = enable
        self.bridge = CvBridge()
        self.buf = collections.deque(maxlen=1)
        self.event = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        if enable and pub_raw is not None:
            self.th.start()

    def push(self, stamp, jpeg_bytes, w, h):
        if not self.enable or self.pub_raw is None:
            return
        self.buf.append((stamp, jpeg_bytes, w, h))
        self.event.set()

    def _run(self):
        while True:
            self.event.wait(1.0)
            while self.buf:
                stamp, jpeg, w, h = self.buf.popleft()
                try:
                    with contextlib.redirect_stderr(io.StringIO()):
                        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                    msg.header.stamp = stamp
                    msg.header.frame_id = self.frame_id
                    self.pub_raw.publish(msg)
                except Exception as e:
                    self.node.get_logger().warn(f'[DECODE] 실패: {e}')
            self.event.clear()

def main():
    rclpy.init()
    node = rclpy.create_node('ai_deck_camera_node')

    # ---- Params ----
    host = node.declare_parameter('host', '192.168.0.145').value
    port = int(node.declare_parameter('port', 5000).value)
    bind_ip = node.declare_parameter('bind_ip', '').value
    frame_id = node.declare_parameter('frame_id', 'camera_optical_frame').value
    publish_raw = bool(node.declare_parameter('publish_raw', True).value)
    drop_corrupt = bool(node.declare_parameter('drop_corrupt', True).value)
    max_jpeg = int(node.declare_parameter('max_jpeg_size', 2_000_000).value)

    # 네트워크 내구성 옵션
    connect_timeout_s = float(node.declare_parameter('connect_timeout_s', 3.0).value)
    read_timeout_s    = float(node.declare_parameter('read_timeout_s', 5.0).value)
    reconnect_backoff = float(node.declare_parameter('reconnect_backoff_s', 1.5).value)
    keepalive_idle    = int(node.declare_parameter('tcp_keepalive_idle_s', 10).value)
    keepalive_intvl   = int(node.declare_parameter('tcp_keepalive_intvl_s', 3).value)
    keepalive_cnt     = int(node.declare_parameter('tcp_keepalive_cnt', 5).value)
    decode_async      = bool(node.declare_parameter('decode_async', True).value)

    # --- FPS 리포트 주기(초) ---
    fps_period_s      = float(node.declare_parameter('fps_report_period_s', 2.0).value)
    fps_last_t = time.monotonic()
    fps_count = 0

    # ---- QoS: Sensor Data ----
    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.history = HistoryPolicy.KEEP_LAST
    qos.durability = DurabilityPolicy.VOLATILE

    pub_comp = node.create_publisher(CompressedImage, '/camera/image/compressed', qos)
    pub_raw  = node.create_publisher(Image, '/camera/image', qos) if publish_raw else None

    decoder = DecodeWorker(node, pub_raw, frame_id, enable=decode_async)

    def connect_once():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        set_tcp_keepalive(s, True, keepalive_idle, keepalive_intvl, keepalive_cnt)
        set_tcp_linger_rst(s, True)
        if bind_ip:
            try:
                s.bind((bind_ip, 0))
                node.get_logger().info(f"[NET] bind {bind_ip}")
            except Exception as e:
                node.get_logger().warn(f"[NET] bind 실패({bind_ip}): {e}")

        s.settimeout(connect_timeout_s)
        node.get_logger().info(f"[NET] connect {host}:{port} ...")
        s.connect((host, port))
        s.settimeout(read_timeout_s)
        node.get_logger().info("[NET] connected")
        return s

    def rx_bytes(s, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket closed by peer")
            buf.extend(chunk)
        return bytes(buf)

    # ---- main loop with auto-reconnect ----
    try:
        while rclpy.ok():
            try:
                s = connect_once()
                # 재연결 시 FPS 기준점 리셋(선택)
                fps_last_t = time.monotonic()
                fps_count = 0
            except Exception as e:
                node.get_logger().warn(f"[NET] connect fail({host}): {e}. retry...")
                time.sleep(reconnect_backoff)
                continue

            try:
                while rclpy.ok():
                    # ---- CPX Packet header ----
                    pkt = rx_bytes(s, 4)                    # <HBB
                    length, routing, function = struct.unpack('<HBB', pkt)
                    if length < 2 or length > 4096:
                        node.get_logger().warn(f"[NET] bad CPX length={length}, resync")
                        continue

                    # ---- Image header ----
                    hdr = rx_bytes(s, length - 2)           # <BHHBBI
                    if len(hdr) != (length - 2):
                        raise ConnectionError("short read on header")

                    magic, w, h, depth, fmt, size = struct.unpack('<BHHBBI', hdr)
                    if magic != 0xBC:
                        node.get_logger().warn(f"[NET] bad magic 0x{magic:02X}, resync")
                        continue
                    if size <= 0 or size > max(max_jpeg, w*h*max(1,depth)*2):
                        node.get_logger().warn(f"[NET] weird size={size}, skip")
                        continue

                    # ---- Receive payload in chunks ----
                    img = bytearray()
                    while len(img) < size:
                        chdr = rx_bytes(s, 4)               # <HBB
                        clen, dst, src = struct.unpack('<HBB', chdr)
                        if clen < 2 or clen > 2048:
                            raise ConnectionError("bad chunk len")
                        chunk = rx_bytes(s, clen - 2)
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
                                msg = CvBridge().cv2_to_imgmsg(bgr, encoding='bgr8')
                                msg.header.stamp = now; msg.header.frame_id = frame_id
                                pub_raw.publish(msg)
                        except Exception as e:
                            node.get_logger().warn(f"[RAW] decode 실패: {e}")
                    else:
                        # JPEG: 압축본 publish
                        comp = CompressedImage()
                        comp.format = 'jpeg'; comp.data = bytes(img)
                        comp.header.stamp = now; comp.header.frame_id = frame_id
                        pub_comp.publish(comp)

                        # 필요 시 RAW 비동기 디코드
                        if publish_raw and decode_async:
                            decoder.push(now, bytes(img), w, h)
                        elif publish_raw and not decode_async:
                            try:
                                with contextlib.redirect_stderr(io.StringIO()):
                                    frame = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                                if frame is not None:
                                    msg = CvBridge().cv2_to_imgmsg(frame, encoding='bgr8')
                                    msg.header.stamp = now; msg.header.frame_id = frame_id
                                    pub_raw.publish(msg)
                                elif not drop_corrupt:
                                    node.get_logger().warn("[JPEG] imdecode 실패, drop")
                            except Exception as e:
                                node.get_logger().warn(f"[JPEG] decode 실패: {e}")

                    # --- FPS 계산/로그 ---
                    fps_count += 1
                    now_mono = time.monotonic()
                    dt = now_mono - fps_last_t
                    if dt >= fps_period_s:
                        fps = fps_count / dt if dt > 0 else 0.0
                        node.get_logger().info(f"[FPS] {fps:.2f} Hz (window {dt:.2f}s, frames {fps_count})")
                        fps_last_t = now_mono
                        fps_count = 0

            except (socket.timeout,) as e:
                node.get_logger().warn(f"[NET] recv timeout: {e} → 재연결")
            except (ConnectionError, OSError) as e:
                node.get_logger().warn(f"[NET] 연결 끊김: {e} → 재연결")
            finally:
                try:
                    s.close()
                except Exception:
                    pass
                time.sleep(reconnect_backoff)

    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
