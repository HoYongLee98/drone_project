#!/usr/bin/env python3
import rclpy, socket, struct, numpy as np, cv2
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

class AIDeckCam(Node):
    def __init__(self):
        super().__init__('ai_deck_camera_node')
        # mode: 'socket' or 'v4l2'
        self.mode  = self.declare_parameter('mode', 'socket').get_parameter_value().string_value
        self.index = self.declare_parameter('index', 0).get_parameter_value().integer_value
        self.host  = self.declare_parameter('host', '192.168.0.145').get_parameter_value().string_value
        self.port  = self.declare_parameter('port', 5000).get_parameter_value().integer_value
        self.bind_ip = self.declare_parameter('bind_ip', '192.168.0.41').get_parameter_value().string_value

        self.pub_raw = self.create_publisher(Image, '/camera/image', 10)
        self.pub_jpg = self.create_publisher(CompressedImage, '/camera/image/compressed', 10)
        self.bridge = CvBridge()

        if self.mode == 'v4l2':
            self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise RuntimeError(f'V4L2 open failed: /dev/video{self.index}')
            self.timer = self.create_timer(1.0/30.0, self.capture_v4l2)
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if self.bind_ip:
                self.sock.bind((self.bind_ip, 0))
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(2.0)
            self.timer = self.create_timer(0.0, self.capture_socket)

    def capture_v4l2(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub_raw.publish(msg)

    def _recv_all(self, n):
        buf = b''
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError('Socket closed')
            buf += chunk
        return buf

    def capture_socket(self):
        # 4-byte little-endian length prefix + JPEG payload (wifi-img-streamer 기본)
        hdr = self._recv_all(4)
        (nbytes,) = struct.unpack('<I', hdr)
        jpeg = self._recv_all(nbytes)

        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        raw = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub_raw.publish(raw)

        comp = CompressedImage()
        comp.format = 'jpeg'
        comp.data = jpeg
        comp.header = raw.header
        self.pub_jpg.publish(comp)

def main():
    rclpy.init()
    node = AIDeckCam()
    rclpy.spin(node)

if __name__ == '__main__':
    main()

