#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import socket
import struct
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# Args
# ----------------------------
default_ip = "192.168.0.145"

parser = argparse.ArgumentParser(description="AI-deck JPEG streamer + YOLO overlay (PC inference)")
parser.add_argument("-n", default=f"{default_ip}", metavar="ip", help="AI-deck IP")
parser.add_argument("-p", type=int, default=5000, metavar="port", help="AI-deck port")
parser.add_argument("--save", action="store_true", help="Save annotated frames to ./stream_out/")
parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path or name")
parser.add_argument("--device", default="", help="torch device hint: '', 'cpu', 'cuda'")
parser.add_argument("--every", type=int, default=1, help="Run inference every N frames (>=1)")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
parser.add_argument("--show-original", dest="show_original", action="store_true",
                    help="Also show raw frame window")  # <= dest 지정
args = parser.parse_args()

deck_port = args.p
deck_ip = args.n

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def rx_bytes(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Socket connection lost")
        data.extend(chunk)
    return data

def draw_fps(img, fps):
    txt = f"FPS: {fps:.1f}"
    cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

print(f"Connecting to socket on {deck_ip}:{deck_port}...")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
client_socket.connect((deck_ip, deck_port))
print("Socket connected")

print(f"Loading YOLO model: {args.model}")
yolo = YOLO(args.model)
if args.device:
    try:
        yolo.to(args.device)
    except Exception as e:
        print(f"Warning: failed to move model to '{args.device}': {e}")

if args.save:
    ensure_dir("stream_out/annotated")
    ensure_dir("stream_out/raw")
    ensure_dir("stream_out/debayer")

count = 0
t0 = time.time()
last_fps_update_t = t0
fps = 0.0

try:
    while True:
        # Packet header (length, routing, function)
        packetInfoRaw = rx_bytes(client_socket, 4)
        length, routing, function = struct.unpack("<HBB", packetInfoRaw)

        # Image header
        imgHeader = rx_bytes(client_socket, length - 2)
        magic, width, height, depth, fmt, size = struct.unpack("<BHHBBI", imgHeader)
        if magic != 0xBC:
            continue

        # Image payload (chunked)
        imgStream = bytearray()
        while len(imgStream) < size:
            packetInfoRaw = rx_bytes(client_socket, 4)
            chunk_len, dst, src = struct.unpack("<HBB", packetInfoRaw)
            chunk = rx_bytes(client_socket, chunk_len - 2)
            imgStream.extend(chunk)

        count += 1
        now = time.time()
        mean_dt = (now - t0) / max(count, 1)
        instantaneous_fps = 1.0 / mean_dt if mean_dt > 0 else 0.0
        if now - last_fps_update_t > 0.2:
            fps = instantaneous_fps
            last_fps_update_t = now

        # --- Format 0: Bayer, else: JPEG ---
        if fmt == 0:
            bayer_img = np.frombuffer(imgStream, dtype=np.uint8)
            try:
                bayer_img = bayer_img.reshape((244, 324))  # 기본 데모 해상도
            except ValueError:
                bayer_img = bayer_img.reshape((height, width))
            frame_bgr = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGR)
            if args.save and (count % args.every == 0):
                cv2.imwrite(f"stream_out/debayer/img_{count:06d}.png", frame_bgr)
                cv2.imwrite(f"stream_out/raw/img_{count:06d}.png", bayer_img)
        else:
            nparr = np.frombuffer(imgStream, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue
            if args.save and (count % args.every == 0):
                cv2.imwrite(f"stream_out/raw/img_{count:06d}.jpg", frame_bgr)

        # Inference every N frames
        if args.every < 1:
            args.every = 1

        show_img = frame_bgr.copy()
        if (count % args.every) == 0:
            results = yolo.predict(show_img, conf=args.conf, verbose=False)
            annotated = results[0].plot()
            draw_fps(annotated, fps)
            show_img = annotated
            if args.save:
                cv2.imwrite(f"stream_out/annotated/img_{count:06d}.jpg", annotated)
        else:
            draw_fps(show_img, fps)

        cv2.imshow("YOLO", show_img)
        if args.show_original:  # <= 여기! 하이픈 대신 언더스코어
            cv2.imshow("Original", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except (KeyboardInterrupt, SystemExit):
    pass
finally:
    client_socket.close()
    cv2.destroyAllWindows()
    print("Closed.")
