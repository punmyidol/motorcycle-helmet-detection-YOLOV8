"""
send_test.py — send a single image to the cloud inference server and print the result.

Usage:
    python send_test.py <image_path> [--ip 127.0.0.1] [--port 5555]

Example:
    python send_test.py test.jpg
    python send_test.py test.jpg --ip 192.168.1.100 --port 5555
"""

import argparse
import sys
import cv2
import zmq


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a test image to the cloud inference server")
    parser.add_argument("image",        help="Path to the image file to send")
    parser.add_argument("--ip",   default="127.0.0.1", help="Cloud server IP (default: 127.0.0.1)")
    parser.add_argument("--port", default=5555, type=int, help="Inference port (default: 5555)")
    args = parser.parse_args()

    # ── LOAD & ENCODE IMAGE ───────────────────────────────────────────────────
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[ERROR] Could not open image: {args.image}")
        sys.exit(1)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        print("[ERROR] Failed to JPEG-encode image")
        sys.exit(1)

    jpeg_bytes = buf.tobytes()
    print(f"[INFO] Loaded '{args.image}' — {frame.shape[1]}x{frame.shape[0]}, {len(jpeg_bytes)} bytes after encoding")

    # ── CONNECT ───────────────────────────────────────────────────────────────
    endpoint = f"tcp://{args.ip}:{args.port}"
    print(f"[INFO] Connecting to {endpoint} ...")

    ctx    = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.setsockopt(zmq.SNDTIMEO, 5_000)
    socket.setsockopt(zmq.RCVTIMEO, 10_000)
    socket.setsockopt(zmq.LINGER,   0)
    socket.connect(endpoint)

    # ── SEND ──────────────────────────────────────────────────────────────────
    print("[INFO] Sending frame...")
    LABEL_COLOURS = {"helmet": "\033[92m", "no-helmet": "\033[91m", "unknown": "\033[93m"}
    RESET = "\033[0m"

    try:
        socket.send(jpeg_bytes)
        label = socket.recv_string()
        colour = LABEL_COLOURS.get(label, "")
        print(f"[OK]   Detection result: {colour}{label}{RESET}")
    except zmq.Again:
        print("[ERROR] Timed out — is the cloud server running and reachable?")
        sys.exit(1)
    finally:
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()