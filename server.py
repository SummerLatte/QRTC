"""
QRTC 符号层实地测试 HTTP 服务。

工作流：
1. 电脑打开页面 → 生成测试码图 → 显示在屏幕上
2. 手机打开页面 → 摄像头对准电脑屏幕拍摄
3. 手机将照片回传 → 服务端用 OpenCVSampler 解码 → 返回匹配结果

用法：
    py -3 server.py [--port 8000] [--host 0.0.0.0]
"""

from __future__ import annotations

import argparse
import base64
import datetime
import io
import json
import os
import random
import socket
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from PIL import Image

from symbol_layer import (
    ColorPalette, ShapeSet, Grid, SymbolEncoder,
    FrameRenderer, OpenCVSampler, SymbolDecoder,
)

# ---------------------------------------------------------------------------
# 全局状态：当前测试
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_current_test: dict | None = None
# {
#   "original_block": list[int],
#   "level": int, "n_colors": int, "n_shapes": int, "module_size": int,
#   "S": int, "M": int,
#   "image_b64": str,       # base64 PNG
#   "timestamp": float,
# }

# ---------------------------------------------------------------------------
# 编解码逻辑
# ---------------------------------------------------------------------------

DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")


def generate_test_image(
    level: int = 1,
    n_colors: int = 4,
    n_shapes: int = 4,
    module_size: int = 12,
    seed: int = 42,
) -> dict:
    """生成一帧测试码图，返回元数据 + base64 PNG。"""
    g = Grid(level)
    pal = ColorPalette(n_colors)
    shp = ShapeSet(n_shapes)
    enc = SymbolEncoder(pal, shp, g)
    M = enc.M
    S = enc.S

    rng = random.Random(seed)
    original_block = [rng.randint(0, M - 1) for _ in range(S)]

    frame = enc.encode(original_block)
    renderer = FrameRenderer(module_size=module_size, quiet_zone_size=4)
    img = renderer.render(frame)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # 保存原始符号块到文件
    os.makedirs(DEBUG_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"L{level}_C{n_colors}_S{n_shapes}_ms{module_size}"
    truth_path = os.path.join(DEBUG_DIR, f"{ts}_{tag}_seed{seed}_truth.json")
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "level": level,
            "n_colors": n_colors,
            "n_shapes": n_shapes,
            "module_size": module_size,
            "S": S,
            "M": M,
            "original_block": original_block,
            "timestamp": time.time(),
        }, f, ensure_ascii=False, indent=2)

    return {
        "original_block": original_block,
        "level": level,
        "n_colors": n_colors,
        "n_shapes": n_shapes,
        "module_size": module_size,
        "S": S,
        "M": M,
        "seed": seed,
        "image_b64": img_b64,
        "image_size": img.size,
        "timestamp": time.time(),
        "truth_file": os.path.basename(truth_path),
    }


def _save_debug_images(img: Image.Image, sampler, decoded, original: list, test: dict, mismatches: int) -> dict:
    """保存调试图像：原始回传、各阶段对齐图、不匹配标注。返回文件名列表。"""
    import cv2
    import numpy as np

    os.makedirs(DEBUG_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"L{test['level']}_C{test['n_colors']}_S{test['n_shapes']}_ms{test['module_size']}"
    prefix = f"{ts}_{tag}_err{mismatches}"

    result = {"debug_dir": DEBUG_DIR}

    # 0. 原始回传图像
    raw_path = os.path.join(DEBUG_DIR, f"{prefix}_0_raw.jpg")
    img.convert("RGB").save(raw_path, "JPEG", quality=95)
    result["debug_raw"] = os.path.basename(raw_path)

    # 保存 sampler 内部各阶段调试图
    if hasattr(sampler, "_debug_images") and sampler._debug_images:
        for stage_name, stage_img in sampler._debug_images.items():
            fname = f"{prefix}_{stage_name}.png"
            fpath = os.path.join(DEBUG_DIR, fname)
            cv2.imwrite(fpath, stage_img)
            result[f"debug_{stage_name}"] = fname

    # 最终不匹配标注图
    annotate_path = os.path.join(DEBUG_DIR, f"{prefix}_annotate.png")
    annotated = sampler._warped.copy()
    ms = sampler._module_size
    qz = sampler._quiet_zone
    g = Grid(test["level"])
    data_coords = g.data_scan_order

    for i, coord in enumerate(data_coords):
        if i >= len(decoded.symbol_block) or i >= len(original):
            break
        if decoded.symbol_block[i] != original[i]:
            px = (coord.col + qz) * ms
            py = (coord.row + qz) * ms
            cv2.rectangle(annotated, (px, py), (px + ms, py + ms), (0, 0, 255), 1)
        elif float(decoded.confidences[i]) < 0.5:
            px = (coord.col + qz) * ms
            py = (coord.row + qz) * ms
            cv2.rectangle(annotated, (px, py), (px + ms, py + ms), (0, 255, 255), 1)

    cv2.imwrite(annotate_path, annotated)
    result["debug_annotate"] = os.path.basename(annotate_path)

    return result


def decode_captured_image(img_bytes: bytes, test: dict) -> dict:
    """用 OpenCVSampler 解码手机回传的图像，与原始符号块对比。"""
    img = Image.open(io.BytesIO(img_bytes))

    g = Grid(test["level"])
    pal = ColorPalette(test["n_colors"])
    shp = ShapeSet(test["n_shapes"])

    try:
        sampler = OpenCVSampler(image=img, grid=g, palette=pal, shape_set=shp, debug=True)
        dec = SymbolDecoder(sampler, g)
        decoded = dec.decode()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        os.makedirs(DEBUG_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fail_path = os.path.join(DEBUG_DIR, f"{ts}_FAIL_raw.jpg")
        img.convert("RGB").save(fail_path, "JPEG", quality=95)
        # 如果 sampler 已部分初始化，保存已有的调试图
        debug_files = {}
        if 'sampler' in dir() and hasattr(sampler, '_debug_images') and sampler._debug_images:
            import cv2
            for stage_name, stage_img in sampler._debug_images.items():
                fname = f"{ts}_FAIL_{stage_name}.png"
                fpath = os.path.join(DEBUG_DIR, fname)
                cv2.imwrite(fpath, stage_img)
                debug_files[f"debug_{stage_name}"] = fname
        return {
            "success": False,
            "error": f"解码失败: {e}",
            "original_block": test["original_block"],
            "debug_raw": os.path.basename(fail_path),
            "debug_dir": DEBUG_DIR,
            "traceback": tb,
            **debug_files,
        }

    original = test["original_block"]
    decoded_block = decoded.symbol_block
    S = len(original)
    mismatches = sum(1 for a, b in zip(decoded_block, original) if a != b)

    avg_conf = sum(float(c) for c in decoded.confidences) / len(decoded.confidences) if decoded.confidences else 0.0
    low_conf_count = sum(1 for c in decoded.confidences if float(c) < 0.5)

    # 保存调试图像
    debug_info = _save_debug_images(img, sampler, decoded, original, test, mismatches)

    return {
        "success": True,
        "decoded_block": [int(v) for v in decoded_block],
        "original_block": [int(v) for v in original],
        "mismatches": int(mismatches),
        "total": int(S),
        "error_rate": float(mismatches) / float(S),
        "avg_confidence": float(avg_conf),
        "low_confidence_count": int(low_conf_count),
        "format_info": {
            "color_level_code": int(decoded.format_info.color_level_code),
            "shape_level_code": int(decoded.format_info.shape_level_code),
        },
        **debug_info,
    }


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # 简洁日志
        ts = time.strftime("%H:%M:%S")
        sys.stdout.write(f"[{ts}] {self.address_string()} {fmt % args}\n")
        sys.stdout.flush()

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, code: int, text: str):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _png(self, img_bytes: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(img_bytes)))
        self.end_headers()
        self.wfile.write(img_bytes)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            try:
                with open(_HTML_PATH, "r", encoding="utf-8") as f:
                    self._html(200, f.read())
            except FileNotFoundError:
                self._html(404, "<h1>templates/index.html not found</h1>")
            return

        if path == "/api/generate":
            level = int(qs.get("level", ["1"])[0])
            n_colors = int(qs.get("n_colors", ["4"])[0])
            n_shapes = int(qs.get("n_shapes", ["4"])[0])
            module_size = int(qs.get("module_size", ["12"])[0])
            seed_str = qs.get("seed", ["42"])[0]
            seed = int(seed_str) if seed_str else 42

            try:
                test = generate_test_image(level, n_colors, n_shapes, module_size, seed)
            except Exception as e:
                self._json(400, {"error": str(e)})
                return

            global _current_test
            with _lock:
                _current_test = test

            self._json(200, {
                "image": test["image_b64"],
                "image_size": test["image_size"],
                "level": test["level"],
                "n_colors": test["n_colors"],
                "n_shapes": test["n_shapes"],
                "module_size": test["module_size"],
                "S": test["S"],
                "M": test["M"],
                "seed": test["seed"],
                "timestamp": test["timestamp"],
            })
            return

        if path == "/api/status":
            with _lock:
                t = _current_test
            if t is None:
                self._json(200, {"has_test": False})
            else:
                self._json(200, {
                    "has_test": True,
                    "level": t["level"],
                    "n_colors": t["n_colors"],
                    "n_shapes": t["n_shapes"],
                    "module_size": t["module_size"],
                    "S": t["S"],
                    "M": t["M"],
                    "timestamp": t["timestamp"],
                })
            return

        if path == "/api/image.png":
            with _lock:
                t = _current_test
            if t is None:
                self.send_error(404, "No test image generated yet")
                return
            img_bytes = base64.b64decode(t["image_b64"])
            self._png(img_bytes)
            return

        if path.startswith("/api/debug/"):
            filename = path[len("/api/debug/"):]
            # 防止路径遍历
            filename = os.path.basename(filename)
            if not filename:
                self.send_error(400, "Missing filename")
                return
            filepath = os.path.join(DEBUG_DIR, filename)
            if not os.path.exists(filepath):
                self.send_error(404, "Debug image not found")
                return
            ext = os.path.splitext(filename)[1].lower()
            ctype = "image/png" if ext == ".png" else "image/jpeg"
            with open(filepath, "rb") as f:
                file_bytes = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(file_bytes)))
            self.end_headers()
            self.wfile.write(file_bytes)
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/decode":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body)
                img_b64 = data.get("image", "")
                # 去掉 data:image/... 前缀
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(img_b64)
            except Exception as e:
                self._json(400, {"error": f"请求解析失败: {e}"})
                return

            with _lock:
                test = _current_test

            if test is None:
                self._json(400, {"error": "尚未生成测试图像，请先生成"})
                return

            try:
                t0 = time.time()
                result = decode_captured_image(img_bytes, test)
                elapsed = time.time() - t0
                result["decode_time_ms"] = round(elapsed * 1000, 1)
                self._json(200, result)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[ERROR] decode failed:\n{tb}", file=sys.stderr, flush=True)
                self._json(500, {"success": False, "error": f"服务端解码异常: {e}", "traceback": tb})
            return

        self.send_error(404, "Not Found")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


CERT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cert")


def ensure_self_signed_cert() -> tuple[str, str] | None:
    """确保自签名证书存在，返回 (cert_path, key_path) 或 None。"""
    cert_path = os.path.join(CERT_DIR, "cert.pem")
    key_path = os.path.join(CERT_DIR, "key.pem")
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path

    os.makedirs(CERT_DIR, exist_ok=True)
    print("  正在生成自签名证书...")
    try:
        subprocess.run([
            sys.executable, "-c",
            f"""
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime

key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "QRTC"),
    x509.NameAttribute(NameOID.COMMON_NAME, "qrtc-local-test"),
])
cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(datetime.datetime.utcnow())
    .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
    .add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(__import__('ipaddress').ip_address("127.0.0.1")),
            x509.IPAddress(__import__('ipaddress').ip_address("0.0.0.0")),
        ]),
        critical=False,
    )
    .sign(key, hashes.SHA256())
)
with open(r"{key_path}", "wb") as f:
    f.write(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
with open(r"{cert_path}", "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))
"""
        ], check=True, capture_output=True, text=True)
        print("  证书生成成功")
        return cert_path, key_path
    except subprocess.CalledProcessError as e:
        print(f"  证书生成失败: {e.stderr}")
        print("  请安装 cryptography: py -3 -m pip install cryptography")
        return None
    except Exception as e:
        print(f"  证书生成失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="QRTC 符号层实地测试服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="监听端口 (默认 8000)")
    parser.add_argument("--no-https", action="store_true", help="禁用 HTTPS (摄像头将无法在手机上使用)")
    args = parser.parse_args()

    ip = get_local_ip()
    use_https = not args.no_https
    cert_info = None

    if use_https:
        cert_info = ensure_self_signed_cert()
        if cert_info is None:
            print("\n  [警告] HTTPS 证书不可用，回退到 HTTP 模式")
            print("  [警告] 手机 Safari 摄像头需要 HTTPS，请安装 cryptography 后重试")
            use_https = False

    server = ThreadingHTTPServer((args.host, args.port), Handler)

    if use_https and cert_info:
        cert_path, key_path = cert_info
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert_path, key_path)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)

    protocol = "https" if use_https else "http"

    print("=" * 60)
    print("  QRTC 符号层实地测试服务")
    print("=" * 60)
    print(f"  本机 IP: {ip}")
    print(f"  协议:   {protocol.upper()}")
    print(f"  监听:   {protocol}://{args.host}:{args.port}")
    print()
    print(f"  电脑访问: {protocol}://localhost:{args.port}")
    print(f"  手机访问: {protocol}://{ip}:{args.port}")
    if use_https:
        print()
        print("  [重要] 首次在手机 Safari 打开时会有证书警告")
        print("  点击「显示详情」→「访问此网站」→「访问」即可")
    print()
    print("  流程:")
    print("    1. 电脑打开页面 → 生成码图 → 显示在屏幕")
    print("    2. 手机打开页面 → 摄像头对准屏幕 → 拍摄")
    print("    3. 服务端解码 → 显示匹配结果")
    print("=" * 60)
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.shutdown()


if __name__ == "__main__":
    main()
