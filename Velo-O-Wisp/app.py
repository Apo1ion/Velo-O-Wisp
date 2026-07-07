from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
BICYCLE_CLASS_ID = 1  # COCO class id for bicycle in YOLOv8 pretrained models

app = Flask(__name__, static_folder="static")
model = YOLO("yolov8n.pt")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/style.css")
def css():
    return send_from_directory(BASE_DIR, "style.css")


@app.route("/assets/<path:filename>")
def assets(filename: str):
    return send_from_directory(BASE_DIR / "assets", filename)


@app.route("/process", methods=["POST"])
def process_image():
    started = time.perf_counter()

    if "image" not in request.files:
        return jsonify({"error": "Файл изображения не передан"}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": "Файл не выбран"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Поддерживаются только JPG, JPEG и PNG"}), 400

    original_name = file.filename
    ext = original_name.rsplit(".", 1)[1].lower()
    uid = uuid.uuid4().hex
    original_filename = f"original_{uid}.{ext}"
    result_filename = f"result_{uid}.jpg"

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Не удалось прочитать изображение"}), 400

    original_path = UPLOAD_DIR / original_filename
    result_path = RESULT_DIR / result_filename
    cv2.imwrite(str(original_path), image)

    results = model(image, verbose=False)
    output = image.copy()
    bicycle_count = 0

    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            if cls_id != BICYCLE_CLASS_ID:
                continue

            bicycle_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 3)
            label = f"bicycle {conf:.2f}"
            cv2.putText(
                output,
                label,
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(result_path), output)
    processing_time = round(time.perf_counter() - started, 2)

    return jsonify({
        "count": bicycle_count,
        "processing_time": processing_time,
        "original_filename": original_name,
        "original_image": f"/static/uploads/{original_filename}",
        "result_image": f"/static/results/{result_filename}",
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
