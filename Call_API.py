from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import os
import base64
from io import BytesIO
import uuid
import sqlite3
import json

app = Flask(__name__)
MODEL_PATH = "yolov8n.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Mô hình {MODEL_PATH} không tồn tại!")
model = YOLO(MODEL_PATH)

STATIC_DIR = "static_img"
os.makedirs(STATIC_DIR, exist_ok=True)

def get_db_connection():
    conn = sqlite3.connect("data_images.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/detect/image/", methods=["POST"])
def detect_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không tìm thấy file trong request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "File rỗng"}), 400
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Không thể đọc file ảnh"}), 400

        results = model(image)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = r.names[int(box.cls)]
                confidence = float(box.conf)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                detections.append({"label": label, "confidence": confidence, "box": [x1, y1, x2, y2]})

        _, buffer = cv2.imencode(".jpg", image)
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        return jsonify({
            "image_data": image_base64,
            "detections": detections,
            "message": "Nhận diện thành công"
        }), 200 

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/detect/video/", methods=["POST"])
def detect_video():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không tìm thấy file trong request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "File rỗng"}), 400

        temp_video_path = os.path.join(STATIC_DIR, f"temp_{uuid.uuid4().hex}.mp4")
        file.save(temp_video_path)

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            os.remove(temp_video_path)
            return jsonify({"error": "Không thể mở file video"}), 400

        ret, frame = cap.read()
        if not ret:
            cap.release()
            os.remove(temp_video_path)
            return jsonify({"error": "Không thể đọc frame từ video"}), 400

        results = model(frame)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = r.names[int(box.cls)]
                confidence = float(box.conf)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                detections.append({"label": label, "confidence": confidence, "box": [x1, y1, x2, y2]})

        _, buffer = cv2.imencode(".jpg", frame)
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        cap.release()
        os.remove(temp_video_path)

        return jsonify({
            "image_data": image_base64,
            "detections": detections,
            "message": "Nhận diện frame đầu tiên thành công"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/save_image/", methods=["POST"])
def save_image():
    try:
        data = request.get_json()
        file_path = data.get("file_path")
        detections = data.get("detections")

        if not file_path or not detections:
            return jsonify({"error": "Thiếu file_path hoặc detections"}), 400
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO images (file_path, detections) VALUES (?, ?)",
                       (file_path, detections))
        conn.commit()
        image_id = cursor.lastrowid
        conn.close()

        return jsonify({
            "image_id": image_id,
            "message": f"Đã lưu ảnh với ID: {image_id}"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/images/", methods=["GET"])
def get_images():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images")
        rows = cursor.fetchall()
        conn.close()

        # Chuyển dữ liệu thành danh sách JSON
        images = []
        for row in rows:
            images.append({
                "id": row["id"],
                "file_path": row["file_path"],
                "detections": json.loads(row["detections"]) if row["detections"] else [],
                "notes": row["notes"]
            })

        return jsonify({
            "images": images,
            "message": "Lấy danh sách ảnh thành công"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# lấy ảnh theo id
@app.route("/images/<int:id>", methods=["GET"])
def get_image_id(id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images WHERE id = ?", (id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Không tìm thấy ảnh với ID này"}), 404

        image_data = {
            "id": row["id"],
            "file_path": row["file_path"],
            "detections": json.loads(row["detections"]) if row["detections"] else [],
            "notes": row["notes"]
        }

        return jsonify({
            "image": image_data,
            "message": f"Lấy ảnh với ID: {id} thành công"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# xóa ảnh theo id
@app.route("/images/<int:id>", methods=["DELETE"])
def delete_image(id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM images WHERE id = ?", (id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return jsonify({"error": "Không tìm thấy ảnh với ID này"}), 404

        file_path = row["file_path"]
        # Xóa file ảnh trong thư mục static
        if os.path.exists(file_path):
            os.remove(file_path)

        # Xóa bản ghi trong cơ sở dữ liệu
        cursor.execute("DELETE FROM images WHERE id = ?", (id,))
        conn.commit()
        conn.close()

        return jsonify({"message": f"Đã xóa ảnh với ID: {id}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/images/<int:id>", methods=["PUT"])
def update_image_notes(id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images WHERE id = ?", (id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return jsonify({"error": "Không tìm thấy ảnh với ID này"}), 404

        # Lấy dữ liệu từ request
        data = request.get_json()
        new_notes = data.get("notes", row["notes"])

        cursor.execute("UPDATE images SET notes = ? WHERE id = ?", (new_notes, id))
        conn.commit()
        conn.close()

        return jsonify({
            "message": f"Đã cập nhật ghi chú cho ảnh với ID: {id}",
            "image": {
                "id": id,
                "file_path": row["file_path"],
                "detections": json.loads(row["detections"]) if row["detections"] else [],
                "notes": new_notes
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)