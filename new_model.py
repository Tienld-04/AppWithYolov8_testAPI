import numpy as np
from flask import Flask, request, jsonify
import sqlite3
import json  # Thêm import json

from ultralytics import YOLO
print(np.__version__)
print("ffhjg")
model = YOLO("yolov8n.pt")

app = Flask(__name__)

# Kết nối cơ sở dữ liệu
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row  # Trả về dữ liệu dưới dạng từ điển
    return conn

# Tạo bảng images nếu chưa tồn tại
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            detections TEXT,
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Endpoint để lưu ảnh và thông tin nhận diện
@app.route("/save_image/", methods=["POST"])
def save_image():
    try:
        data = request.get_json()
        file_path = data.get("file_path")
        detections = data.get("detections")

        if not file_path or not detections:
            return jsonify({"error": "Thiếu file_path hoặc detections"}), 400

        # Chuyển detections từ list thành chuỗi JSON
        detections_json = json.dumps(detections)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO images (file_path, detections) VALUES (?, ?)",
            (file_path, detections_json)
        )
        conn.commit()
        image_id = cursor.lastrowid
        conn.close()

        return jsonify({"message": f"Đã lưu ảnh với ID: {image_id}", "image_id": image_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint để lấy danh sách ảnh
@app.route("/images/", methods=["GET"])
def get_images():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images")
        rows = cursor.fetchall()
        conn.close()

        images = []
        for row in rows:
            # Chuyển detections từ chuỗi JSON thành list
            detections = json.loads(row["detections"]) if row["detections"] else []
            images.append({
                "id": row["id"],
                "file_path": row["file_path"],
                "detections": detections,
                "notes": row["notes"]
            })

        return jsonify({"images": images, "message": "Lấy danh sách ảnh thành công"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint để cập nhật ghi chú
@app.route("/images/<int:id>", methods=["PUT"])
def update_image_notes(id):
    try:
        data = request.get_json()
        notes = data.get("notes")

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images WHERE id = ?", (id,))
        image = cursor.fetchone()

        if not image:
            conn.close()
            return jsonify({"error": "Không tìm thấy ảnh với ID này"}), 404

        cursor.execute(
            "UPDATE images SET notes = ? WHERE id = ?",
            (notes, id)
        )
        conn.commit()

        # Chuyển detections từ chuỗi JSON thành list
        detections = json.loads(image["detections"]) if image["detections"] else []
        updated_image = {
            "id": image["id"],
            "file_path": image["file_path"],
            "detections": detections,
            "notes": notes
        }
        conn.close()

        return jsonify({"message": f"Đã cập nhật ghi chú cho ảnh với ID: {id}", "image": updated_image}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint để xóa ảnh
@app.route("/images/<int:id>", methods=["DELETE"])
def delete_image(id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images WHERE id = ?", (id,))
        image = cursor.fetchone()

        if not image:
            conn.close()
            return jsonify({"error": "Không tìm thấy ảnh với ID này"}), 404

        cursor.execute("DELETE FROM images WHERE id = ?", (id,))
        conn.commit()
        conn.close()

        return jsonify({"message": f"Đã xóa ảnh với ID: {id}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Khởi tạo cơ sở dữ liệu khi chạy ứng dụng
if __name__ == "__main__":
    init_db()
    app.run(debug=True)