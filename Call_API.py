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
    conn.row_factory = sqlite3.Row  # Để trả về dữ liệu dưới dạng dictionary
    return conn

@app.route("/detect/image/", methods=["POST"])
def detect_image():
    # Cách hàm detect_image hoạt động:
    # 1. Nhận file ảnh từ request
    # 2. Đọc file ảnh bằng OpenCV
    # 3. Nhận diện đối tượng trong ảnh bằng mô hình YOLO
    # 4. Vẽ hình chữ nhật và nhãn lên ảnh
    # 5. Chuyển ảnh thành base64
    # 6. Trả về dữ liệu ảnh và thông tin nhận diện
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không tìm thấy file trong request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "File rỗng"}), 400

        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        # file.read() đọc dữ nhị phân từ file
        #np.uint8 chuyển đổi dữ liệu thành mảng số nguyên 8 bit
        #cv2.IMREAD_COLOR đọc ảnh màu
        #cv2.imdecode giải mã ảnh từ mảng byte thành định dạng ảnh OpenCV
        #np.frombuffer tạo mảng numpy từ buffer byte
        if image is None:
            return jsonify({"error": "Không thể đọc file ảnh"}), 400

        results = model(image) # Nhận diện đối tượng trong ảnh với mô hình YOLO
        # Kết quả trả về là một danh sách các đối tượng được phát hiện trong ảnh
        detections = [] # Danh sách chứa thông tin về các đối tượng được phát hiện
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # Chuyển đổi tọa độ box thành số nguyên
                # Tọa độ box là một danh sách chứa 4 giá trị: [x1, y1, x2, y2]
                label = r.names[int(box.cls)] # Lấy tên nhãn của đối tượng từ mô hình
                confidence = float(box.conf) # Lấy độ tin cậy của đối tượng từ mô hình
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Vẽ hình chữ nhật quanh đối tượng
                cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                # Vẽ nhãn và độ tin cậy lên ảnh
                #cv2.putText vẽ văn bản lên ảnh
                #cv2.FONT_HERSHEY_SIMPLEX là kiểu chữ
                detections.append({"label": label, "confidence": confidence, "box": [x1, y1, x2, y2]}) # Thêm thông tin vào danh sách detections
                
        # Chuyển ảnh thành base64 để gửi về frontend
        _, buffer = cv2.imencode(".jpg", image) #cv2.imencode mã hóa ảnh thành định dạng JPEG
        #cv2.imencode trả về một tuple, phần tử đầu tiên là mã hóa thành công hay không, phần tử thứ hai là buffer chứa dữ liệu ảnh
        image_base64 = base64.b64encode(buffer).decode("utf-8") # Chuyển đổi buffer thành chuỗi base64
        # Chuyển đổi dữ liệu nhị phân thành chuỗi base64 để gửi về frontend
        #base64.b64encode mã hóa dữ liệu nhị phân thành chuỗi base64
        return jsonify({
            "image_data": image_base64, # Chuỗi base64 của ảnh đã nhận diện
            "detections": detections, # Danh sách chứa thông tin về các đối tượng được phát hiện
            "message": "Nhận diện thành công"
        }), 200 

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/detect/video/", methods=["POST"])
def detect_video():
    # Cách hàm detect_video hoạt động:
    # 1. Nhận file video từ request
    # 2. Lưu file video tạm thời
    # 3. Mở video bằng OpenCV
    # 4. Đọc frame đầu tiên từ video
    # 5. Nhận diện đối tượng trong frame đầu tiên bằng mô hình YOLO
    # 6. Vẽ hình chữ nhật và nhãn lên frame
    # 7. Chuyển frame thành base64
    # 8. Xóa file video tạm thời
    # 9. Trả về dữ liệu ảnh và thông tin nhận diện
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không tìm thấy file trong request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "File rỗng"}), 400

        temp_video_path = os.path.join(STATIC_DIR, f"temp_{uuid.uuid4().hex}.mp4") 
        # os.path.join kết hợp đường dẫn thư mục với tên file
        # uuid.uuid4().hex tạo một chuỗi ngẫu nhiên để đặt tên file tạm thời
        # uuid là một thư viện trong Python để tạo ra các ID duy nhất
        # uuid4() tạo một UUID ngẫu nhiên
        file.save(temp_video_path)

        cap = cv2.VideoCapture(temp_video_path) # Mở video bằng OpenCV
        if not cap.isOpened():
            os.remove(temp_video_path) # Nếu không mở được video, xóa file tạm thời
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

        # Chuyển frame thành base64
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
    #Cách hàm save_image hoạt động:
    # 1. Nhận dữ liệu từ request (file_path và detections)
    # 2. Kiểm tra xem file_path và detections có tồn tại không
    # 3. Kết nối đến cơ sở dữ liệu SQLite
    # 4. Thực hiện truy vấn SQL để lưu dữ liệu vào bảng images
    # 5. Đóng kết nối đến cơ sở dữ liệu
    # 6. Trả về ID của ảnh đã lưu và thông báo thành công
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
    # Cách hàm get_images hoạt động:
    # 1. Kết nối đến cơ sở dữ liệu SQLite
    # 2. Thực hiện truy vấn SQL để lấy tất cả dữ liệu từ bảng images
    # 3. Đóng kết nối đến cơ sở dữ liệu
    # 4. Chuyển đổi dữ liệu thành danh sách JSON
    # 5. Trả về danh sách ảnh và thông báo thành công
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
    # Cách hàm get_image_id hoạt động:
    # 1. Kết nối đến cơ sở dữ liệu SQLite
    # 2. Thực hiện truy vấn SQL để lấy dữ liệu từ bảng images theo ID
    # 3. Đóng kết nối đến cơ sở dữ liệu
    # 4. Chuyển đổi dữ liệu thành JSON
    # 5. Trả về dữ liệu ảnh và thông báo thành công
    # 6. Nếu không tìm thấy ảnh theo ID, trả về thông báo lỗi
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
    # Cách hàm delete_image hoạt động:
    # 1. Kết nối đến cơ sở dữ liệu SQLite
    # 2. Thực hiện truy vấn SQL để lấy dữ liệu từ bảng images theo ID
    # 3. Nếu không tìm thấy ảnh theo ID, trả về thông báo lỗi
    # 4. Nếu tìm thấy ảnh, xóa file ảnh trong thư mục static
    # 5. Xóa bản ghi trong cơ sở dữ liệu
    # 6. Đóng kết nối đến cơ sở dữ liệu
    # 7. Trả về thông báo thành công
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
    # Cách hàm update_image_notes hoạt động:
    # 1. Kết nối đến cơ sở dữ liệu SQLite
    # 2. Thực hiện truy vấn SQL để lấy dữ liệu từ bảng images theo ID
    # 3. Nếu không tìm thấy ảnh theo ID, trả về thông báo lỗi
    # 4. Nếu tìm thấy ảnh, lấy dữ liệu từ request (notes mới)
    # 5. Cập nhật ghi chú trong cơ sở dữ liệu
    # 6. Đóng kết nối đến cơ sở dữ liệu
    # 7. Trả về thông báo thành công và dữ liệu ảnh đã cập nhật
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
        new_notes = data.get("notes", row["notes"])  # Nếu không có notes mới, giữ nguyên notes cũ

        # Cập nhật ghi chú trong cơ sở dữ liệu
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