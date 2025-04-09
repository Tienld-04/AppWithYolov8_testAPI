import sqlite3

# Kết nối đến cơ sở dữ liệu
conn = sqlite3.connect("data_images.db")
cursor = conn.cursor()

# Truy vấn tất cả dữ liệu từ bảng images
cursor.execute("SELECT * FROM images")
rows = cursor.fetchall()

# In tiêu đề cột
print("ID | File Path | notes")
print("-" * 30)

# In dữ liệu
for row in rows:
    print(f"{row[0]} | {row[1]}  | {row[3]}")
# Đóng kết nối
conn.close()