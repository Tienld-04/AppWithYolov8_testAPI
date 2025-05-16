import sqlite3
conn = sqlite3.connect("data_images.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM images")
rows = cursor.fetchall()

print("ID | File Path | notes")
print("-" * 30)

for row in rows:
    print(f"{row[0]} | {row[1]}  | {row[3]}")
conn.close()