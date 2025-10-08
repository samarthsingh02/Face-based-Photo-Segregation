import sqlite3

# List of the specific image filenames to remove
# (You only need the filename, not the whole path)
files_to_delete = [
    "WIN_20201112_15_26_35_Pro.jpg",
    "WIN_20201112_15_26_40_Pro.jpg",
    "WIN_20201112_15_26_44_Pro.jpg",
    "WIN_20250929_17_47_46_Pro.jpg"
]

conn = sqlite3.connect("faces.db")
cursor = conn.cursor()

for filename in files_to_delete:
    # Use a LIKE query to find any path that ends with this filename
    sql_query = "DELETE FROM faces WHERE image_path LIKE ?"
    # The '%' is a wildcard, so this finds the file no matter what folder it was in
    cursor.execute(sql_query, (f'%{filename}',))
    print(f"Deleted entries for {filename}. Rows affected: {cursor.rowcount}")

conn.commit()
conn.close()

print("Cleanup complete.")