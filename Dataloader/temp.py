import os
import shutil
import pandas as pd
from tqdm import tqdm

base_dir = r'/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/German_51k'
csv_path = os.path.join(base_dir, 'Test.csv')

output_dir = os.path.join(base_dir, 'Test_organized')

print("Đang đọc file CSV...")
df = pd.read_csv(csv_path)

print(f"Tổng số ảnh trong file CSV: {len(df)}")
print("Cấu trúc file CSV:")
print(df[['ClassId', 'Path']].head())

success_count = 0
error_count = 0

print("\nBắt đầu phân loại ảnh...")
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tiến trình"):
    class_id = str(row['ClassId'])  # Nhãn lớp (ví dụ: '0', '1', ..., '42')
    img_rel_path = row['Path']  # Đường dẫn tương đối từ CSV

    # Lấy tên file ảnh (ví dụ: 00000.png)
    img_name = os.path.basename(img_rel_path)

    # Xác định vị trí file ảnh gốc
    src_path = os.path.join(base_dir, 'Test', img_name)
    if not os.path.exists(src_path):
        src_path = os.path.join(base_dir, img_rel_path)

    # Tạo thư mục đích cho ClassId (ví dụ: German_51k/Test_organized/16/)
    target_class_dir = os.path.join(output_dir, class_id)
    os.makedirs(target_class_dir, exist_ok=True)

    # Đường dẫn file mới
    dst_path = os.path.join(target_class_dir, img_name)

    # Sao chép file
    if os.path.exists(src_path):
        # Dùng shutil.copy để giữ nguyên folder Test gốc (an toàn)
        # Nếu muốn di chuyển trực tiếp để tiết kiệm dung lượng, đổi shutil.copy thành shutil.move
        shutil.copy(src_path, dst_path)
        success_count += 1
    else:
        error_count += 1

print("\n================ HOÀN THÀNH ================")
print(f"✓ Đã phân loại thành công: {success_count} ảnh")
if error_count > 0:
    print(f"✗ Không tìm thấy: {error_count} ảnh")
print(f"📁 Thư mục kết quả: {output_dir}")