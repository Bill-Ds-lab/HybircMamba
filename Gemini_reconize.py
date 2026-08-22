import os
import shutil
import time
from PIL import Image
import google.generativeai as genai
from tqdm import tqdm

INPUT_TXT_PATH = "misclassified_AdditiveGaussianNoise_d255.txt"
TRASH_DIR = "./trash_unrecognizable_images"
GEMINI_API_KEY = os.getenv()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3.6-flash")

os.makedirs(TRASH_DIR, exist_ok=True)
PROMPT = """
Phân tích ảnh biển báo giao thông này.

Nhiệm vụ của bạn là xác định xem ảnh có đủ rõ để nhận diện CHÍNH XÁC loại biển báo hay không.

Nếu ảnh:
- quá tối hoặc quá sáng;
- bị nhiễu, mờ, rung hoặc biến dạng nghiêm trọng;
- biển báo bị che khuất quá nhiều;
- biển báo quá nhỏ hoặc không nhìn rõ các chi tiết quan trọng;
- ảnh đen hoàn toàn hoặc gần như không nhìn thấy biển báo;
- hoặc bạn chỉ có thể xác định nhóm/chủng loại chung của biển báo nhưng KHÔNG thể xác định chính xác biển báo cụ thể;

thì hãy trả lời đúng một từ:
UNKNOWN

Đây là làm sạch dữ liệu trước khi cho model học ảnh thực tế, ảnh có thể bị làm nhĩu nhưng nếu nhĩu đến mức mất đặc trưng 
của từng loại biển báo, hoàn toàn không thể nhận ra thì hãy trả lời đúng 1 từ :
UNKNOWN


Ví dụ:
- Chỉ biết đây là biển giới hạn tốc độ nhưng không đọc/xác định được con số cụ thể → UNKNOWN.
- Chỉ biết đây là biển cấm nhưng không xác định được loại biển cấm cụ thể → UNKNOWN.
- Nhìn thấy hình tròn nhưng không thể xác định nội dung biển → UNKNOWN.
- Không nhìn rõ biểu tượng hoặc chữ/số quan trọng → UNKNOWN.

Chỉ khi ảnh đủ rõ để xác định chính xác loại biển báo, hãy trả lời tên/loại biển báo đó.

Ví dụ:
Speed Limit 50 km/h
Stop
Yield
No Entry
Turn Left
Pedestrian Crossing

CHỈ TRẢ VỀ MỘT KẾT QUẢ:
- UNKNOWN
hoặc
- Tên chính xác của biển báo.
Không giải thích thêm.
"""


def parse_paths_from_file(txt_path):
    """Trích xuất danh sách đường dẫn ảnh từ file log txt."""
    image_paths = []
    if not os.path.exists(txt_path):
        print(f"❌ Không tìm thấy file: {txt_path}")
        return image_paths

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Path:" in line:
                path = line.split("Path:")[-1].strip()
                if os.path.exists(path):
                    image_paths.append(path)
                else:
                    print(f"⚠️ Bỏ qua (ảnh không tồn tại): {path}")
    return image_paths


def process_misclassified_images():
    target_files = parse_paths_from_file(INPUT_TXT_PATH)
    print(f"📦 Đã tìm thấy {len(target_files):,} ảnh từ file '{INPUT_TXT_PATH}'")

    if not target_files:
        print("⚡ Không có ảnh nào để xử lý.")
        return

    removed_count = 0
    kept_count = 0

    for img_path in tqdm(target_files, desc="Đang kiểm tra với Gemini"):
        try:
            image = Image.open(img_path)

            # Gửi ảnh sang Gemini API
            response = model.generate_content([PROMPT, image])
            result_text = response.text.strip().upper() if response.text else ""

            if "UNKNOWN" in result_text or len(result_text) == 0:
                filename = os.path.basename(img_path)
                dest_path = os.path.join(TRASH_DIR, filename)

                # Tránh ghi đè nếu trùng tên file trong thư mục rác
                if os.path.exists(dest_path):
                    dest_path = os.path.join(TRASH_DIR, f"{time.time_ns()}_{filename}")

                shutil.move(img_path, dest_path)
                removed_count += 1
            else:
                kept_count += 1

            time.sleep(1.0)  # Giới hạn rate limit API

        except Exception as e:
            print(f"\n⚠️ Lỗi xử lý ảnh {img_path}: {e}")
            time.sleep(2.0)

    print("\n" + "=" * 50)
    print("✅ Hoàn thành!")
    print(f"🗑️ Đã loại bỏ (chuyển vào {TRASH_DIR}): {removed_count:,} ảnh")
    print(f"🖼️ Giữ lại (nhận diện được): {kept_count:,} ảnh")
    print("=" * 50)


if __name__ == "__main__":
    process_misclassified_images()