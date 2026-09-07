"""
MambaTSR Single-Transformation Dataset Generator (High-Intensity Weather Edition)
----------------------------------------------------------------------------------
Kịch bản tạo Dataset mới trong đó MỖI ẢNH ĐƯỢC TĂNG CƯỜNG/BIẾN ĐỔI THEO ĐÚNG 1 KIỂU ĐƠN LẺ.
Thuật toán tạo mưa đã được đồng bộ chuẩn cường độ mạnh với weather.py.
"""

import os
import argparse
import random
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def resize_image(img, target_size=(32, 32)):
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)


def transform_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = random.uniform(0.4, 1.6)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def transform_contrast(img):
    alpha = random.uniform(0.4, 1.6)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    adjusted = mean + alpha * (img.astype(np.float32) - mean)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def transform_occlusion(img):
    h, w, _ = img.shape
    mask_h = max(1, int(h * random.uniform(0.2, 0.4)))
    mask_w = max(1, int(w * random.uniform(0.2, 0.4)))
    top = random.randint(0, h - mask_h)
    left = random.randint(0, w - mask_w)

    masked = img.copy()
    fill_color = random.choice([0, 128, 255])
    masked[top:top + mask_h, left:left + mask_w, :] = fill_color
    return masked


def transform_gamma(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def transform_rain_weather_py(img, w_size=3):
    """
    Tái tạo 100% thuật toán tạo mưa từ weather.py:
    - get_noise (value=500)
    - rain_blur (length 15-25 thích ứng kích thước 32x32, angle -60 đến 60)
    - alpha_rain (beta=0.6)
    """
    h, w, _ = img.shape

    # 1. Hàm get_noise(img, value=500)
    noise = np.random.uniform(0, 256, (h, w))
    v = 500 * 0.01  # v = 5
    noise[noise < (256 - v)] = 0  # Giữ lại các đỉnh nhiễu lớn hơn 251

    k_sharpen = np.array([[0, 0.1, 0],
                          [0.1, 8, 0.1],
                          [0, 0.1, 0]])
    noise = cv2.filter2D(noise, -1, k_sharpen)

    # 2. Hàm rain_blur
    length = random.randint(15, 25)  # Scale độ dài vệt mưa vừa vặn khung ảnh 32x32
    angle = random.randint(-60, 60)

    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))
    k = cv2.warpAffine(dig, trans, (length, length))

    # Đảm bảo kernel Gaussian Blur là số lẻ > 0
    k_blur_size = max(1, int(w_size))
    if k_blur_size % 2 == 0:
        k_blur_size += 1
    k = cv2.GaussianBlur(k, (k_blur_size, k_blur_size), 0)

    blurred = cv2.filter2D(noise, -1, k)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    rain_layer = np.array(blurred, dtype=np.float32)

    # 3. Hàm alpha_rain (sửa lỗi ngoặc kép () bị dư)
    rain_3d = np.expand_dims(rain_layer, 2)
    rain_result = img.copy().astype(np.float32)
    beta = 0.6

    for c in range(3):
        rain_result[:, :, c] = rain_result[:, :, c] * (255.0 - rain_3d[:, :, 0]) / 255.0 + beta * rain_3d[:, :, 0]

    return np.clip(rain_result, 0, 255).astype(np.uint8)


def transform_noise(img, mean=0, std=25):
    noise = np.random.normal(mean, std, img.shape)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


# Danh sách các kiểu biến đổi đơn lẻ
TRANSFORM_DICT = {
    "brightness": lambda img: transform_brightness(img),
    "contrast": lambda img: transform_contrast(img),
    "occlusion": lambda img: transform_occlusion(img),
    "dim_g05": lambda img: transform_gamma(img, 0.5),
    "dim_g02": lambda img: transform_gamma(img, 0.2),
    "exp_g15": lambda img: transform_gamma(img, 1.5),
    "exp_g18": lambda img: transform_gamma(img, 1.8),
    "rain_w1": lambda img: transform_rain_weather_py(img, w_size=1),
    "rain_w3": lambda img: transform_rain_weather_py(img, w_size=3),
    "rain_w5": lambda img: transform_rain_weather_py(img, w_size=5),
    "rain_w7": lambda img: transform_rain_weather_py(img, w_size=7),
    "noise_m-120": lambda img: transform_noise(img, mean=-120),
    "noise_m0": lambda img: transform_noise(img, mean=0),
    "noise_m120": lambda img: transform_noise(img, mean=120),
}


def process_dataset(input_dir, output_dir, mode="all_types", target_type=None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm'}

    for split in ['Train', 'Test']:
        split_dir = input_path / split
        if not split_dir.exists():
            continue

        print(f"\n================ Xử lý tập: {split} ================")
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]

        for class_dir in tqdm(class_dirs, desc=f"Tiến trình {split}"):
            out_class_dir = output_path / split / class_dir.name
            out_class_dir.mkdir(parents=True, exist_ok=True)

            image_files = [f for f in class_dir.iterdir() if f.suffix.lower() in valid_extensions]

            for img_file in image_files:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                img_base = resize_image(img, (32, 32))
                base_name = img_file.stem
                ext = img_file.suffix

                if mode == "all_types":
                    cv2.imwrite(str(out_class_dir / f"{base_name}_base32{ext}"), img_base)
                    for t_name, t_func in TRANSFORM_DICT.items():
                        transformed = t_func(img_base)
                        cv2.imwrite(str(out_class_dir / f"{base_name}_{t_name}{ext}"), transformed)

                elif mode == "random_type":
                    t_name = random.choice(list(TRANSFORM_DICT.keys()))
                    transformed = TRANSFORM_DICT[t_name](img_base)
                    cv2.imwrite(str(out_class_dir / f"{base_name}_{t_name}{ext}"), transformed)

                elif mode == "single_type":
                    if target_type not in TRANSFORM_DICT:
                        raise ValueError(
                            f"Kiểu '{target_type}' không hợp lệ. Chọn 1 trong: {list(TRANSFORM_DICT.keys())}")
                    transformed = TRANSFORM_DICT[target_type](img_base)
                    cv2.imwrite(str(out_class_dir / f"{base_name}_{target_type}{ext}"), transformed)

    print(f"\n[Hoàn thành] Dataset đã được lưu tại: {output_path.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tạo Dataset với từng phép biến đổi đơn lẻ")
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/Belgium_TFS',
        help='Thư mục dataset gốc'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/Belgium_ar',
        help='Thư mục lưu dataset mới'
    )
    parser.add_argument('--mode', type=str, default='all_types', choices=['all_types', 'random_type', 'single_type'])
    parser.add_argument('--type', type=str, default='rain_w3')

    args = parser.parse_args()

    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        target_type=args.type
    )