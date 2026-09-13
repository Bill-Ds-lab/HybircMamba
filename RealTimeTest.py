import argparse
import os
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
import time

from models.mambaTRS.Vmamba_ultils import Super_Mamba
from models.HybricMamba.HybricMamba import HybricMamba

GTSRB_CLASSES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for heavy vehicles",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Heavy vehicles prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing by heavy vehicles"
}

"""
def build_light_hybric_mamba(num_classes=43):
    return HybricMamba(
        dims=(3, 16, 32, 56, 96),
        num_classes=num_classes,
        mbconv_expand_ratio=4,
        ssm_d_state=8,
        mamba_blocks=(1, 1),
        ssm_frac=0.5,
        conv_frac=0.3,
        use_aux=True,
    )
"""
def build_light_hybric_mamba(num_classes=43):
    return Super_Mamba(dims=3, depth=3, num_classes=num_classes)

def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {checkpoint_path}")

    print(f"--> Đang tải checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']

    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    return model


def get_test_transform(img_size=32):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def main():
    parser = argparse.ArgumentParser(description="Real-time Traffic Sign Recognition")
    parser.add_argument(
        '--weights',
        type=str,
        default="Ressult/TFJ/SUPER_MAMBA_DEPT_3/German/SUPER_MAMBA_DEPT_3_best.pth",
        help="Đường dẫn tới file checkpoint .pth"
    )
    parser.add_argument('--camera_id', type=int, default=0, help="ID Webcam (thường là 0)")
    parser.add_argument('--roi_size', type=int, default=224, help="Kích thước vùng cắt trên màn hình camera")
    parser.add_argument('--conf_threshold', type=float, default=85.0, help="Ngưỡng độ tin cậy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    model = build_light_hybric_mamba(num_classes=43).to(device)
    model = load_checkpoint(model, args.weights, device)

    transform = get_test_transform(img_size=32)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Lỗi: Không thể kết nối tới camera ID {args.camera_id}")
        return

    print("Mở camera thành công. Nhấn 'q' để thoát.")
    last_predict_time = 0
    predict_interval = 0.3
    label_text = "Dang quet..."
    conf_text = ""
    box_color = (0, 255, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được luồng hình ảnh từ camera.")
            break

        h, w, _ = frame.shape
        box_size = args.roi_size
        x1 = max(0, (w - box_size) // 2)
        y1 = max(0, (h - box_size) // 2)
        x2 = min(w, x1 + box_size)
        y2 = min(h, y1 + box_size)

        current_time = time.time()

        if current_time - last_predict_time >= predict_interval:
            roi = frame[y1:y2, x1:x2]
            if roi.size != 0:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(roi_rgb)
                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    if isinstance(output, tuple):
                        output = output[0]

                    probabilities = torch.softmax(output, dim=1)
                    conf, pred_class = torch.max(probabilities, 1)

                    class_id = pred_class.item()
                    confidence_score = conf.item() * 100

                    if confidence_score >= args.conf_threshold:
                        class_name = GTSRB_CLASSES.get(class_id, "Unknown")
                        label_text = f"ID [{class_id}]: {class_name}"
                        conf_text = f"Confidence: {confidence_score:.2f}%"
                        box_color = (0, 255, 0)  # Xanh lá khi nhận diện được
                    else:
                        label_text = "Khong tim thay trong tap du lieu"
                        conf_text = f"Confidence: {confidence_score:.2f}% (<{args.conf_threshold:.0f}%)"
                        box_color = (0, 0, 255)  # Đỏ khi không đủ độ tin cậy

            last_predict_time = current_time

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, "Dua bien bao vao day", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        cv2.rectangle(frame, (10, 10), (520, 85), (0, 0, 0), -1)

        text_color = (0, 255, 255) if box_color == (0, 255, 0) else (0, 0, 255)
        cv2.putText(frame, label_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(frame, conf_text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("GTSRB Real-time Recognition - LIGHT_HYBRIC_MAMBA", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()