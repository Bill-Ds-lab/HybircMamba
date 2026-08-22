import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.CNN_Mamba_CNN_Mamba_Enhanced.HybricMamba import HybricMamba

DATASET_PATH = "/home/biu-linux/DeepLearning_Projects/DoAnNganh/dataset_reOrgan"
CHECKPOINT_PATH = "/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult/TFJ/Heavy_HybricMamba/German/Heavy_HybricMamba_best.pth"
OUTPUT_TXT_PATH = "Full_imPth_wrong_image.txt"


def create_model():
    model = HybricMamba(
        dims=(3, 32, 64, 112, 176),
        num_classes=43,
        mbconv_expand_ratio=6,
        ssm_d_state=16,
        ssm_ratio=2.0,
        mamba_blocks=(2, 3),
        cnn_blocks=(2, 2),
        ssm_frac=0.7,
        conv_frac=0.2,
        use_aux=True,
    )
    return model


def load_model(checkpoint_path, device):
    model = create_model()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint

    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()} if isinstance(state_dict, dict) else state_dict
    model.load_state_dict(clean_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("📦 Đang đọc dataset...")
    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

    classes = sorted(dataset.classes, key=lambda x: int(x))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    dataset.classes = classes
    dataset.class_to_idx = class_to_idx
    dataset.samples = dataset.make_dataset(DATASET_PATH, class_to_idx, dataset.extensions)
    dataset.targets = [s[1] for s in dataset.samples]

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    print("📦 Đang nạp model...")
    model = load_model(CHECKPOINT_PATH, device)
    print("✅ Load model thành công!")

    misclassified_list = []
    total_images = len(dataset)
    global_idx = 0

    print(f"\n🔍 Đang chạy dự đoán trên toàn bộ dataset ({total_images:,} ảnh)...")

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Processing"):
            images = images.to(device)
            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                confidence = probs[i][pred_label].item() * 100
                img_path = dataset.samples[global_idx + i][0]

                if pred_label != true_label:
                    misclassified_list.append((img_path, true_label, pred_label, confidence))

            global_idx += len(labels)

    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 85 + "\n")
        f.write("DANH SÁCH TẤT CẢ ẢNH BỊ DỰ ĐOÁN SAI TRÊN TOÀN BỘ DATASET\n")
        f.write(f"Tổng số ảnh trong dataset: {total_images:,} ảnh\n")
        f.write(f"Tổng số trường hợp bị đoán sai: {len(misclassified_list):,} ảnh\n")
        if total_images > 0:
            error_rate = (len(misclassified_list) / total_images) * 100
            f.write(f"Tỷ lệ lỗi (Error Rate): {error_rate:.2f}%\n")
            f.write(f"Độ chính xác (Accuracy): {100 - error_rate:.2f}%\n")
        f.write("=" * 85 + "\n\n")

        for path, true_cls, pred_cls, conf in misclassified_list:
            f.write(f"[True: {true_cls:02d} | Pred: {pred_cls:02d} | Conf: {conf:6.2f}%] Path: {path}\n")

    print("\n" + "=" * 50)
    print(f"Hoàn thành!")
    print(f" Đã quét {total_images:,} ảnh thuộc dataset.")
    print(f" Phát hiện {len(misclassified_list):,} ảnh bị đoán sai.")
    print(f" Kết quả chi tiết đã được lưu tại: {OUTPUT_TXT_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()