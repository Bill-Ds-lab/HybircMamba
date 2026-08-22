import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.CNN_Mamba_CNN_Mamba_Enhanced.HybricMamba import HybricMamba

DATASET_PATH = "/home/biu-linux/DeepLearning_Projects/DoAnNganh/dataset_reOrgan"
CHECKPOINT_PATH = "/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult/TFJ/HybricMamba/German/HybricMamba_best.pth"
OUTPUT_TXT_PATH = "misclassified_as_class10.txt"
TARGET_CLASS = 1


def create_model():
    model = HybricMamba(
        dims=(3, 16, 32, 56, 96),
        num_classes=43,
        mbconv_expand_ratio=4,
        ssm_d_state=8,
        mamba_blocks=(1, 1),
        ssm_frac=0.5,
        conv_frac=0.3,
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

    print("Đang đọc dataset...")
    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

    classes = sorted(dataset.classes, key=lambda x: int(x))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    dataset.classes = classes
    dataset.class_to_idx = class_to_idx
    dataset.samples = dataset.make_dataset(DATASET_PATH, class_to_idx, dataset.extensions)
    dataset.targets = [s[1] for s in dataset.samples]

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    print(" Đang nạp model...")
    model = load_model(CHECKPOINT_PATH, device)
    print(" Load model thành công!")

    misclassified_list = []
    global_idx = 0

    print(f"\n🔍 Đang quét dataset để tìm các ảnh bị đoán nhầm sang Class {TARGET_CLASS:02d}...")

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

                if pred_label == TARGET_CLASS and true_label != TARGET_CLASS:
                    misclassified_list.append((img_path, true_label, pred_label, confidence))

            global_idx += len(labels)

    # Ghi danh sách ra file text
    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 85 + "\n")
        f.write(f"DANH SÁCH ẢNH BỊ DỰ ĐOÁN NHẦM THÀNH CLASS {TARGET_CLASS:02d} (FALSE POSITIVES)\n")
        f.write(f"Tổng số trường hợp bị nhầm: {len(misclassified_list):,} ảnh\n")
        f.write("=" * 85 + "\n\n")

        for path, true_cls, pred_cls, conf in misclassified_list:
            f.write(f"[True: {true_cls:02d} | Pred: {pred_cls:02d} | Conf: {conf:6.2f}%] Path: {path}\n")

    print("\n" + "=" * 50)
    print(f" Hoàn thành! Tìm thấy {len(misclassified_list):,} ảnh bị đoán nhầm sang Class {TARGET_CLASS:02d}.")
    print(f" Kết quả đã lưu vào file: {OUTPUT_TXT_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()