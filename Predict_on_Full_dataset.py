import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.CNN_Mamba_CNN_Mamba_Enhanced.HybricMamba import HybricMamba

# Cấu hình đường dẫn
DATASET_PATH = "/home/biu-linux/DeepLearning_Projects/DoAnNganh/dataset_reOrgan"
CHECKPOINT_PATH = "/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult/TFJ/Heavy_HybricMamba/German/Heavy_HybricMamba_best.pth"
OUTPUT_DIR = "evaluation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    """Load weights từ checkpoint vào model"""
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


def evaluate_model(model, dataloader, device):
    """Chạy dự đoán trên toàn bộ dataloader"""
    all_preds = []
    all_labels = []

    print("🔍 Đang dự đoán trên toàn bộ dataset...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)

            # Nếu output dạng tuple (main_out, aux_out)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, num_classes, save_path, normalize=False, title='Confusion Matrix'):
    """Vẽ và lưu Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    if normalize:
        cm_plot = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        fmt = '.2f'
        title = f'{title} (Normalized)'
    else:
        cm_plot = cm
        fmt = 'd'
        title = f'{title} (Counts)'

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=[str(i) for i in range(num_classes)],
        yticklabels=[str(i) for i in range(num_classes)],
        cbar=True
    )

    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu confusion matrix: {save_path}")


def plot_class_accuracy(y_true, y_pred, num_classes, save_path):
    """Vẽ biểu đồ độ chính xác từng class"""
    class_acc = []
    for i in range(num_classes):
        mask = y_true == i
        if np.sum(mask) > 0:
            acc = np.sum(y_pred[mask] == i) / np.sum(mask)
            class_acc.append(acc)
        else:
            class_acc.append(0)

    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(num_classes), class_acc, color='steelblue', alpha=0.8)

    for i, bar in enumerate(bars):
        if class_acc[i] < 0.7:
            bar.set_color('red')
        elif class_acc[i] < 0.85:
            bar.set_color('orange')

    plt.xlabel('Class ID', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Accuracy per Class', fontsize=16)
    plt.xticks(range(num_classes), [f"{i:02d}" for i in range(num_classes)], rotation=90)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu biểu đồ accuracy per class: {save_path}")


def save_evaluation_report(y_true, y_pred, num_classes, output_dir):
    """Ghi báo cáo tổng quan ra file .txt"""
    report_path = os.path.join(output_dir, 'evaluation_report.txt')

    total_samples = len(y_true)
    correct = np.sum(y_true == y_pred)
    accuracy = correct / total_samples

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("📊 BÁO CÁO ĐÁNH GIÁ MÔ HÌNH CNN_Mamba_CNN_Mamba_EnhancedV4\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"📈 TỔNG QUAN:\n")
        f.write(f"  - Tổng số mẫu dataset: {total_samples:,}\n")
        f.write(f"  - Dự đoán đúng: {correct:,} ({accuracy * 100:.2f}%)\n")
        f.write(f"  - Dự đoán sai: {total_samples - correct:,} ({(1 - accuracy) * 100:.2f}%)\n\n")

        f.write("📊 ACCURACY THEO TỪNG CLASS:\n")
        f.write("-" * 80 + "\n")
        for i in range(num_classes):
            mask = y_true == i
            if np.sum(mask) > 0:
                acc = np.sum(y_pred[mask] == i) / np.sum(mask)
                correct_count = np.sum(y_pred[mask] == i)
                total_count = np.sum(mask)
                f.write(f"  Class {i:02d}: {correct_count:6d}/{total_count:6d} = {acc * 100:6.2f}%\n")
            else:
                f.write(f"  Class {i:02d}: Không có mẫu dữ liệu\n")
        f.write("\n" + "=" * 80 + "\n")

    print(f"✅ Đã lưu báo cáo đánh giá: {report_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")

    # Fix 1: Thiết lập resize chuẩn (32, 32)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("📦 Đang đọc dataset...")
    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

    # Fix 2: Đảm bảo thứ tự Class theo đúng dạng số (0, 1, 2... 42) thay vì sắp xếp chuỗi
    classes = sorted(dataset.classes, key=lambda x: int(x))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    dataset.classes = classes
    dataset.class_to_idx = class_to_idx
    dataset.samples = dataset.make_dataset(DATASET_PATH, class_to_idx, dataset.extensions)
    dataset.targets = [s[1] for s in dataset.samples]

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    print(f"📊 Tổng số ảnh: {len(dataset):,}")
    print(f"📊 Tổng số class: {len(dataset.classes)}")

    # Load Model đúng cấu hình
    print("📦 Đang nạp model và weights...")
    model = load_model(CHECKPOINT_PATH, device)
    print("✅ Model loaded successfully!")

    # Chạy đánh giá
    y_pred, y_true = evaluate_model(model, dataloader, device)

    # Xuất đồ thị và báo cáo
    print("\n📊 Đang tạo biểu đồ và Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred, num_classes=43, save_path=os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), normalize=False)
    plot_confusion_matrix(y_true, y_pred, num_classes=43, save_path=os.path.join(OUTPUT_DIR, 'confusion_matrix_normalized.png'), normalize=True)
    plot_class_accuracy(y_true, y_pred, num_classes=43, save_path=os.path.join(OUTPUT_DIR, 'class_accuracy.png'))
    save_evaluation_report(y_true, y_pred, num_classes=43, output_dir=OUTPUT_DIR)

    total = len(y_true)
    correct = np.sum(y_true == y_pred)
    acc = (correct / total) * 100

    print("\n" + "=" * 50)
    print("🎉 ĐÃ HOÀN THÀNH ĐÁNH GIÁ!")
    print(f"📈 Total Accuracy: {acc:.2f}% ({correct:,}/{total:,})")
    print(f"📁 Tệp kết quả lưu trong thư mục: {OUTPUT_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    main()