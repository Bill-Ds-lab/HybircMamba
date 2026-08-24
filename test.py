

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.CNN_Mamba_CNN_Mamba_Enhanced.CNN_Mamba_CNN_Mamba_EnhancedV4 import HybricMamba


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dataset_path',
                   default="/home/biu-linux/DeepLearning_Projects/DoAnNganh/dataset_reOrgan",
                   type=str)
    p.add_argument('--checkpoint',
                   default="/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult/TFJ/CNNMambaCNNMambaEnhancedV4/German/CNNMambaCNNMambaEnhancedV4_train_on_German.pth",type=str)
    p.add_argument('--target_class_idx', default=10, type=int)
    p.add_argument('--picture_size', default=32, type=int)
    p.add_argument('--seed', default=42, type=int)
    return p.parse_args()


class TrafficSignDataset(Dataset):
    def __init__(self, root, transform=None, samples=None, class_to_idx=None, shuffle_samples=False):
        self.root = root
        self.transform = transform

        if samples is not None and class_to_idx is not None:
            self.samples = samples
            self.class_to_idx = class_to_idx
            if shuffle_samples:
                random.shuffle(self.samples)
            return

        self.samples = []
        class_dirs = sorted([d for d in os.listdir(root)
                             if os.path.isdir(os.path.join(root, d))])

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}

        for class_name in class_dirs:
            class_path = os.path.join(root, class_name)
            if not os.path.isdir(class_path):
                continue
            image_files = [f for f in os.listdir(class_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                self.samples.append((img_path, class_name))

        if shuffle_samples:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[class_name]
        return img, label


def get_model_config_from_checkpoint(checkpoint_path, device='cpu'):
    """Đọc config từ checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_config' in ckpt:
        return ckpt['model_config']

    if 's2' in checkpoint_path or '98k' in checkpoint_path:
        return {
            'dims': (3, 20, 40, 64, 104),
            'ssm_d_state': 12,
            'conv_frac': 0.4,
        }
    else:
        return {
            'dims': (3, 16, 32, 56, 96),
            'ssm_d_state': 8,
            'conv_frac': 0.3,
        }


def print_confusion_matrix_summary(cm, class_names=None, target_class=None):
    """In chi tiết confusion matrix"""
    n_classes = cm.shape[0]

    print("\n" + "=" * 70)
    print("📊 CONFUSION MATRIX SUMMARY")
    print("=" * 70)

    # Accuracy per class
    print("\n📈 Accuracy per class:")
    for i in range(n_classes):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = 100 * correct / max(total, 1)
        marker = " <<< TARGET" if i == target_class else ""
        class_name = class_names[i] if class_names else f"class_{i}"
        print(f"  Class {i:2d} ({class_name:>6}): {acc:5.1f}% ({correct:>4}/{total:>4}){marker}")

    # Most confused classes
    print("\n🔍 Top 10 cặp class bị nhầm lẫn nhiều nhất:")
    confused_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i][j] > 0:
                confused_pairs.append((cm[i][j], i, j))

    confused_pairs.sort(reverse=True)
    for count, true_class, pred_class in confused_pairs[:10]:
        true_name = class_names[true_class] if class_names else f"class_{true_class}"
        pred_name = class_names[pred_class] if class_names else f"class_{pred_class}"
        print(f"  {true_name} → {pred_name}: {count} ảnh")

    # Target class analysis
    if target_class is not None:
        row = cm[target_class]
        total = row.sum()
        correct = row[target_class]

        print(f"\n🎯 Class {target_class} ({class_names[target_class] if class_names else ''}):")
        print(f"  - Total: {total} ảnh test")
        print(f"  - Đúng: {correct} ({100 * correct / max(total, 1):.1f}%)")
        print(f"  - Sai: {total - correct} ({100 * (total - correct) / max(total, 1):.1f}%)")

        # Các class bị nhầm
        wrong_preds = [(row[j], j) for j in range(n_classes) if j != target_class and row[j] > 0]
        wrong_preds.sort(reverse=True)
        if wrong_preds:
            print("  - Bị nhầm sang:")
            for count, pred_class in wrong_preds[:10]:
                pred_name = class_names[pred_class] if class_names else f"class_{pred_class}"
                print(f"    → {pred_name}: {count} ảnh ({100 * count / max(total, 1):.1f}%)")


def main():
    args = get_args()
    random.seed(args.seed)

    print("=" * 70)
    print("BƯỚC 1: Đếm số ảnh mỗi lớp — kiểm tra mất cân bằng dữ liệu")
    print("=" * 70)

    full = TrafficSignDataset(args.root_dataset_path, shuffle_samples=False)
    idx_to_class = {v: k for k, v in full.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    counts = {}
    for _, class_name in full.samples:
        counts[class_name] = counts.get(class_name, 0) + 1

    sorted_counts = sorted(counts.items(), key=lambda kv: full.class_to_idx[kv[0]])
    for class_name, cnt in sorted_counts:
        marker = "  <<< TARGET" if full.class_to_idx[class_name] == args.target_class_idx else ""
        print(f"  class_idx={full.class_to_idx[class_name]:>3} (folder='{class_name}') : {cnt:>5} ảnh{marker}")

    target_class_name = idx_to_class[args.target_class_idx]
    target_count = counts[target_class_name]
    avg_count = sum(counts.values()) / len(counts)

    print(f"\n--> Class {args.target_class_idx} (folder '{target_class_name}'): {target_count} ảnh")
    print(f"--> Trung bình mỗi lớp: {avg_count:.0f} ảnh")

    if args.checkpoint is None:
        print("\n(Không truyền --checkpoint nên bỏ qua confusion matrix)")
        return

    print("\n" + "=" * 70)
    print("BƯỚC 2: Tính confusion matrix chi tiết từ checkpoint")
    print("=" * 70)

    transform_test = transforms.Compose([
        transforms.Resize((args.picture_size, args.picture_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    indices = list(range(len(full)))
    labels = [full.class_to_idx[full.samples[i][1]] for i in indices]

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=args.seed, shuffle=True, stratify=labels
    )

    test_samples = [full.samples[i] for i in test_idx]

    test_dataset = TrafficSignDataset(
        root=args.root_dataset_path,
        transform=transform_test,
        samples=test_samples,
        class_to_idx=full.class_to_idx,
        shuffle_samples=False
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_model_config_from_checkpoint(args.checkpoint, device)
    print(f"📊 Model config từ checkpoint:")
    print(f"   dims: {config['dims']}")
    print(f"   ssm_d_state: {config['ssm_d_state']}")
    print(f"   conv_frac: {config['conv_frac']}")

    model = HybricMamba(
        dims=(3, 16, 32, 56, 96),
        num_classes=43,
        mbconv_expand_ratio=4,
        ssm_d_state=8,
        mamba_blocks=(1, 1),
        ssm_frac=0.5,
        conv_frac=0.3,
        use_aux=True,
    ).to(device)

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Evaluation
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            pred = out.argmax(dim=1).cpu().numpy()
            true_labels.extend(y.numpy())
            pred_labels.extend(pred)

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # ============================================================
    # IN CHI TIẾT CONFUSION MATRIX
    # ============================================================
    print_confusion_matrix_summary(cm, class_names, args.target_class_idx)

    # ============================================================
    # VẼ HEATMAP CONFUSION MATRIX (toàn bộ 43x43)
    # ============================================================
    print("\n" + "=" * 70)
    print("BƯỚC 3: Vẽ Confusion Matrix Heatmap")
    print("=" * 70)

    # Vẽ full confusion matrix
    fig, ax = plt.subplots(figsize=(25, 22))

    # Sử dụng log scale để thấy rõ các giá trị nhỏ
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    cm_display.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)

    ax.set_title(f'Confusion Matrix - 43 Classes (Test set)', fontsize=16)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    full_cm_path = 'confusion_matrix_full.png'
    plt.savefig(full_cm_path, dpi=200, bbox_inches='tight')
    print(f"✅ Đã lưu full confusion matrix: {full_cm_path}")
    plt.close()

    print("\n🔍 Vẽ chi tiết cho class 10...")

    # Lấy row và column của target class
    target_row = cm[args.target_class_idx]
    target_col = cm[:, args.target_class_idx]

    relevant_classes = set()
    for j in range(len(target_row)):
        if target_row[j] > 0 and j != args.target_class_idx:
            relevant_classes.add(j)

    for i in range(len(target_col)):
        if target_col[i] > 0 and i != args.target_class_idx:
            relevant_classes.add(i)

    # Thêm target class
    relevant_classes.add(args.target_class_idx)

    # Nếu có quá nhiều, lấy top 20
    if len(relevant_classes) > 20:
        # Sắp xếp theo tổng tương tác
        interactions = {}
        for c in relevant_classes:
            interactions[c] = target_row[c] + target_col[c]
        sorted_classes = sorted(interactions.items(), key=lambda x: -x[1])
        relevant_classes = {c for c, _ in sorted_classes[:20]}

    relevant_classes = sorted(relevant_classes)

    # Tạo sub-matrix
    sub_cm = cm[np.ix_(relevant_classes, relevant_classes)]
    sub_labels = [class_names[i] for i in relevant_classes]

    fig, ax = plt.subplots(figsize=(14, 12))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=sub_cm, display_labels=sub_labels)
    cm_display.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)

    # Đánh dấu target class
    target_pos = relevant_classes.index(args.target_class_idx)
    ax.add_patch(plt.Rectangle((target_pos - 0.5, target_pos - 0.5), 1, 1,
                               fill=False, edgecolor='red', linewidth=3))

    ax.set_title(f'Confusion Matrix - Classes liên quan đến class {args.target_class_idx}', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, fontsize=9)
    plt.yticks(fontsize=9)

    plt.tight_layout()
    zoom_cm_path = f'confusion_matrix_class{args.target_class_idx}_zoom.png'
    plt.savefig(zoom_cm_path, dpi=200, bbox_inches='tight')
    print(f"✅ Đã lưu zoom confusion matrix: {zoom_cm_path}")
    plt.close()

    # ============================================================
    # IN BẢNG CHI TIẾT CHO CLASS 10
    # ============================================================
    print("\n" + "=" * 70)
    print(f"📊 CHI TIẾT CLASS {args.target_class_idx} ({target_class_name})")
    print("=" * 70)

    row = cm[args.target_class_idx]
    total = row.sum()
    correct = row[args.target_class_idx]

    print(f"\n📈 Thống kê:")
    print(f"  - Tổng số ảnh test: {total}")
    print(f"  - Dự đoán đúng: {correct} ({100 * correct / max(total, 1):.1f}%)")
    print(f"  - Dự đoán sai: {total - correct} ({100 * (total - correct) / max(total, 1):.1f}%)")

    print(f"\n📉 Phân phối dự đoán sai:")
    wrong_preds = [(row[j], j, class_names[j]) for j in range(len(row))
                   if j != args.target_class_idx and row[j] > 0]
    wrong_preds.sort(reverse=True)

    for count, pred_idx, pred_name in wrong_preds:
        print(f"  → Class {pred_idx:2d} ({pred_name:>6}): {count:>4} ảnh ({100 * count / max(total, 1):>5.1f}%)")

    # Thống kê class nào bị nhầm sang class 10
    col = cm[:, args.target_class_idx]
    confused_from = [(col[i], i, class_names[i]) for i in range(len(col))
                     if i != args.target_class_idx and col[i] > 0]
    confused_from.sort(reverse=True)

    if confused_from:
        print(f"\n📈 Các class bị nhầm sang class {args.target_class_idx}:")
        for count, from_idx, from_name in confused_from[:10]:
            print(f"  ← Class {from_idx:2d} ({from_name:>6}): {count:>4} ảnh")


if __name__ == "__main__":
    main()