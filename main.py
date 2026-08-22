

import argparse
import logging
import math
import os
import random
import re
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.CNN_Mamba_CNN_Mamba_Enhanced.HybricMamba import HybricMamba


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def natural_sort_key(s):
    """Hàm sắp xếp chuẩn cho cả số và chuỗi (Tránh lỗi '10' đứng trước '2')"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]


def get_args():
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition Training with Mamba")

    parser.add_argument('--dataset_name', default="German", type=str)
    parser.add_argument('--class_num', default=43, type=int)
    parser.add_argument('--root_dataset_path',
                        default="/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/dataset",
                        type=str)
    parser.add_argument('--save_path',
                        default="/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult/TFJ",
                        type=str)

    parser.add_argument('--picture_size', default=32, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--finetune_lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.02, type=float)
    parser.add_argument('--clip_grad', default=5.0, type=float)
    parser.add_argument('--label_smoothing', default=0.0, type=float)

    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_path',
                        default="/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult/TFJ/CNNMambaCNNMambaEnhancedV4/German/CNNMambaCNNMambaEnhancedV4_train_on_German.pth",
                        type=str)

    return parser.parse_args()



class TrafficSignFullDataset(Dataset):
    def __init__(self, root, transform=None, samples=None, class_to_idx=None):
        self.root = root
        self.transform = transform

        if samples is not None:
            self.samples = samples
            self.class_to_idx = class_to_idx
            return

        self.samples = []
        domains = sorted(os.listdir(root), key=natural_sort_key)
        for domain in domains:
            domain_path = os.path.join(root, domain)
            if not os.path.isdir(domain_path):
                continue

            classes_in_domain = sorted(os.listdir(domain_path), key=natural_sort_key)
            for class_name in classes_in_domain:
                class_path = os.path.join(domain_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                for img_name in sorted(os.listdir(class_path)):
                    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        self.samples.append((img_path, class_name))

        classes = sorted(list(set(s[1] for s in self.samples)), key=natural_sort_key)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[class_name]
        return img, label


def get_transforms(img_size=32):
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=12),  # Giảm từ 15
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),  # Giảm
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), value=0)  # Giảm từ 0.25
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return transform_train, transform_test


def dataloader_prepare(root, batchsize, img_size=32, test_ratio=0.2, seed=42):
    transform_train, transform_test = get_transforms(img_size)
    full_dataset = TrafficSignFullDataset(root)

    num_classes = len(full_dataset.class_to_idx)
    print(f"Số nhãn (classes): {num_classes}")
    print(f"Tổng số ảnh: {len(full_dataset)}")

    indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i][1] for i in indices]

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels
    )

    train_samples = [full_dataset.samples[i] for i in train_idx]
    test_samples = [full_dataset.samples[i] for i in test_idx]

    train_dataset = TrafficSignFullDataset(
        root=root, transform=transform_train, samples=train_samples, class_to_idx=full_dataset.class_to_idx
    )
    test_dataset = TrafficSignFullDataset(
        root=root, transform=transform_test, samples=test_samples, class_to_idx=full_dataset.class_to_idx
    )

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    return train_loader, test_loader


def load_checkpoint_full(model, checkpoint_path, device, optimizer=None):

    print(f"\n[RESUME] Đang nạp checkpoint từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 1. Load model weights
    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ['model_state_dict', 'state_dict', 'model', 'net']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"--> Tìm thấy weights trong Key: '{key}'")
                break
        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        clean_state_dict[name] = v

    missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)

    if len(missing_keys) > 0:
        print(f"⚠️ Warning: Thiếu {len(missing_keys)} keys trong weights!")
        print(f"   - Các key thiếu: {missing_keys[:5]}...")
    if len(unexpected_keys) > 0:
        print(f"⚠️ Warning: Thừa {len(unexpected_keys)} keys không khớp mô hình!")

    # 2. Load optimizer state nếu có
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("--> Đã load optimizer state")
        except Exception as e:
            print(f"⚠️ Warning: Không thể load optimizer state: {e}")

    # 3. Lấy epoch và best_acc
    start_epoch = checkpoint.get('epoch', 0) + 1 if isinstance(checkpoint, dict) else 0
    best_acc = checkpoint.get('best_acc', 0.0) if isinstance(checkpoint, dict) else 0.0

    # 4. Lấy thông tin criterion đã dùng
    used_label_smoothing = checkpoint.get('label_smoothing', 0.0) if isinstance(checkpoint, dict) else 0.0
    used_aux_weight = checkpoint.get('aux_weight', 0.3) if isinstance(checkpoint, dict) else 0.3

    print(f"--> Epoch tiếp theo: {start_epoch}")
    print(f"--> Best Accuracy: {best_acc:.4f}")
    print(f"--> Label Smoothing đã dùng: {used_label_smoothing}")
    print(f"--> Aux Weight đã dùng: {used_aux_weight}")

    return model, optimizer, start_epoch, best_acc, used_label_smoothing, used_aux_weight



def get_aux_weight(epoch, total_epochs, start_weight=0.3, end_weight=0.05):

    progress = min(1.0, epoch / total_epochs)
    weight = start_weight * (1 - progress) + end_weight * progress
    return max(end_weight, weight)


def train_and_evaluate(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_name = "CNNMambaCNNMambaEnhancedV4"
    folder_path = os.path.join(args.save_path, model_name, args.dataset_name)
    os.makedirs(folder_path, exist_ok=True)

    # Tạo logging
    logging.basicConfig(
        filename=os.path.join(folder_path, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    train_loader, test_loader = dataloader_prepare(
        root=args.root_dataset_path,
        batchsize=args.batch_size,
        img_size=args.picture_size
    )

    # Tạo model
    model = HybricMamba(
        dims=(3, 16, 32, 56, 96),
        num_classes=args.class_num,
        mbconv_expand_ratio=4,
        ssm_d_state=8,
        mamba_blocks=(1, 1),
        ssm_frac=0.5,
        conv_frac=0.3,
        use_aux=True,
    ).to(device)

    start_epoch = 0
    best_acc = 0.0
    is_resumed = False
    used_label_smoothing = 0.0
    used_aux_weight = 0.3
    base_lr = args.lr
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=args.weight_decay
    )

    if args.resume and os.path.exists(args.resume_path):
        model, optimizer, start_epoch, best_acc, used_label_smoothing, used_aux_weight = load_checkpoint_full(
            model, args.resume_path, device, optimizer
        )
        is_resumed = True

        if args.finetune_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.finetune_lr
            print(f"--> Đặt LR finetune: {args.finetune_lr}")
            base_lr = args.finetune_lr

        model.eval()
        test_loss, correct, total = 0.0, 0, 0

        test_criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                out = model(x_val)
                logits = out[0] if isinstance(out, tuple) else out
                loss = test_criterion(logits, y_val)

                test_loss += loss.item()
                correct += (logits.argmax(dim=1) == y_val).sum().item()
                total += y_val.size(0)
                break

        init_loss = test_loss
        init_acc = (correct / total) * 100
        print(f"\n[SANITY CHECK] Mẫu thử ngay sau khi nạp Weights:")
        print(f"--> Batch Loss: {init_loss:.4f} | Batch Accuracy: {init_acc:.2f}%")

        if init_loss > 1.5:
            print("LỖI: Loss quá cao (> 1.5). Có thể checkpoint không khớp!")
            print("   Hãy kiểm tra lại file .pth")
        else:
            print(" Checkpoint HỢP LỆ! Bắt đầu Fine-tune...")
            logging.info(f"Resumed from {args.resume_path} at epoch {start_epoch}")

    if is_resumed:
        criterion = nn.CrossEntropyLoss()
        print("--> Dùng CrossEntropyLoss (không label_smoothing) khi resume")
    else:
        smoothing = args.label_smoothing if args.label_smoothing > 0 else 0.0
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        print(f"--> Dùng CrossEntropyLoss với label_smoothing={smoothing}")

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    checkpoint_path = os.path.join(folder_path, f"{model_name}_latest.pth")
    best_checkpoint_path = os.path.join(folder_path, f"{model_name}_best.pth")

    for epoch in range(start_epoch, args.num_epoch):
        progress = float(epoch - start_epoch) / float(max(1, args.num_epoch - start_epoch))
        if is_resumed and epoch < start_epoch + 3:
            warmup_progress = (epoch - start_epoch) / 3.0
            current_lr = base_lr * (0.1 + 0.9 * warmup_progress)
        else:
            current_lr = args.min_lr + 0.5 * (base_lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        aux_weight = get_aux_weight(epoch, args.num_epoch,
                                    start_weight=0.3 if not is_resumed else 0.15,
                                    end_weight=0.05)

        model.train()
        running_loss = 0.0
        num_batches = 0

        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch}/{args.num_epoch}") as tepoch:
            for data, target in tepoch:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    output = model(data)
                    if isinstance(output, tuple):
                        logits = output[0]
                        loss = criterion(logits, target)
                        if len(output) > 1:
                            for aux in output[1:]:
                                loss += aux_weight * criterion(aux, target)
                    else:
                        loss = criterion(output, target)

                if not torch.isfinite(loss):
                    logging.warning(f"Epoch {epoch}: loss NaN, skipping batch")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                num_batches += 1

                tepoch.set_postfix(
                    loss=f"{running_loss / max(num_batches, 1):.4f}",
                    aux_w=f"{aux_weight:.3f}",
                    lr=f"{current_lr:.6f}",
                    gn=f"{grad_norm:.2f}"
                )
        model.eval()
        num_correct = 0
        num_samples = 0
        eval_loss = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                preds = model(x)
                if isinstance(preds, tuple):
                    preds = preds[0]
                loss = criterion(preds, y)
                eval_loss += loss.item()

                predictions = preds.argmax(dim=1)
                num_correct += (predictions == y).sum().item()
                num_samples += y.size(0)

        acc = num_correct / max(num_samples, 1)
        avg_loss = running_loss / max(num_batches, 1)
        avg_eval_loss = eval_loss / max(len(test_loader), 1)

        print(
            f"Epoch {epoch} | Acc: {acc:.4f} | Loss: {avg_loss:.4f} | Eval Loss: {avg_eval_loss:.4f} | LR: {current_lr:.6f}")
        logging.info(
            f"Epoch {epoch} | Acc: {acc:.4f} | Loss: {avg_loss:.4f} | Eval Loss: {avg_eval_loss:.4f} | LR: {current_lr:.6f}")

        # ============================================================
        # SAVE CHECKPOINT
        # ============================================================
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': max(best_acc, acc),
            'best_acc_epoch': epoch if acc > best_acc else best_acc,
            'label_smoothing': args.label_smoothing if not is_resumed else 0.0,
            'aux_weight': aux_weight,
            'used_label_smoothing': 0.0 if is_resumed else args.label_smoothing,
        }

        # Lưu latest
        torch.save(checkpoint_state, checkpoint_path)

        # Lưu best
        if acc > best_acc:
            best_acc = acc
            best_acc_epoch = epoch
            checkpoint_state['best_acc'] = best_acc
            checkpoint_state['best_acc_epoch'] = best_acc_epoch
            torch.save(checkpoint_state, best_checkpoint_path)
            print(f"✓ Saved best model: Acc={best_acc:.4f} at epoch {best_acc_epoch}")
            logging.info(f"Best model saved: Acc={best_acc:.4f} at epoch {best_acc_epoch}")

    print("\nĐang tạo confusion matrix...")
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            predicted = outputs.argmax(dim=1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(30, 28))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Traffic Sign Confusion Matrix")
    plt.tight_layout()

    cm_path = os.path.join(folder_path, "confusion_matrix_all.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved → {cm_path}")
    print(f"\nBest Accuracy: {best_acc:.4f}")
    print(f"Best Epoch: {best_acc_epoch if 'best_acc_epoch' in locals() else 'N/A'}")


if __name__ == "__main__":
    args = get_args()
    train_and_evaluate(args)