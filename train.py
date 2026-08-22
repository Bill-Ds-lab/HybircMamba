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

from models.CNN_Mamba_CNN_Mamba_Enhanced.CNN_Mamba_CNN_Mamba_EnhancedV4 import HybricMamba
from models.CNN_Mamba_CNN_Mamba_Enhanced.CNN_Mamba_CNN_Mamba_EnhancedV4_v2 import CNN_Mamba_CNN_Mamba_EnhancedV4_v2
from models.vmamba.Vmamba_ultils import Super_Mamba


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]


def setup_logging(folder_path):
    log_file = os.path.join(folder_path, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition Training with Mamba")

    parser.add_argument('--dataset_name', default="German", type=str)
    parser.add_argument('--class_num', default=43, type=int)
    parser.add_argument('--root_dataset_path',
                        default="/home/biu-linux/DeepLearning_Projects/DoAnNganh/dataset_reOrgan",
                        type=str)  # Đã cập nhật đường dẫn mới
    parser.add_argument('--save_path',
                        default="/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult/TFJ",
                        type=str)

    parser.add_argument('--picture_size', default=32, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epoch', default=110, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--finetune_lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.02, type=float)
    parser.add_argument('--clip_grad', default=5.0, type=float)
    parser.add_argument('--label_smoothing', default=0.0, type=float)

    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_path',
                        default="/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult/TFJ/Super_Mamba_dim_3/German/Super_Mamba_dim_3_best.pth",
                        type=str)

    return parser.parse_args()


# ============================================================
# DATASET MỚI - CẤU TRÚC ĐƠN GIẢN (class trực tiếp)
# ============================================================

class TrafficSignDataset(Dataset):


    def __init__(self, root, transform=None, samples=None, class_to_idx=None, shuffle_samples=True):
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
                             if os.path.isdir(os.path.join(root, d))],
                            key=natural_sort_key)

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}

        # Duyệt qua từng class và lấy ảnh
        for class_name in class_dirs:
            class_path = os.path.join(root, class_name)
            if not os.path.isdir(class_path):
                continue

            # Lấy tất cả ảnh trong class
            image_files = [f for f in os.listdir(class_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                self.samples.append((img_path, class_name))

        # Trộn dữ liệu
        if shuffle_samples:
            random.shuffle(self.samples)

        print(f"✅ Dataset: {len(self.samples)} samples, {len(self.class_to_idx)} classes")

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
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.06), value=0)
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return transform_train, transform_test


def dataloader_prepare(root, batchsize, img_size=32, test_ratio=0.2, seed=42, logger=None):

    transform_train, transform_test = get_transforms(img_size)

    full_dataset = TrafficSignDataset(root, shuffle_samples=True)

    num_classes = len(full_dataset.class_to_idx)
    log_msg = f"Số nhãn (classes): {num_classes}"
    print(log_msg)
    if logger:
        logger.info(log_msg)

    log_msg = f"Tổng số ảnh: {len(full_dataset)}"
    print(log_msg)
    if logger:
        logger.info(log_msg)

    indices = list(range(len(full_dataset)))
    labels = [full_dataset.class_to_idx[full_dataset.samples[i][1]] for i in indices]

    # Split train/test với stratified
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels
    )

    train_samples = [full_dataset.samples[i] for i in train_idx]
    test_samples = [full_dataset.samples[i] for i in test_idx]

    # Tạo train dataset (có trộn)
    train_dataset = TrafficSignDataset(
        root=root,
        transform=transform_train,
        samples=train_samples,
        class_to_idx=full_dataset.class_to_idx,
        shuffle_samples=True
    )

    # Tạo test dataset (không trộn)
    test_dataset = TrafficSignDataset(
        root=root,
        transform=transform_test,
        samples=test_samples,
        class_to_idx=full_dataset.class_to_idx,
        shuffle_samples=False
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=True,  # Vẫn shuffle để trộn thêm mỗi epoch
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    log_msg = f"Train: {len(train_dataset)} | Test: {len(test_dataset)}"
    print(log_msg)
    if logger:
        logger.info(log_msg)

    # Kiểm tra phân bố batch
    check_batch_distribution(train_loader, num_batches=5, logger=logger)

    return train_loader, test_loader


def check_batch_distribution(dataloader, num_batches=5, logger=None):
    """Kiểm tra xem dữ liệu có được trộn đều không"""
    log_msg = "\n🔍 Kiểm tra phân bố class trong batch:"
    print(log_msg)
    if logger:
        logger.info(log_msg)

    class_counts = []
    for i, (_, labels) in enumerate(dataloader):
        if i >= num_batches:
            break
        unique_classes = torch.unique(labels).tolist()
        class_counts.append(len(unique_classes))
        log_msg = f"  Batch {i}: {len(unique_classes)} classes (first 5: {unique_classes[:5]})"
        print(log_msg)
        if logger:
            logger.info(log_msg)

    avg_classes = np.mean(class_counts) if class_counts else 0
    if avg_classes < 10:
        log_msg = f"⚠️ WARNING: Trung bình {avg_classes:.1f} classes/batch - Dữ liệu chưa được trộn đều!"
        print(log_msg)
        if logger:
            logger.warning(log_msg)
    else:
        log_msg = f"✅ Shuffle tốt! Trung bình {avg_classes:.1f} classes/batch"
        print(log_msg)
        if logger:
            logger.info(log_msg)


def load_checkpoint_safely(model, checkpoint_path, device, logger=None):
    log_msg = f"\n[RESUME] Đang nạp checkpoint từ: {checkpoint_path}"
    print(log_msg)
    if logger:
        logger.info(log_msg)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ['model_state_dict', 'state_dict', 'model', 'net']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                log_msg = f"--> Tìm thấy weights trong Key: '{key}'"
                print(log_msg)
                if logger:
                    logger.info(log_msg)
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
        log_msg = f"⚠️ Warning: Thiếu {len(missing_keys)} keys trong weights!"
        print(log_msg)
        if logger:
            logger.warning(log_msg)
    if len(unexpected_keys) > 0:
        log_msg = f"⚠️ Warning: Thừa {len(unexpected_keys)} keys không khớp mô hình!"
        print(log_msg)
        if logger:
            logger.warning(log_msg)

    start_epoch = checkpoint.get('epoch', -1) + 1 if isinstance(checkpoint, dict) else 0
    best_acc = checkpoint.get('best_acc', 0.0) if isinstance(checkpoint, dict) else 0.0

    log_msg = f"--> Epoch tiếp theo: {start_epoch}"
    print(log_msg)
    if logger:
        logger.info(log_msg)

    log_msg = f"--> Best Accuracy: {best_acc:.4f}"
    print(log_msg)
    if logger:
        logger.info(log_msg)

    return model, start_epoch, best_acc


    """
    Light
    def get_lr(epoch, base_lr=0.0001):
    if epoch < 20:
        return base_lr
    elif epoch < 25:
        return base_lr * 0.1
    elif epoch < 30:
        return base_lr * 0.005
    elif epoch < 35:
        return base_lr * 0.001
    elif epoch < 40:
        return base_lr * 0.0005
    elif epoch < 45:
        return base_lr * 0.0001
    elif epoch < 50:
        return base_lr * 0.00005
    elif epoch < 55:
        return base_lr * 0.00001
    elif epoch < 60:
        return base_lr * 0.000005
    else:
        return base_lr * 0.0000001

"""


def get_lr(epoch, base_lr=0.0001):
    if epoch < 10:
        return base_lr
    elif epoch < 15:
        return base_lr * 0.1
    elif epoch < 18:
        return base_lr * 0.01
    elif epoch < 23:
        return base_lr * 0.001
    else:
        return base_lr * 0.0001


def train_and_evaluate(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_name = ("Super_Mamba_dim_3")
    folder_path = os.path.join(args.save_path, model_name, args.dataset_name)
    os.makedirs(folder_path, exist_ok=True)

    logger = setup_logging(folder_path)
    logger.info("=" * 60)
    logger.info(f"Starting training: {model_name}")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Dataset path: {args.root_dataset_path}")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Num epochs: {args.num_epoch}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Clip grad: {args.clip_grad}")
    logger.info(f"Label smoothing: {args.label_smoothing}")
    logger.info("=" * 60)

    train_loader, test_loader = dataloader_prepare(
        root=args.root_dataset_path,
        batchsize=args.batch_size,
        img_size=args.picture_size,
        logger=logger
    )
    """
    Light_HybricMamba =HybricMamba(
        dims=(3, 16, 32, 56, 96),
        num_classes=43,
        mbconv_expand_ratio=4,
        ssm_d_state=8,
        mamba_blocks=(1, 1),
        ssm_frac=0.5,
        conv_frac=0.3,
        use_aux=True,
    )

    Medium_HybricMamba = HybricMamba(
        dims=(3, 24, 48, 80, 128),
        num_classes=43,
        mbconv_expand_ratio=4,
        ssm_d_state=12,
        ssm_ratio=1.5,
        mamba_blocks=(2, 2),
        cnn_blocks=(1, 2),
        ssm_frac=0.6,
        conv_frac=0.25,
        use_aux=True,
    )

    Heavy_HybricMamba = HybricMamba(
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

    SuperMamba= Super_Mamba(dims=3, depth=4, num_classes=43)
    """

    model= Super_Mamba(dims=3, depth=3, num_classes=43).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    start_epoch = 0
    best_acc = 0.0
    is_resumed = False

    if args.resume and os.path.exists(args.resume_path):
        model, start_epoch, best_acc = load_checkpoint_safely(model, args.resume_path, device, logger)
        is_resumed = True

        # SANITY CHECK
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                out = model(x_val)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, y_val)

                test_loss += loss.item()
                correct += (logits.argmax(dim=1) == y_val).sum().item()
                total += y_val.size(0)
                break

        init_loss = test_loss
        init_acc = (correct / total) * 100
        log_msg = f"\n[SANITY CHECK] Mẫu thử ngay sau khi nạp Weights:"
        print(log_msg)
        logger.info(log_msg)
        log_msg = f"--> Batch Loss: {init_loss:.4f} | Batch Accuracy: {init_acc:.2f}%"
        print(log_msg)
        logger.info(log_msg)

        if init_loss > 1.5:
            log_msg = "❌ LỖI: Loss quá cao (> 1.5). Checkpoint không khớp!"
            print(log_msg)
            logger.error(log_msg)
            return
        else:
            log_msg = "✅ CHECKPOINT HỢP LỆ! Bắt đầu Fine-tune..."
            print(log_msg)
            logger.info(log_msg)

    base_lr = args.finetune_lr if is_resumed else args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)

    total_epochs_to_run = max(1, args.num_epoch - start_epoch)

    for epoch in range(start_epoch, args.num_epoch):
        progress = float(epoch - start_epoch) / float(total_epochs_to_run)
        current_lr = get_lr(epoch, args.lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

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
                                loss += 0.3 * criterion(aux, target)
                    else:
                        loss = criterion(output, target)

                if not torch.isfinite(loss):
                    logger.warning(f"Epoch {epoch}: loss NaN, skipping batch")
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
                    lr=f"{current_lr:.6f}",
                    gn=f"{grad_norm:.2f}"
                )

        # EVALUATION
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

        log_msg = f"Epoch {epoch} | Acc: {acc:.4f} | Loss: {avg_loss:.4f} | Eval Loss: {avg_eval_loss:.4f} | LR: {current_lr:.6f}"
        print(log_msg)
        logger.info(log_msg)

        # SAVE CHECKPOINT
        checkpoint_path = os.path.join(folder_path, f"{model_name}_latest.pth")
        best_checkpoint_path = os.path.join(folder_path, f"{model_name}_best.pth")

        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': max(best_acc, acc),
            'label_smoothing': args.label_smoothing,
        }
        torch.save(checkpoint_state, checkpoint_path)

        if acc > best_acc:
            best_acc = acc
            checkpoint_state['best_acc'] = best_acc
            torch.save(checkpoint_state, best_checkpoint_path)
            log_msg = f"✓ Saved best model: Acc={best_acc:.4f} at epoch {epoch}"
            print(log_msg)
            logger.info(log_msg)

    # CONFUSION MATRIX
    print("\nĐang tạo confusion matrix...")
    logger.info("Creating confusion matrix...")
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

    log_msg = f"Confusion matrix saved → {cm_path}"
    print(log_msg)
    logger.info(log_msg)
    log_msg = f"\nBest Accuracy: {best_acc:.4f}"
    print(log_msg)
    logger.info(log_msg)


if __name__ == "__main__":
    args = get_args()
    train_and_evaluate(args)