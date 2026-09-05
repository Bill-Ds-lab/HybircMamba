import argparse
import csv
import io
import logging
import math
import os
import random
import re
import sys

# =================================================================
# 1. BỔ SUNG ĐƯỜNG DẪN MODULE TRÊN KAGGLE
# =================================================================
sys.path.append("/kaggle/working")
if os.path.exists("/kaggle/input"):
    sys.path.append("/kaggle/input")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import TensorFlow để đọc file .tfrec / .tfrecord trên Kaggle
try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from Dataloader.DATASET import TrafficSignDataset
from models.CNN_Mamba_CNN_Mamba_Enhanced.HybricMamba import HybricMamba
from models.vmamba.Vmamba_ultils import Super_Mamba


# ================================================================= SET ARGS =======================================

def get_args():
    parser = argparse.ArgumentParser(description="ImageNet-1k Training with PyTorch on Kaggle")

    parser.add_argument('--model_name', default="RESNET18", type=str)
    parser.add_argument('--dataset_name', default="ImageNet-1k", type=str)
    parser.add_argument('--csv_filename', default="", type=str)
    parser.add_argument('--class_num', default=1000, type=int)

    # 📌 Đường dẫn dataset ImageNet-1k TFRecords trên Kaggle
    parser.add_argument('--root_dataset_path',
                        default="/kaggle/input/imagenet-1k-tfrecords-ilsvrc2012-part-0",
                        type=str)

    # 📌 Thư mục lưu checkpoint & logs trên Kaggle Working
    parser.add_argument('--save_path',
                        default="/kaggle/working/Result",
                        type=str)

    parser.add_argument('--picture_size', default=224, type=int)
    parser.add_argument('--early_stop_patience', default=10, type=int)
    parser.add_argument('--SEED', default=2223, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.02, type=float)
    parser.add_argument('--clip_grad', default=5.0, type=float)
    parser.add_argument('--label_smoothing', default=0.1, type=float)

    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_path', default="", type=str)

    # Giải quyết lỗi parse_args trên Jupyter Notebook
    args, _ = parser.parse_known_args()
    return args


# ================================================================= BUILD MODEL ==============================================

def build_Model(name, num_classes=1000, pretrained=True, img_size=224):
    if name == "LIGHT_HYBRIC_MAMBA":
        return HybricMamba(
            dims=(3, 16, 32, 56, 96),
            num_classes=num_classes,
            mbconv_expand_ratio=4,
            ssm_d_state=8,
            mamba_blocks=(1, 1),
            ssm_frac=0.7,
            conv_frac=0.2,
            use_aux=True,
        )
    elif name == "MEDIUM_HYBRIC_MAMBA":
        return HybricMamba(
            dims=(3, 24, 48, 80, 128),
            num_classes=num_classes,
            mbconv_expand_ratio=4,
            ssm_d_state=12,
            ssm_ratio=1.5,
            mamba_blocks=(2, 2),
            cnn_blocks=(1, 2),
            ssm_frac=0.6,
            conv_frac=0.25,
            use_aux=True,
        )
    elif name == "HEAVY_HYBRIC_MAMBA":
        return HybricMamba(
            dims=(3, 32, 64, 112, 176),
            num_classes=num_classes,
            mbconv_expand_ratio=6,
            ssm_d_state=16,
            ssm_ratio=2.0,
            mamba_blocks=(2, 3),
            cnn_blocks=(2, 2),
            ssm_frac=0.7,
            conv_frac=0.2,
            use_aux=True,
        )
    elif name == "SUPER_MAMBA_DEPT_4":
        return Super_Mamba(dims=3, depth=4, num_classes=num_classes)
    elif name == "SUPER_MAMBA_DEPT_3":
        return Super_Mamba(dims=3, depth=3, num_classes=num_classes)

    # Benchmarks chuẩn cho ImageNet
    elif name in ["VGG16", "VGG-16"]:
        return timm.create_model('vgg16', pretrained=pretrained, num_classes=num_classes)
    elif name in ["RESNET18", "ResNet18"]:
        return timm.create_model('resnet18', pretrained=pretrained, num_classes=num_classes)
    elif name in ["VIT_B", "ViT-B"]:
        return timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes,
                                 img_size=img_size)
    elif name in ["VIT_S", "ViT-S"]:
        return timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes,
                                 img_size=img_size)
    elif name in ["EFFICIENTNET_B0", "EfficientNet-B0"]:
        return timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
    elif name in ["MOBILENETV3_SMALL", "MobileNetV3-Small"]:
        return timm.create_model('mobilenetv3_small_100', pretrained=pretrained, num_classes=num_classes)
    elif name in ["GHOSTNET", "GhostNet"]:
        return timm.create_model('ghostnet_100', pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Tên mô hình '{name}' không tồn tại.")


# ================================== TRANSFORMS CHUẨN IMAGENET =================================================

def get_transforms(img_size=224):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return transform_train, transform_test


# ================================= DATASET LOADER DÀNH CHO TFRECORDS =================================

class ImageNetTFRecordDataset(Dataset):
    """Dataset đọc trực tiếp định dạng TFRecord trên Kaggle sang PyTorch Tensor"""

    def __init__(self, root_dir, transform=None, max_samples=None, desc="Loading TFRecords"):
        self.transform = transform
        self.samples = []

        if not TF_AVAILABLE:
            raise ImportError("Vui lòng cài đặt TensorFlow để giải mã TFRecords!")

        tfrec_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.tfrec') or file.endswith('.tfrecord'):
                    tfrec_files.append(os.path.join(root, file))

        if not tfrec_files:
            raise FileNotFoundError(f"Không tìm thấy file .tfrec/.tfrecord trong {root_dir}")

        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
        }

        # Đọc dữ liệu từ TFRecords
        raw_dataset = tf.data.TFRecordDataset(tfrec_files)
        for raw_record in tqdm(raw_dataset, desc=desc):
            example = tf.io.parse_single_example(raw_record, feature_description)
            img_bytes = example['image/encoded'].numpy()
            label = int(example['image/class/label'].numpy())

            # Normalize label về khoảng 0-999
            if label > 0 and label <= 1000:
                label -= 1

            self.samples.append((img_bytes, label))
            if max_samples and len(self.samples) >= max_samples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_bytes, label = self.samples[idx]
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# ======================================== SET SEED & LOGS =====================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    log_file = os.path.join(folder_path, 'training.log')

    logger = logging.getLogger(folder_path)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# =============================================================== DATA LOADER PREPARE ======================================================

def dataloader_prepare(root_path, batchsize, img_size=224, seed=42, logger=None):
    transform_train, transform_test = get_transforms(img_size)

    # 📌 Tự động xác định đường dẫn train và validation
    train_dir = os.path.join(root_path, "train")
    val_dir = os.path.join(root_path, "validation")

    # Kiểm tra cấu trúc nếu thư mục gốc không chứa trực tiếp train/validation
    if not os.path.exists(train_dir):
        for root, dirs, _ in os.walk(root_path):
            if "train" in dirs and "validation" in dirs:
                train_dir = os.path.join(root, "train")
                val_dir = os.path.join(root, "validation")
                break

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Không tìm thấy tập train/validation hợp lệ trong {root_path}")

    # Đọc trực tiếp dữ liệu tập Train và Validation
    train_dataset = ImageNetTFRecordDataset(
        root_dir=train_dir,
        transform=transform_train,
        desc="Loading Train TFRecords"
    )
    val_dataset = ImageNetTFRecordDataset(
        root_dir=val_dir,
        transform=transform_test,
        desc="Loading Val TFRecords"
    )

    # Tập Test sử dụng chung bộ dữ liệu Validation
    test_dataset = ImageNetTFRecordDataset(
        root_dir=val_dir,
        transform=transform_test,
        desc="Loading Test TFRecords"
    )

    num_classes = 1000

    log_msg = f"Dataset ImageNet-1k | Số nhãn: {num_classes} | Train: {len(train_dataset)} | Val: {len(val_dataset)}"
    print(log_msg)
    if logger:
        logger.info(log_msg)

    # 📌 Giới hạn num_workers để tối ưu tài nguyên CPU Kaggle
    num_workers = min(4, os.cpu_count() or 2)

    train_loader = DataLoader(
        train_dataset, batch_size=batchsize, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, num_classes


# =========================================================== LOAD CHECKPOINT ========================================================================

def load_checkpoint_safely(model, checkpoint_path, device, logger=None, optimizer=None, scaler=None):
    log_msg = f"\n[RESUME] Đang nạp checkpoint từ: {checkpoint_path}"
    print(log_msg)
    if logger:
        logger.info(log_msg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ['model_state_dict', 'state_dict', 'model', 'net']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean_state_dict = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)

    if len(missing_keys) > 0 and logger:
        logger.warning(f"⚠️ Thiếu {len(missing_keys)} keys trong weights!")
    if len(unexpected_keys) > 0 and logger:
        logger.warning(f"⚠️ Thừa {len(unexpected_keys)} keys không khớp mô hình!")

    if optimizer is not None and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            if logger: logger.warning(f"⚠️ Không thể khôi phục optimizer state: {e}")

    if scaler is not None and isinstance(checkpoint, dict) and 'scaler_state_dict' in checkpoint:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except Exception as e:
            if logger: logger.warning(f"⚠️ Không thể khôi phục scaler state: {e}")

    start_epoch = checkpoint.get('epoch', -1) + 1 if isinstance(checkpoint, dict) else 0
    best_val_acc = checkpoint.get('best_val_acc', 0.0) if isinstance(checkpoint, dict) else 0.0

    return model, start_epoch, best_val_acc


# ================================================================== LEARNING RATE SCHEDULE =========================================================

def get_lr(epoch, base_lr=1e-3, min_lr=1e-6, total_epochs=50):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs

    progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
    return max(lr, min_lr)


# ======================================================================== TRAIN AND VAL ===============================================

def train_and_evaluate(args, model, train_loader, val_loader, test_loader, logger):
    set_seed(args.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.model_name
    folder_path = os.path.join(args.save_path, model_name, args.dataset_name)
    os.makedirs(folder_path, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Starting training: {model_name}")
    logger.info(f"Dataset: {args.dataset_name} | Path: {args.root_dataset_path}")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    start_epoch = 0
    best_val_acc = 0.0
    base_lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)

    if args.resume and os.path.exists(args.resume_path):
        model, start_epoch, best_val_acc = load_checkpoint_safely(
            model, args.resume_path, device, logger, optimizer=optimizer, scaler=scaler
        )

    best_checkpoint_path = os.path.join(folder_path, f"{model_name}_best.pth")
    patience_counter = 0

    for epoch in range(start_epoch, args.num_epoch):
        current_lr = get_lr(epoch, base_lr, min_lr=args.min_lr, total_epochs=args.num_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        model.train()
        running_train_loss, train_correct, train_total, num_train_batches = 0.0, 0, 0, 0

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
                        logits = output
                        loss = criterion(logits, target)

                if not torch.isfinite(loss):
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

                scaler.step(optimizer)
                scaler.update()

                running_train_loss += loss.item()
                preds = logits.argmax(dim=1)
                train_correct += (preds == target).sum().item()
                train_total += target.size(0)
                num_train_batches += 1

                tepoch.set_postfix(
                    train_loss=f"{running_train_loss / max(num_train_batches, 1):.4f}",
                    train_acc=f"{train_correct / max(train_total, 1):.4f}",
                    lr=f"{current_lr:.6f}",
                    gn=f"{grad_norm:.2f}"
                )

        train_loss = running_train_loss / max(num_train_batches, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validation
        model.eval()
        val_correct, val_total, running_val_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                preds = model(x)
                if isinstance(preds, tuple):
                    preds = preds[0]
                loss = criterion(preds, y)
                running_val_loss += loss.item()
                val_correct += (preds.argmax(dim=1) == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / max(val_total, 1)
        val_loss = running_val_loss / max(len(val_loader), 1)

        log_msg = f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        print(log_msg)
        logger.info(log_msg)

        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_acc': max(best_val_acc, val_acc),
        }
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint_state['best_val_acc'] = best_val_acc
            torch.save(checkpoint_state, best_checkpoint_path)
            logger.info(f"✓ Saved best model: Val Acc={best_val_acc:.4f} at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                logger.info(
                    f"⏹ Early stopping tại epoch {epoch} (không cải thiện sau {args.early_stop_patience} epoch, best={best_val_acc:.4f})")
                break

    # Evaluate on Test Set
    logger.info("Evaluating Best Model on TEST Set...")
    if os.path.exists(best_checkpoint_path):
        model, _, _ = load_checkpoint_safely(model, best_checkpoint_path, device, logger)

    model.eval()
    test_correct, test_total = 0, 0
    true_labels, predicted_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            if isinstance(outputs, tuple): outputs = outputs[0]

            predicted = outputs.argmax(dim=1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    test_acc = test_correct / max(test_total, 1)
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')

    logger.info(f"🎯 [TEST RESULT] Test Acc: {test_acc:.4f} | F1-Macro: {f1_macro:.4f} | F1-Weighted: {f1_weighted:.4f}")


# ======================================================================== MAIN =================================================

if __name__ == "__main__":
    models_to_train = [
        "HEAVY_HYBRIC_MAMBA",
        "MEDIUM_HYBRIC_MAMBA",
        "LIGHT_HYBRIC_MAMBA",
    ]

    args = get_args()

    # 📌 TỰ ĐỘNG KHỞI TẠO VÀ TÌM KIẾM ĐƯỜNG DẪN DATASET CHÍNH XÁC TRÊN KAGGLE
    input_path = "/kaggle/input"
    if os.path.exists(input_path):
        available_dirs = os.listdir(input_path)
        matched_dir = next((d for d in available_dirs if "imagenet" in d.lower()), None)
        if matched_dir:
            args.root_dataset_path = os.path.join(input_path, matched_dir)
            print(f"-> Đã tự động phát hiện đường dẫn Dataset: {args.root_dataset_path}")

    args.save_path = "/kaggle/working/Result"
    args.dataset_name = "ImageNet-1k"
    args.class_num = 1000
    args.picture_size = 224
    args.batch_size = 64
    args.num_epoch = 50

    for model_name in models_to_train:
        args.model_name = model_name
        args.resume_path = os.path.join(
            args.save_path,
            args.model_name,
            args.dataset_name,
            f"{args.model_name}_best.pth"
        )

        folder_path = os.path.join(args.save_path, args.model_name, args.dataset_name)
        logger = setup_logging(folder_path)

        train_loader, val_loader, test_loader, num_classes = dataloader_prepare(
            root_path=args.root_dataset_path,
            batchsize=args.batch_size,
            img_size=args.picture_size,
            seed=args.SEED,
            logger=logger
        )

        model = build_Model(
            name=args.model_name,
            num_classes=num_classes,
            pretrained=True,
            img_size=args.picture_size
        )

        train_and_evaluate(
            args=args,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            logger=logger
        )