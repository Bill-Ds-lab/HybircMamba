import argparse
import csv
import logging
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
from Dataloader.DATASET import TrafficSignDataset
from models.CNN_Mamba_CNN_Mamba_Enhanced.HybricMamba import HybricMamba
from models.vmamba.Vmamba_ultils import Super_Mamba


# =================================================================SET Args =======================================

def get_args():
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition Training with Mamba")

    parser.add_argument('--model_name', default="LIGHT_HYBRIC_MAMBA", type=str)
    # Hỗ trợ: "German" (folder root/<class>), "German_CSV" (root/Train.csv), "Belgium" (root/Train/<class> + root/Test/<class>)
    parser.add_argument('--dataset_name', default="German", type=str,
                        choices=["German", "German_CSV", "Belgium"])
    parser.add_argument('--csv_filename', default="Train.csv", type=str,
                        help="Chỉ dùng khi dataset_name=German_CSV")
    parser.add_argument('--class_num', default=43, type=int)
    parser.add_argument('--root_dataset_path',
                        default="/home/biu-linux/DeepLearning_Projects/DoAnNganh/dataset_reOrgan",
                        type=str)
    """"
    parser.add_argument('--save_path',
                        default="/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult/TFJ",
                        type=str)
                        """
    parser.add_argument('--save_path',
                        default="/kaggle/working/",
                        type=str)

    parser.add_argument('--picture_size', default=32, type=int)

    parser.add_argument('--early_stop_patience', default=30, type=int)
    parser.add_argument('--SEED', default=2223, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epoch', default=130, type=int)
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


# =================================================================BUILD MODEL ==============================================

def build_Model(name, num_classes, pretrained=True):
    if name == "LIGHT_HYBRIC_MAMBA":
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

    # 10 Mô hình Benchmark chuẩn
    elif name in ["VGG16", "VGG-16"]:
        return timm.create_model('vgg16', pretrained=pretrained, num_classes=num_classes)
    elif name in ["RESNET18", "ResNet18"]:
        return timm.create_model('resnet18', pretrained=pretrained, num_classes=num_classes)
    elif name in ["VIT_B", "ViT-B"]:
        return timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes,img_size=32)
    elif name in ["VIT_S", "ViT-S"]:
        return timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes,img_size=32)
    elif name in ["EFFICIENTNET_B0", "EfficientNet-B0"]:
        return timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)

    elif name in ["MOBILENETV3_SMALL", "MobileNetV3-Small"]:
        return timm.create_model('mobilenetv3_small_100', pretrained=pretrained, num_classes=num_classes)

    elif name in ["GHOSTNET", "GhostNet"]:
        return timm.create_model('ghostnet_100', pretrained=pretrained, num_classes=num_classes)

    else:
        raise ValueError(f"Tên mô hình '{name}' không tồn tại.")


# ================================== TRANSFORMS =================================================================

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


# ======================================== SET SEED =====================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================================== SET LOG ================================================

def setup_logging(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    log_file = os.path.join(folder_path, 'training.log')

    logger = logging.getLogger(folder_path)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# =================================== DATASET CLASS (DUY NHẤT) =================================================

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.ppm')



# =============================================================== DATA LOADER FACTORY ======================================================

def dataloader_prepare(full_dataset, dataset_name, root, batchsize, img_size=32, seed=42, logger=None):

    transform_train, transform_test = get_transforms(img_size)
    dataset_class = type(full_dataset)

    num_classes = len(full_dataset.class_to_idx)
    log_msg = f"Dataset: {dataset_name} | Số nhãn (classes): {num_classes} | Tổng số ảnh: {len(full_dataset)}"
    print(log_msg)
    if logger:
        logger.info(log_msg)

    # 2. Split dataset theo Stratified Split
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.class_to_idx[full_dataset.samples[i][1]] for i in indices]

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, shuffle=True, stratify=labels
    )
    temp_labels = [labels[i] for i in temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, shuffle=True, stratify=temp_labels
    )

    train_samples = [full_dataset.samples[i] for i in train_idx]
    val_samples = [full_dataset.samples[i] for i in val_idx]
    test_samples = [full_dataset.samples[i] for i in test_idx]

    # 3. Khởi tạo Dataset thành phần (chỉ copy samples có sẵn, không quét lại ổ đĩa)
    train_dataset = dataset_class(
        root=root, transform=transform_train, samples=train_samples,
        class_to_idx=full_dataset.class_to_idx, shuffle_samples=True
    )
    val_dataset = dataset_class(
        root=root, transform=transform_test, samples=val_samples,
        class_to_idx=full_dataset.class_to_idx, shuffle_samples=False
    )
    test_dataset = dataset_class(
        root=root, transform=transform_test, samples=test_samples,
        class_to_idx=full_dataset.class_to_idx, shuffle_samples=False
    )

    # 4. Tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)

    log_msg = f"Dữ liệu đã chia -> Train (70%): {len(train_dataset)} | Val (15%): {len(val_dataset)} | Test (15%): {len(test_dataset)}"
    print(log_msg)
    if logger:
        logger.info(log_msg)

    check_batch_distribution(train_loader, num_batches=5, logger=logger)

    return train_loader, val_loader, test_loader, num_classes


def check_batch_distribution(dataloader, num_batches=5, logger=None):
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


# =========================================================== LOAD CHECKPOINT ========================================================================
"""
def load_checkpoint_safely(model, checkpoint_path, device, logger=None, optimizer=None, scaler=None):
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
"""
def load_checkpoint_safely(
    model,
    checkpoint_path,
    device,
    logger=None,
    optimizer=None,
    scaler=None
):
    log_msg = f"\n[RESUME] Đang nạp checkpoint từ: {checkpoint_path}"
    print(log_msg)

    if logger:
        logger.info(log_msg)

    # =========================================================
    # LOAD CHECKPOINT TRÊN CPU
    # =========================================================
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    # =========================================================
    # LẤY MODEL STATE
    # =========================================================
    state_dict = None

    if isinstance(checkpoint, dict):
        for key in [
            'model_state_dict',
            'state_dict',
            'model',
            'net'
        ]:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # =========================================================
    # REMOVE "module."
    # =========================================================
    clean_state_dict = {}

    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        clean_state_dict[name] = v

    # =========================================================
    # LOAD MODEL WEIGHT
    # =========================================================
    missing_keys, unexpected_keys = model.load_state_dict(
        clean_state_dict,
        strict=False
    )

    if len(missing_keys) > 0 and logger:
        logger.warning(
            f"⚠️ Thiếu {len(missing_keys)} keys trong weights!"
        )

    if len(unexpected_keys) > 0 and logger:
        logger.warning(
            f"⚠️ Thừa {len(unexpected_keys)} keys không khớp mô hình!"
        )

    # =========================================================
    # CHỈ LOAD OPTIMIZER KHI RESUME TRAINING
    # =========================================================
    if (
        optimizer is not None
        and isinstance(checkpoint, dict)
        and 'optimizer_state_dict' in checkpoint
    ):
        try:
            optimizer.load_state_dict(
                checkpoint['optimizer_state_dict']
            )
        except Exception as e:
            if logger:
                logger.warning(
                    f"⚠️ Không thể khôi phục optimizer state: {e}"
                )

    # =========================================================
    # LOAD SCALER KHI RESUME TRAINING
    # =========================================================
    if (
        scaler is not None
        and isinstance(checkpoint, dict)
        and 'scaler_state_dict' in checkpoint
    ):
        try:
            scaler.load_state_dict(
                checkpoint['scaler_state_dict']
            )
        except Exception as e:
            if logger:
                logger.warning(
                    f"⚠️ Không thể khôi phục scaler state: {e}"
                )

    start_epoch = (
        checkpoint.get('epoch', -1) + 1
        if isinstance(checkpoint, dict)
        else 0
    )

    best_val_acc = (
        checkpoint.get('best_val_acc', 0.0)
        if isinstance(checkpoint, dict)
        else 0.0
    )

    del checkpoint
    del state_dict
    del clean_state_dict

    return model, start_epoch, best_val_acc

# ================================================================== LEARNING RATE SCHEDULE =========================================================

def get_lr(epoch, base_lr=1e-3, min_lr=1e-6):
    if epoch < 5:
        return base_lr * (epoch + 1) / 5
    elif epoch < 30:
        lr = base_lr
    elif epoch < 50:
        lr = base_lr * 0.1
    elif epoch < 75:
        lr = base_lr * 0.01
    else:
        lr = base_lr * 0.001

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
    #base_lr = args.finetune_lr if args.resume and os.path.exists(args.resume_path) else args.lr

    base_lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)

    if args.resume and os.path.exists(args.resume_path):
        model, start_epoch, best_val_acc = load_checkpoint_safely(
            model, args.resume_path, device, logger, optimizer=optimizer, scaler=scaler
        )

    best_checkpoint_path = os.path.join(folder_path, f"{model_name}_best.pth")
    patience_counter = 0
    early_stop_patience=args.early_stop_patience
    for epoch in range(start_epoch, args.num_epoch):
        current_lr = get_lr(epoch, base_lr)
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
            if patience_counter >= early_stop_patience:
                logger.info(f"⏹ Early stopping tại epoch {epoch} "
                            f"(không cải thiện sau {early_stop_patience} epoch, best={best_val_acc:.4f})")
                break



    logger.info("Evaluating Best Model on TEST Set...")
    if os.path.exists(best_checkpoint_path):
        model, _, _ = load_checkpoint_safely(model, best_checkpoint_path, device, logger)

    model.eval()
    test_correct, test_total, running_test_loss = 0, 0, 0.0
    true_labels, predicted_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            if isinstance(outputs, tuple): outputs = outputs[0]

            loss = criterion(outputs, labels)
            running_test_loss += loss.item()
            predicted = outputs.argmax(dim=1)

            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    test_acc = test_correct / max(test_total, 1)
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')

    logger.info(f"🎯 [TEST RESULT] Test Acc: {test_acc:.4f} | F1-Macro: {f1_macro:.4f} | F1-Weighted: {f1_weighted:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(20, 18))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    cm_path = os.path.join(folder_path, "confusion_matrix_test.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()


# ======================================================================== MAIN =================================================

if __name__ == "__main__":
    modelname = [
        "LIGHT_HYBRIC_MAMBA",
        "MEDIUM_HYBRIC_MAMBA",
        "HEAVY_HYBRIC_MAMBA",
        "SUPER_MAMBA_DEPT_3",
        "SUPER_MAMBA_DEPT_4",
        "VGG16",
        "RESNET18",
        "VIT_B",
        "VIT_S",
        "EFFICIENTNET_B0",
        "MOBILENETV3_SMALL",
        "GHOSTNET",
    ]
    datasetname = [
        "German",
        "Belgium",
        "German_51k",
        "NEU-DET_surface-dec"
    ]
    datasetpath=[
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/dataset_reOrgan",
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/Belgium_TFS",
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/German_51k",
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/NEU-DET",
        "/kaggle/input/datasets/thanhsangtrn/german-trafic-sign/dataset_reOrgan"
    ]

    args = get_args()
    for i in range(0,12,1):
        args.__setattr__("model_name", modelname[i])

        args.__setattr__("dataset_name", datasetname[0])
        args.__setattr__("root_dataset_path", datasetpath[4])
        args.__setattr__("batch_size", 526)
        args.__setattr__("img_size", 32)
        args.__setattr__("class_num", 43)
        args.__setattr__("num_epoch", 100)

        args.__setattr__("model_name", modelname[i])

        args.__setattr__("resume_path",
                         os.path.join(
                             args.save_path,
                             args.model_name,
                             args.dataset_name,
                             f"{args.model_name}_best.pth"
                         )
                         )

        folder_path = os.path.join(args.save_path, args.model_name, args.dataset_name)
        logger = setup_logging(folder_path)

        full_dataset = TrafficSignDataset(
            root=args.root_dataset_path,
            dataset_name=args.dataset_name,
            csv_filename=args.csv_filename,
            shuffle_samples=True
        )

        train_loader, val_loader, test_loader, num_classes = dataloader_prepare(
            full_dataset=full_dataset,
            dataset_name=args.dataset_name,
            root=args.root_dataset_path,
            batchsize=args.batch_size,
            img_size=args.picture_size,
            seed=args.SEED,
            logger=logger
        )

        model = build_Model(
            name=args.model_name,
            num_classes=num_classes,
            pretrained=True
        )

        train_and_evaluate(
            args=args,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            logger=logger
        )