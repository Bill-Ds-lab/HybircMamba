import argparse
import csv
import gc
import os
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from Dataloader.loadDataset import TrafficSignDataset
from Dataloader.data_split_utils import get_or_create_split
from models.CNN_Mamba_CNN_Mamba_Enhanced.HybricMamba import HybricMamba
from models.mambaTRS.Vmamba_ultils import Super_Mamba
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark & So sánh các mô hình Traffic Sign")

    parser.add_argument('--dataset_name', default="NEU-DET_surface-dec", type=str,
                         choices=["German", "German_CSV", "Belgium", "German_51k", "NEU-DET_surface-dec"])
    parser.add_argument('--csv_filename', default="Train.csv", type=str)

    parser.add_argument(
        '--root_dataset_path',
        default=str(PROJECT_ROOT / "data" / "German_51k"),
        type=str
    )

    parser.add_argument(
        '--save_path',
        default=str(PROJECT_ROOT / "Ressult" / "TFJ"),
        type=str
    )
    parser.add_argument('--output_dir', default="./benchmark_outputs", type=str)

    parser.add_argument('--picture_size', default=32, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--SEED', default=2223, type=int)

    parser.add_argument('--models', nargs='+', default=[
        "LIGHT_HYBRIC_MAMBA",
        "MEDIUM_HYBRIC_MAMBA",
        "HEAVY_HYBRIC_MAMBA",
        "SUPER_MAMBA_DEPT_3",
        "SUPER_MAMBA_DEPT_4",
        "EFFICIENTNET_B0",
        "MOBILENETV3_SMALL",
        "RESNET18",
        "VIT_S",
        "GHOSTNET",
        "VGG16",
        "VIT_B",
    ])

    parser.add_argument('--skip_missing_checkpoint', action='store_true', default=True)
    parser.add_argument('--latency_batch_sizes', nargs='+', type=int, default=[1, 8, 16, 32])
    parser.add_argument('--n_warmup', default=30, type=int)
    parser.add_argument('--n_runs', default=200, type=int)
    parser.add_argument('--n_latency_repeats', default=5, type=int)

    return parser.parse_args()


def auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _friendly_cuda_error(e: Exception, device: str) -> str:
    msg = str(e)
    if "is_cuda" in msg or "CUDA" in msg:
        return f"Mamba selective-scan kernel yêu cầu GPU (device='{device}'). Lỗi gốc: {msg}"
    return msg


# --------------------------------------------------------------------------- #
# BUILD MODEL - KHỚP VỚI train.py
# --------------------------------------------------------------------------- #
def build_Model(name, num_classes=6, pretrained=False):
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
            ssm_frac=0.3,
            conv_frac=0.5,
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
    elif name in ["VGG16", "VGG-16"]:
        return timm.create_model("vgg16", pretrained=pretrained, num_classes=num_classes)
    elif name in ["RESNET18", "ResNet18"]:
        return timm.create_model("resnet18", pretrained=pretrained, num_classes=num_classes)
    elif name in ["VIT_B", "ViT-B"]:
        return timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes, img_size=32)
    elif name in ["VIT_S", "ViT-S"]:
        return timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=num_classes, img_size=32)
    elif name in ["EFFICIENTNET_B0", "EfficientNet-B0"]:
        return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
    elif name in ["MOBILENETV3_SMALL", "MobileNetV3-Small"]:
        return timm.create_model("mobilenetv3_small_100", pretrained=pretrained, num_classes=num_classes)
    elif name in ["GHOSTNET", "GhostNet"]:
        return timm.create_model("ghostnet_100", pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Tên mô hình '{name}' không hợp lệ.")


def load_checkpoint_safely(model, checkpoint_path, device):
    print(f"[LOAD] Nạp checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean_state_dict = {
        (k.replace("module.", "") if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)

    if missing_keys:
        print(f"   Thiếu {len(missing_keys)} keys (VD: {missing_keys[:3]})")
    if unexpected_keys:
        print(f"   Thừa {len(unexpected_keys)} keys (VD: {unexpected_keys[:3]})")
    if not missing_keys and not unexpected_keys:
        print("   Checkpoint khớp 100%.")

    best_val_acc = checkpoint.get("best_val_acc", None) if isinstance(checkpoint, dict) else None

    del checkpoint, state_dict, clean_state_dict
    return model, missing_keys, unexpected_keys, best_val_acc


def find_checkpoint(save_path, model_name, dataset_name):
    path = os.path.join(save_path, model_name, dataset_name, f"{model_name}_best.pth")
    return path if os.path.exists(path) else None


def get_transforms(img_size=32):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


# --------------------------------------------------------------------------- #
# TEST LOADER
# --------------------------------------------------------------------------- #

def build_eval_loaders(args):
    transform_test = get_transforms(args.picture_size)

    full_dataset = TrafficSignDataset(
        root=args.root_dataset_path,
        dataset_name=args.dataset_name,
        csv_filename=args.csv_filename,
        transform=transform_test,
        shuffle_samples=False,
    )
    num_classes = len(full_dataset.class_to_idx)
    dataset_class = type(full_dataset)

    full_loader = DataLoader(
        full_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    split_path_exists = os.path.exists(
        os.path.join(args.save_path, "_dataset_splits", f"{args.dataset_name}_split_seed{args.SEED}.json")
    )
    if not split_path_exists:
        print("\n" + "!" * 90)
        print(" Chưa có file split cố định cho dataset này.")
        print("    Split sẽ được TẠO MỚI ngay bây giờ và lưu lại cho các lần sau.")
        print("!" * 90 + "\n")

    _, _, test_samples,_ = get_or_create_split(
        full_dataset, args.save_path, args.dataset_name, seed=args.SEED
    )

    test_dataset = dataset_class(
        root=args.root_dataset_path,
        transform=transform_test,
        samples=test_samples,
        class_to_idx=full_dataset.class_to_idx,
        shuffle_samples=False,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"[DATA] Full dataset: {len(full_dataset)} ảnh | Test set: {len(test_dataset)} ảnh | {num_classes} classes")
    return full_loader, test_loader, num_classes

def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 ** 2)
    return {"total_params": total, "trainable_params": trainable, "size_mb": size_mb, "size_m": total / 1e6}


from fvcore.nn import FlopCountAnalysis


def selective_scan_flop_jit(inputs, outputs):
    try:
        if len(inputs) < 3:
            return 0
        u_shape = inputs[0].type().sizes()
        if u_shape is None or len(u_shape) < 3:
            return 0
        B, D, L = u_shape[0], u_shape[1], u_shape[2]
        a_shape = inputs[2].type().sizes()
        if a_shape is None or len(a_shape) < 1:
            return 0
        N = a_shape[1] if len(a_shape) > 1 else a_shape[0]
        flops = 9 * B * L * D * N + B * D * L
        return flops
    except Exception:
        return 0


def count_flops(model, input_size=(1, 3, 224, 224), device=None):
    if device is None:
        device = next(model.parameters()).device

    model = model.to(device).eval()
    dummy_input = torch.randn(*input_size, device=device)

    supported_ops = {
        "aten::silu": None,
        "aten::gelu": None,
        "aten::relu": None,
        "aten::sigmoid": None,
        "aten::exp": None,
        "aten::neg": None,
        "aten::add": None,
        "aten::mul": None,
        "aten::flip": None,
        "aten::view": None,
        "aten::permute": None,
        "aten::transpose": None,
        "aten::reshape": None,
        "prim::PythonOp.CrossScan": None,
        "prim::PythonOp.CrossMerge": None,
        "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
    }

    fca = FlopCountAnalysis(model, dummy_input)

    for op_name, handle in supported_ops.items():
        fca.set_op_handle(op_name, handle)

    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)

    total_flops = fca.total()

    print(f" Total FLOPs: {total_flops / 1e6:.2f}M ({total_flops / 1e9:.4f}G)")

    return {
        "flops": total_flops,
        "flops_m": total_flops / 1e6,
        "flops_g": total_flops / 1e9,
        "method": "fvcore"
    }


def measure_inference_time(model, input_size=(1, 3, 32, 32), device="cpu", n_warmup=30, n_runs=200) -> dict:
    model = model.to(device).eval()
    dummy_input = torch.randn(*input_size).to(device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    with torch.no_grad():
        for _ in range(n_warmup):
            try:
                _ = model(dummy_input)
            except RuntimeError as e:
                raise RuntimeError(_friendly_cuda_error(e, device)) from e

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        if device == "cuda":
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(n_runs):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))
        else:
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(dummy_input)
                times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    return {"median_ms": float(np.median(times))}


def measure_peak_memory(model, input_size=(1, 3, 32, 32), device="cuda") -> dict:
    if device != "cuda" or not torch.cuda.is_available():
        return {"peak_memory_mb": None}
    model = model.to(device).eval()
    dummy_input = torch.randn(*input_size).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {"peak_memory_mb": peak_mb}


def evaluate_accuracy(model, dataloader, device="cpu"):
    model = model.to(device).eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

            label_arr = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            all_labels.extend(label_arr)

    acc = accuracy_score(all_labels, all_preds) * 100
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm, model_name, save_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def compute_ids(accuracy: float, params_million: float) -> float:
    return accuracy / params_million if params_million > 0 else 0.0

"""
def benchmark_one_model(model_name, args, test_loader, num_classes, device):
    print(f"\n{'=' * 70}\n>>> BENCHMARK: {model_name}\n{'=' * 70}")
    result = {"model_name": model_name}

    model = build_Model(model_name, num_classes=num_classes, pretrained=False)
    params_info = count_parameters(model)
    result["params_m"] = params_info["size_m"]
    result["size_mb"] = params_info["size_mb"]

    ckpt_path = find_checkpoint(args.save_path, model_name, args.dataset_name)
    result["checkpoint_found"] = ckpt_path is not None

    if ckpt_path is not None:
        model, missing, unexpected, _ = load_checkpoint_safely(model, ckpt_path, device)
        result["arch_mismatch"] = bool(missing or unexpected)
    else:
        print(f"  ⚠️ Không tìm thấy checkpoint tại: {args.save_path}/{model_name}/{args.dataset_name}/{model_name}_best.pth")
        result["arch_mismatch"] = None

    model = model.to(device)

    try:
        flops_info = count_flops(model, input_size=(1, 3, args.picture_size, args.picture_size), device=device)
        result["flops_m"] = flops_info["flops_m"]
    except Exception as e:
        print(f"  FLOPs lỗi: {_friendly_cuda_error(e, device)}")
        result["flops_m"] = None

    result["latency_ms"] = {}
    for bs in args.latency_batch_sizes:
        try:
            medians = []
            for _ in range(args.n_latency_repeats):
                t_info = measure_inference_time(
                    model, input_size=(bs, 3, args.picture_size, args.picture_size),
                    device=device, n_warmup=args.n_warmup, n_runs=args.n_runs,
                )
                medians.append(t_info["median_ms"])
                time.sleep(0.05)
            result["latency_ms"][bs] = float(np.median(medians))
            print(f"  Latency (bs={bs}): {result['latency_ms'][bs]:.3f} ms")
        except RuntimeError as e:
            result["latency_ms"][bs] = None
            print(f"  Latency (bs={bs}) LỖI: {e}")

    try:
        mem_info = measure_peak_memory(model, input_size=(1, 3, args.picture_size, args.picture_size), device=device)
        result["peak_memory_mb"] = mem_info["peak_memory_mb"]
    except Exception as e:
        result["peak_memory_mb"] = None
        print(f"  Memory lỗi: {e}")

    if ckpt_path is not None and not result["arch_mismatch"]:
        try:
            acc_info = evaluate_accuracy(model, test_loader, device=device)
            result["accuracy"] = acc_info["accuracy"]
            result["precision_macro"] = acc_info["precision_macro"]
            result["recall_macro"] = acc_info["recall_macro"]
            result["f1_macro"] = acc_info["f1_macro"]
            result["f1_weighted"] = acc_info["f1_weighted"]
            result["ids"] = compute_ids(acc_info["accuracy"], params_info["size_m"])

            cm_path = plot_confusion_matrix(acc_info["confusion_matrix"], model_name, args.output_dir)
            result["confusion_matrix_path"] = cm_path
            print(
                f"  ✅ Test Acc: {acc_info['accuracy']:.2f}% | "
                f"Prec-Macro: {acc_info['precision_macro']:.2f}% | "
                f"Rec-Macro: {acc_info['recall_macro']:.2f}% | "
                f"F1-Macro: {acc_info['f1_macro']:.2f}% | "
                f"F1-Weighted: {acc_info['f1_weighted']:.2f}% | "
                f"IDS: {result['ids']:.2f}"
            )
        except Exception as e:
            print(f"  ❌ Lỗi khi tính toán metrics: {e}")
            result["accuracy"] = None
            result["precision_macro"] = None
            result["recall_macro"] = None
            result["f1_macro"] = None
            result["f1_weighted"] = None
    elif result["arch_mismatch"]:
        print("  ⚠️ Bỏ qua accuracy vì kiến trúc không khớp checkpoint.")
        result["accuracy"] = None
        result["precision_macro"] = None
        result["recall_macro"] = None
        result["f1_macro"] = None
        result["f1_weighted"] = None
    else:
        result["accuracy"] = None
        result["precision_macro"] = None
        result["recall_macro"] = None
        result["f1_macro"] = None
        result["f1_weighted"] = None

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return result
"""
def benchmark_one_model(model_name, args, full_loader, test_loader, num_classes, device):
    print(f"\n{'=' * 70}\n>>> BENCHMARK: {model_name}\n{'=' * 70}")
    result = {"model_name": model_name}

    model = build_Model(model_name, num_classes=num_classes, pretrained=False)
    params_info = count_parameters(model)
    result["params_m"] = params_info["size_m"]
    result["size_mb"] = params_info["size_mb"]

    ckpt_path = find_checkpoint(args.save_path, model_name, args.dataset_name)
    result["checkpoint_found"] = ckpt_path is not None

    if ckpt_path is not None:
        model, missing, unexpected, _ = load_checkpoint_safely(model, ckpt_path, device)
        result["arch_mismatch"] = bool(missing or unexpected)
    else:
        print(f"  ⚠️ Không tìm thấy checkpoint tại: {args.save_path}/{model_name}/{args.dataset_name}/{model_name}_best.pth")
        result["arch_mismatch"] = None

    model = model.to(device)

    try:
        flops_info = count_flops(model, input_size=(1, 3, args.picture_size, args.picture_size), device=device)
        result["flops_m"] = flops_info["flops_m"]
    except Exception as e:
        print(f"  FLOPs lỗi: {_friendly_cuda_error(e, device)}")
        result["flops_m"] = None

    result["latency_ms"] = {}
    for bs in args.latency_batch_sizes:
        try:
            medians = []
            for _ in range(args.n_latency_repeats):
                t_info = measure_inference_time(
                    model, input_size=(bs, 3, args.picture_size, args.picture_size),
                    device=device, n_warmup=args.n_warmup, n_runs=args.n_runs,
                )
                medians.append(t_info["median_ms"])
                time.sleep(0.05)
            result["latency_ms"][bs] = float(np.median(medians))
            print(f"  Latency (bs={bs}): {result['latency_ms'][bs]:.3f} ms")
        except RuntimeError as e:
            result["latency_ms"][bs] = None
            print(f"  Latency (bs={bs}) LỖI: {e}")

    try:
        mem_info = measure_peak_memory(model, input_size=(1, 3, args.picture_size, args.picture_size), device=device)
        result["peak_memory_mb"] = mem_info["peak_memory_mb"]
    except Exception as e:
        result["peak_memory_mb"] = None
        print(f"  Memory lỗi: {e}")

    if ckpt_path is not None and not result["arch_mismatch"]:
        try:
            full_acc_info = evaluate_accuracy(model, full_loader, device=device)
            result["full_accuracy"] = full_acc_info["accuracy"]

            acc_info = evaluate_accuracy(model, test_loader, device=device)
            result["test_accuracy"] = acc_info["accuracy"]
            result["precision_macro"] = acc_info["precision_macro"]
            result["recall_macro"] = acc_info["recall_macro"]
            result["f1_macro"] = acc_info["f1_macro"]
            result["f1_weighted"] = acc_info["f1_weighted"]
            result["ids"] = compute_ids(acc_info["accuracy"], params_info["size_m"])

            cm_path = plot_confusion_matrix(acc_info["confusion_matrix"], model_name, args.output_dir)
            result["confusion_matrix_path"] = cm_path

            print(
                f"  ✅ Full Dataset Acc: {result['full_accuracy']:.2f}% | "
                f"Test Acc: {result['test_accuracy']:.2f}% | "
                f"Prec-Macro: {acc_info['precision_macro']:.2f}% | "
                f"Rec-Macro: {acc_info['recall_macro']:.2f}% | "
                f"F1-Macro: {acc_info['f1_macro']:.2f}% | "
                f"F1-Weighted: {acc_info['f1_weighted']:.2f}% | "
                f"IDS: {result['ids']:.2f}"
            )
        except Exception as e:
            print(f"  ❌ Lỗi khi tính toán metrics: {e}")
            result["full_accuracy"] = None
            result["test_accuracy"] = None
            result["precision_macro"] = None
            result["recall_macro"] = None
            result["f1_macro"] = None
            result["f1_weighted"] = None
    elif result["arch_mismatch"]:
        print("  ⚠️ Bỏ qua accuracy vì kiến trúc không khớp checkpoint.")
        result["full_accuracy"] = None
        result["test_accuracy"] = None
        result["precision_macro"] = None
        result["recall_macro"] = None
        result["f1_macro"] = None
        result["f1_weighted"] = None
    else:
        result["full_accuracy"] = None
        result["test_accuracy"] = None
        result["precision_macro"] = None
        result["recall_macro"] = None
        result["f1_macro"] = None
        result["f1_weighted"] = None

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return result
"""
def main():
    datasetname = [
        "German",
        "Belgium",
        "German_51k",
        "NEU-DET_surface-dec",
        "DCID",
        "Belgium_ar"
    ]
    datasetpath = [
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/dataset_reOrgan",
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/Belgium_TFS",
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/German_51k",
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/NEU-DET",
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/DCID/DCID-512-35",
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/Belgium_ar"
    ]
    args = get_args()
    device = auto_device()
    for i in range(0, 6):
        args.__setattr__("dataset_name", datasetname[i])
        args.__setattr__("root_dataset_path", datasetpath[i])
        os.makedirs(args.output_dir, exist_ok=True)

        print("=" * 135)
        print(f"SO SÁNH MÔ HÌNH | Dataset: {args.dataset_name} | Device: {device}")
        print("=" * 135)

        test_loader, num_classes = build_test_loader(args)

        all_results = []
        for model_name in args.models:
            try:
                res = benchmark_one_model(model_name, args, test_loader, num_classes, device)
            except Exception as e:
                print(f"❌ LỖI với model {model_name}: {e}")
                res = {"model_name": model_name, "error": str(e)}
            all_results.append(res)

        print("\n" + "=" * 135)
        print("BẢNG TỔNG HỢP KẾT QUẢ")
        print("=" * 135)
        header = f"{'Model':<22s} | {'Params(M)':<9s} | {'FLOPs(M)':<9s} | {'Lat bs=1(ms)':<12s} | {'Mem(MB)':<8s} | {'Acc(%)':<7s} | {'Prec-Mac':<8s} | {'Rec-Mac':<8s} | {'F1-Mac':<8s} | {'F1-Wtd':<8s} | {'IDS':<7s}"
        print(header)
        print("-" * 135)

        csv_rows = []
        for r in all_results:
            params_m = f"{r.get('params_m', 0):.3f}" if r.get("params_m") is not None else "N/A"
            flops_m = f"{r.get('flops_m', 0):.2f}" if r.get("flops_m") is not None else "N/A"
            lat1 = r.get("latency_ms", {}).get(1) if r.get("latency_ms") else None
            lat1_str = f"{lat1:.3f}" if lat1 is not None else "N/A"
            mem = f"{r.get('peak_memory_mb'):.1f}" if r.get("peak_memory_mb") is not None else "N/A"
            acc = f"{r.get('accuracy'):.2f}" if r.get("accuracy") is not None else "N/A"
            prec_m = f"{r.get('precision_macro'):.2f}" if r.get("precision_macro") is not None else "N/A"
            rec_m = f"{r.get('recall_macro'):.2f}" if r.get("recall_macro") is not None else "N/A"
            f1m = f"{r.get('f1_macro'):.2f}" if r.get("f1_macro") is not None else "N/A"
            f1w = f"{r.get('f1_weighted'):.2f}" if r.get("f1_weighted") is not None else "N/A"
            ids = f"{r.get('ids'):.2f}" if r.get("ids") is not None else "N/A"

            print(
                f"{r['model_name']:<22s} | {params_m:<9s} | {flops_m:<9s} | {lat1_str:<12s} | {mem:<8s} | {acc:<7s} | {prec_m:<8s} | {rec_m:<8s} | {f1m:<8s} | {f1w:<8s} | {ids:<7s}"
            )

            csv_rows.append({
                "model_name": r["model_name"],
                "params_m": r.get("params_m"),
                "flops_m": r.get("flops_m"),
                **{f"latency_bs{bs}_ms": r.get("latency_ms", {}).get(bs) for bs in args.latency_batch_sizes},
                "peak_memory_mb": r.get("peak_memory_mb"),
                "accuracy": r.get("accuracy"),
                "precision_macro": r.get("precision_macro"),
                "recall_macro": r.get("recall_macro"),
                "f1_macro": r.get("f1_macro"),
                "f1_weighted": r.get("f1_weighted"),
                "ids": r.get("ids"),
                "checkpoint_found": r.get("checkpoint_found"),
                "arch_mismatch": r.get("arch_mismatch"),
            })

        print("=" * 135)

        csv_path = os.path.join(args.output_dir, f"benchmark_results_{args.dataset_name}.csv")
        if csv_rows:
            fieldnames = list(csv_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\nĐã lưu kết quả CSV: {csv_path}")
"""


def main():
    datasetname = [
        "German",
        "Belgium",
        "German_51k",
        "NEU-DET_surface-dec",
        "DCID",
        "Belgium_ar"
    ]
    datasetpath = [
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/dataset_reOrgan",
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/Belgium_TFS",
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/German_51k",
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/NEU-DET",
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/DCID/DCID-512-35",
        "/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/Belgium_ar"
    ]
    args = get_args()
    device = auto_device()
    for i in range(4, 5):
        args.__setattr__("dataset_name", datasetname[i])
        args.__setattr__("root_dataset_path", datasetpath[i])
        os.makedirs(args.output_dir, exist_ok=True)

        print("=" * 145)
        print(f"SO SÁNH MÔ HÌNH | Dataset: {args.dataset_name} | Device: {device}")
        print("=" * 145)

        # Lấy cả 2 loader
        full_loader, test_loader, num_classes = build_eval_loaders(args)

        all_results = []
        for model_name in args.models:
            try:
                res = benchmark_one_model(model_name, args, full_loader, test_loader, num_classes, device)
            except Exception as e:
                print(f"❌ LỖI với model {model_name}: {e}")
                res = {"model_name": model_name, "error": str(e)}
            all_results.append(res)

        print("\n" + "=" * 145)
        print("BẢNG TỔNG HỢP KẾT QUẢ")
        print("=" * 145)
        header = f"{'Model':<22s} | {'Params(M)':<9s} | {'FLOPs(M)':<9s} | {'Lat bs=1(ms)':<12s} | {'Mem(MB)':<8s} | {'Full Acc':<8s} | {'Test Acc':<8s} | {'Prec-Mac':<8s} | {'Rec-Mac':<8s} | {'F1-Mac':<8s} | {'F1-Wtd':<8s} | {'IDS':<7s}"
        print(header)
        print("-" * 145)

        csv_rows = []
        for r in all_results:
            params_m = f"{r.get('params_m', 0):.3f}" if r.get("params_m") is not None else "N/A"
            flops_m = f"{r.get('flops_m', 0):.2f}" if r.get("flops_m") is not None else "N/A"
            lat1 = r.get("latency_ms", {}).get(1) if r.get("latency_ms") else None
            lat1_str = f"{lat1:.3f}" if lat1 is not None else "N/A"
            mem = f"{r.get('peak_memory_mb'):.1f}" if r.get("peak_memory_mb") is not None else "N/A"

            full_acc = f"{r.get('full_accuracy'):.2f}" if r.get("full_accuracy") is not None else "N/A"
            test_acc = f"{r.get('test_accuracy'):.2f}" if r.get("test_accuracy") is not None else "N/A"

            prec_m = f"{r.get('precision_macro'):.2f}" if r.get("precision_macro") is not None else "N/A"
            rec_m = f"{r.get('recall_macro'):.2f}" if r.get("recall_macro") is not None else "N/A"
            f1m = f"{r.get('f1_macro'):.2f}" if r.get("f1_macro") is not None else "N/A"
            f1w = f"{r.get('f1_weighted'):.2f}" if r.get("f1_weighted") is not None else "N/A"
            ids = f"{r.get('ids'):.2f}" if r.get("ids") is not None else "N/A"

            print(
                f"{r['model_name']:<22s} | {params_m:<9s} | {flops_m:<9s} | {lat1_str:<12s} | {mem:<8s} | {full_acc:<8s} | {test_acc:<8s} | {prec_m:<8s} | {rec_m:<8s} | {f1m:<8s} | {f1w:<8s} | {ids:<7s}"
            )

            csv_rows.append({
                "model_name": r["model_name"],
                "params_m": r.get("params_m"),
                "flops_m": r.get("flops_m"),
                **{f"latency_bs{bs}_ms": r.get("latency_ms", {}).get(bs) for bs in args.latency_batch_sizes},
                "peak_memory_mb": r.get("peak_memory_mb"),
                "full_accuracy": r.get("full_accuracy"),
                "test_accuracy": r.get("test_accuracy"),
                "precision_macro": r.get("precision_macro"),
                "recall_macro": r.get("recall_macro"),
                "f1_macro": r.get("f1_macro"),
                "f1_weighted": r.get("f1_weighted"),
                "ids": r.get("ids"),
                "checkpoint_found": r.get("checkpoint_found"),
                "arch_mismatch": r.get("arch_mismatch"),
            })

        print("=" * 145)

        csv_path = os.path.join(args.output_dir, f"benchmark_results_{args.dataset_name}.csv")
        if csv_rows:
            fieldnames = list(csv_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\n Đã lưu kết quả CSV: {csv_path}")

if __name__ == "__main__":
    main()