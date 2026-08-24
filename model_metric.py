
import time
import numpy as np
import torch
import torch.nn as nn

from models.CNN_Mamba_CNN_Mamba_Enhanced.HybricMamba import HybricMamba
from models.vmamba.Vmamba_ultils import Super_Mamba


def auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _friendly_cuda_error(e: Exception, device: str) -> str:

    msg = str(e)
    if "is_cuda" in msg or "CUDA" in msg:
        return (
            f"Model này dùng Mamba selective-scan CUDA kernel nên chỉ chạy "
            f"được trên GPU, không chạy được trên CPU (device='{device}'). "
            f"Hãy gọi lại với device='cuda' (cần máy có GPU + driver CUDA "
            f"phù hợp với bản mamba_ssm đã cài). Lỗi gốc: {msg}"
        )
    return msg


# --------------------------------------------------------------------------- #
# 1. SỐ LƯỢNG THAM SỐ
# --------------------------------------------------------------------------- #
def count_parameters(model: nn.Module) -> dict:
    """
    Trả về tổng số tham số, số tham số trainable, và kích thước ước tính (MB, float32).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 ** 2)  # float32 = 4 bytes

    result = {
        "total_params": total,
        "trainable_params": trainable,
        "size_mb": size_mb,
        "size_m": total / 1e6,  # đơn vị "M" như trong paper (90k -> 0.09M)
    }
    return result


def count_by_top_module(model: nn.Module) -> dict:
    """Phân bố số tham số theo từng module con (top-level)."""
    breakdown = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        breakdown[name] = n
    return breakdown


# --------------------------------------------------------------------------- #
# 2. FLOPs
# --------------------------------------------------------------------------- #
def count_flops(model: nn.Module, input_size=(1, 3, 32, 32), device="cpu") -> dict:
    """
    Tính FLOPs của model. Ưu tiên dùng `thop`, nếu không có thì thử `fvcore`.
    Trả về dict gồm flops (đơn vị FLOPs thô) và các đơn vị quy đổi (K/M/G).

    Lưu ý: thop trả về "MACs" nhưng thực chất báo là flops (tuỳ version),
    một số paper quy ước FLOPs = 2 * MACs -> tự kiểm tra lại nếu cần khớp
    số liệu công bố.
    """
    model = model.to(device).eval()
    dummy_input = torch.randn(*input_size).to(device)

    try:
        from thop import profile
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        method = "thop"
    except Exception as e1:
        try:
            from fvcore.nn import FlopCountAnalysis
            fca = FlopCountAnalysis(model, dummy_input)
            flops = fca.total()
            params = sum(p.numel() for p in model.parameters())
            method = "fvcore"
        except Exception as e2:
            raise RuntimeError(
                f"Không thể tính FLOPs. Lỗi thop: {e1}; Lỗi fvcore: {e2}. "
                f"Hãy cài: pip install thop fvcore --break-system-packages"
            )

    return {
        "flops": flops,
        "flops_k": flops / 1e3,
        "flops_m": flops / 1e6,
        "flops_g": flops / 1e9,
        "params_from_flops_tool": params,
        "method": method,
    }


# --------------------------------------------------------------------------- #
# 3. THỜI GIAN SUY LUẬN (INFERENCE TIME)
# --------------------------------------------------------------------------- #
def measure_inference_time(
    model: nn.Module,
    input_size=(1, 3, 32, 32),
    device="cpu",
    n_warmup=100,
    n_runs=800,
) -> dict:
    """
    Đo thời gian suy luận trung bình cho 1 batch (mặc định batch=1, giống
    "single-frame inference time" trong paper).

    Trên GPU dùng torch.cuda.Event để đo chính xác (tránh nhiễu do
    Python/CUDA kernel-launch overhead so với time.perf_counter thô).
    """
    model = model.to(device).eval()
    dummy_input = torch.randn(*input_size).to(device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # warm-up để tránh sai lệch do khởi tạo CUDA context / cache / cuDNN autotune
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
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(n_runs):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))  # ms
        else:
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms

    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "median_ms": float(np.median(times)),
        "n_runs": n_runs,
        "device": device,
    }


# --------------------------------------------------------------------------- #
# 4. ACCURACY + CONFUSION MATRIX
# --------------------------------------------------------------------------- #
def evaluate_accuracy(model: nn.Module, dataloader, device="cpu", class_names=None):
    """
    Chạy model qua toàn bộ dataloader (test set) và tính accuracy + confusion matrix.

    dataloader: phải trả về (images, labels) theo batch.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix

    model = model.to(device).eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "y_true": all_labels,
        "y_pred": all_preds,
        "class_names": class_names,
    }


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """Vẽ confusion matrix bằng seaborn (giống Fig.6-8 trong paper)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


# --------------------------------------------------------------------------- #
# 5. INFORMATION DENSITY (IDS)
# --------------------------------------------------------------------------- #
def compute_ids(accuracy: float, params_million: float) -> float:
    """
    IDS = ACC(%) / params(M)

    Lưu ý quan trọng: công thức gốc trong paper MambaTSR ghi là
    IDS = ACC*100/np với "np là tham số tính theo MB", NHƯNG khi đối
    chiếu ngược lại số liệu thực tế họ công bố (VD: German, ACC=99.00,
    params~0.088M -> IDS=1123.72), số liệu chỉ khớp khi np là số tham số
    tính theo ĐƠN VỊ TRIỆU (M), không phải MB thực tế (byte). Vì vậy hàm
    này dùng params tính theo M để tái tạo đúng số liệu paper.

    accuracy: đơn vị %, ví dụ 99.0 (không phải 0.99)
    params_million: số tham số / 1e6 (dùng field 'size_m' từ count_parameters)
    """
    return accuracy / params_million


# --------------------------------------------------------------------------- #
# 6. ROBUSTNESS TEST (dim / exposure / noise / rain)
# --------------------------------------------------------------------------- #
def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Mô phỏng cảnh tối/sáng bằng gamma correction. image: HWC, uint8 hoặc float [0,255]."""
    inv = 1.0 if gamma == 0 else gamma
    img = np.clip(image, 0, 255).astype(np.float32)
    out = np.power(img / 255.0, inv) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def add_gaussian_noise(image: np.ndarray, mean: float = 0, std: float = 25) -> np.ndarray:
    """Thêm nhiễu Gaussian cộng tính."""
    img = image.astype(np.float32)
    noise = np.random.normal(mean, std, img.shape)
    out = img + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_rain_effect(image: np.ndarray, size_factor: float = 3.0,
                     angle_range=(-60, 60), density: float = 0.02) -> np.ndarray:
    """
    Mô phỏng hiệu ứng giọt mưa đơn giản bằng các vệt trắng ngẫu nhiên.
    (Bản đơn giản hoá; paper gốc dùng thuật toán riêng, xem repo MambaTSR/Rain).
    """
    import cv2
    img = image.copy()
    h, w = img.shape[:2]
    n_drops = int(h * w * density)

    for _ in range(n_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = int(size_factor * np.random.uniform(2, 5))
        angle = np.deg2rad(np.random.uniform(*angle_range))
        x2 = int(x + length * np.sin(angle))
        y2 = int(y + length * np.cos(angle))
        cv2.line(img, (x, y), (x2, y2), (255, 255, 255), 1)

    return img


def robustness_test(model: nn.Module, dataloader_factory, device="cpu",
                     scenarios: dict = None) -> dict:
    if scenarios is None:
        scenarios = {
            "dim_gamma_0.5": lambda img: adjust_gamma(img, 0.5),
            "dim_gamma_0.2": lambda img: adjust_gamma(img, 0.2),
            "exposure_gamma_1.5": lambda img: adjust_gamma(img, 1.5),
            "exposure_gamma_1.8": lambda img: adjust_gamma(img, 1.8),
            "rain_s3": lambda img: add_rain_effect(img, size_factor=3.0),
            "rain_s7": lambda img: add_rain_effect(img, size_factor=7.0),
            "noise_mean0": lambda img: add_gaussian_noise(img, mean=0, std=25),
            "noise_mean-120": lambda img: add_gaussian_noise(img, mean=-120, std=25),
            "noise_mean120": lambda img: add_gaussian_noise(img, mean=120, std=25),
        }

    results = {}
    for name, transform in scenarios.items():
        loader = dataloader_factory(transform)
        res = evaluate_accuracy(model, loader, device=device)
        results[name] = res["accuracy"]

    return results


# --------------------------------------------------------------------------- #
# 7. GRAD-CAM (activation map)
# --------------------------------------------------------------------------- #
def plot_gradcam(model: nn.Module, target_layer, image_tensor: torch.Tensor,
                  original_image: np.ndarray, device="cpu", save_path=None):
    """
    Vẽ activation map bằng Grad-CAM (giống Fig.14 trong paper).

    target_layer: layer cụ thể trong model để trích xuất gradient
                  (thường là layer conv/mamba cuối cùng trước classifier).
    image_tensor: ảnh đã qua preprocess, shape (1, C, H, W)
    original_image: ảnh gốc dạng numpy HWC, float [0,1], dùng để overlay
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import matplotlib.pyplot as plt

    model = model.to(device).eval()
    image_tensor = image_tensor.to(device)

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor)[0]
    visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return visualization


# --------------------------------------------------------------------------- #
# 8. BÁO CÁO TỔNG HỢP
# --------------------------------------------------------------------------- #
def full_report(model: nn.Module, model_name: str = "Model",
                 input_size=(1, 3, 32, 32), device=None,
                 accuracy: float = None) -> dict:
    """
    Chạy các phép đo cơ bản (params, FLOPs, inference time) và in báo cáo.
    accuracy: nếu đã có sẵn (từ evaluate_accuracy), sẽ tính luôn IDS.

    device: nếu để None sẽ tự chọn 'cuda' nếu máy có GPU, ngược lại 'cpu'.
            Với các model Mamba dùng CUDA kernel (selective_scan_cuda_core),
            BẮT BUỘC phải chạy trên 'cuda', không dùng được 'cpu'.
    """
    if device is None:
        device = auto_device()

    print(f"\n{'=' * 60}")
    print(f"BÁO CÁO ĐÁNH GIÁ: {model_name}  (device={device})")
    print(f"{'=' * 60}")

    # 1. Params
    params_info = count_parameters(model)
    print(f"\n[1] Tham số:")
    print(f"    Tổng           : {params_info['total_params']:,} "
          f"({params_info['size_m']:.3f} M)")
    print(f"    Trainable      : {params_info['trainable_params']:,}")
    print(f"    Kích thước     : {params_info['size_mb']:.3f} MB")

    breakdown = count_by_top_module(model)
    if breakdown:
        print(f"    Phân bố module :")
        for name, n in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            pct = 100 * n / params_info["total_params"] if params_info["total_params"] else 0
            print(f"      - {name:<20s}: {n:>10,} ({pct:5.1f}%)")

    # 2. FLOPs
    try:
        flops_info = count_flops(model, input_size=input_size, device=device)
        print(f"\n[2] FLOPs:")
        print(f"    {flops_info['flops_m']:.3f} M  (phương pháp: {flops_info['method']})")
    except Exception as e:
        flops_info = None
        print(f"\n[2] FLOPs: Không tính được\n    {_friendly_cuda_error(e, device)}")

    # 3. Inference time
    try:
        time_info = measure_inference_time(model, input_size=input_size, device=device)
        print(f"\n[3] Thời gian suy luận ({device}):")
        print(f"    Trung bình     : {time_info['mean_ms']:.3f} ms "
              f"(± {time_info['std_ms']:.3f}, {time_info['n_runs']} lần chạy)")
    except RuntimeError as e:
        time_info = None
        print(f"\n[3] Thời gian suy luận: LỖI\n    {e}")

    # 4. IDS (nếu có accuracy)
    ids_value = None
    if accuracy is not None:
        ids_value = compute_ids(accuracy, params_info["size_m"])
        print(f"\n[4] Information Density (IDS):")
        print(f"    ACC = {accuracy:.2f}%  ->  IDS = {ids_value:.2f}")

    return {
        "model_name": model_name,
        "params": params_info,
        "flops": flops_info,
        "inference_time": time_info,
        "accuracy": accuracy,
        "ids": ids_value,
    }


if __name__ == "__main__":
    device = auto_device()
    print("=" * 80)
    print(f"BẮT ĐẦU BENCHMARK TRÊN DEVICE: {device}")
    print("=" * 80)

    models_dict = {
        "Light_HybricMamba": (
            HybricMamba(
                dims=(3, 16, 32, 56, 96),
                num_classes=43,
                mbconv_expand_ratio=4,
                ssm_d_state=8,
                mamba_blocks=(1, 1),
                ssm_frac=0.5,
                conv_frac=0.3,
                use_aux=True,
            ),
            97.46,
        ),
        "Medium_HybricMamba": (
            HybricMamba(
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
            ),
            97.86,
        ),
        "Heavy_HybricMamba": (
            HybricMamba(
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
            ),
            98.55,
        ),
        "SuperMamba_dim3": (Super_Mamba(dims=3, depth=3, num_classes=43), 98.06),
        "SuperMamba_dim4": (Super_Mamba(dims=3, depth=4, num_classes=43), 98.43),
    }

    final_results = {}

    for name, (model, acc) in models_dict.items():
        full_report(model, model_name=name, accuracy=acc, device=device)

        latencies = []
        for _ in range(10):
            t_info = measure_inference_time(model, device=device, n_warmup=50, n_runs=400)
            latencies.append(t_info["median_ms"])
            time.sleep(1)

        final_results[name] = latencies

        time.sleep(10)

    # 3. In kết quả tổng hợp
    print("\n" + "=" * 80)
    print("TỔNG HỢP KẾT QUẢ SUY LUẬN (MEDIAN 10 LẦN THỬ NGHIỆM)")
    print("=" * 80)
    for name, times in final_results.items():
        print(
            f"{name:<20s} - median: {np.median(times):.3f} ms "
            f"(min={min(times):.3f}, max={max(times):.3f})"
        )