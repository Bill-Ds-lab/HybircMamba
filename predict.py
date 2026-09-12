import os
import torch
from PIL import Image
from torchvision import transforms
from models.CNN_Mamba_CNN_Mamba_Enhanced.HybricMamba import HybricMamba

CHECKPOINT_PATH = "Ressult/TFJ/LIGHT_HYBRIC_MAMBA/German/LIGHT_HYBRIC_MAMBA_best.pth"
IMAGE_PATHS = [
    "image/class0.jpg",
    "image/class4.jpg",
    "image/class41.jpg",
    "image/class5.jpg",
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HybricMamba(dims=(3, 16, 32, 56, 96),
            num_classes=43,
            mbconv_expand_ratio=4,
            ssm_d_state=8,
            mamba_blocks=(1, 1),
            ssm_frac=0.5,
            conv_frac=0.3,
            use_aux=True,
        )
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
if isinstance(state_dict, dict):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict, strict=False)
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_paths = [p for p in IMAGE_PATHS if os.path.exists(p)]
if valid_paths:
    input_tensor = torch.stack([transform(Image.open(p).convert("RGB")) for p in valid_paths]).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = output[0] if isinstance(output, tuple) else output
        probs = torch.softmax(output, dim=1)
        preds = torch.argmax(probs, dim=1)

    print("=" * 45)
    for path, pred, prob in zip(valid_paths, preds, probs):
        conf = prob[pred].item() * 100
        print(f"Ảnh: {os.path.basename(path)} | Class: {pred.item():02d} | Conf: {conf:.2f}%")
    print("=" * 45)
else:
    print(" Không tìm thấy đường dẫn ảnh hợp lệ.")