import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)
print("Loaded image size:", image.size)  # (640, 480)

pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(
    pretrained_model_name,
    device_map="auto",
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)  # torch.Size([1, 384])
print(
    "last hidden state shape:", outputs.last_hidden_state.shape
)  # torch.Size([1, 201, 384])

last = outputs.last_hidden_state  # [B, 201, 384]
num_reg = model.config.num_register_tokens  # 4

cls = last[:, 0, :]  # [B, 384]
regs = last[:, 1 : 1 + num_reg, :]  # [B, 4, 384]
patch = last[:, 1 + num_reg :, :]  # [B, 196, 384]

# 14x14 のパッチグリッドに戻す（B, 14,14,384）
B = patch.shape[0]
patch_grid = patch.unflatten(1, (14, 14))

print("pathch grid shape:", patch_grid.shape)  # torch.Size([1, 14, 14, 384])
