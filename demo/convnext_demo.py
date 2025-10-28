import torch
from transformers import AutoImageProcessor, AutoModel, AutoBackbone
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)
print("Loaded image size:", image.size) # (640, 480)

# ConvNeXtの4ステージを全部取り出す（stage1〜stage4）
pretrained_model_name = "facebook/dinov3-convnext-base-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    
print("Output keys:", outputs.keys())
print("Pooled output shape:", outputs.pooler_output.shape)  # torch.Size([1, 1024])
print("Last hidden state shape:", outputs.last_hidden_state.shape)  # torch.Size([1, 50, 1024])

try:
    print(type(outputs.hidden_states), len(outputs.hidden_states))  # tuple, 個数はモデルによる
    for i, h in enumerate(outputs.hidden_states):
        print(f" hidden_states[{i}] shape: {h.shape}")
except AttributeError:
    print("No hidden_states in outputs")
    print("Available keys:", outputs.keys())
