# --- int8_quant.py ---
from __future__ import annotations
from typing import Iterable, Optional, List
import torch
import torch.nn as nn

try:
    import bitsandbytes as bnb
except ImportError as e:
    raise ImportError("bitsandbytes が必要です: pip install bitsandbytes") from e


def _get_parent_and_attr(model: nn.Module, name: str):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def convert_linear_to_int8(
    model: nn.Module,
    *,
    skip_modules: Optional[Iterable[str]] = None,
    has_fp16_weights: bool = False,
    threshold: float = 6.0,
    memory_efficient_backward: bool = False,
) -> nn.Module:
    """
    nn.Linear を bnb.nn.Linear8bitLt に置き換える（inference/微調整向け）。
    """
    skip_modules = list(skip_modules or [])
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if any(k in name for k in skip_modules):
                continue

            parent, attr = _get_parent_and_attr(model, name)
            new_layer = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=(module.bias is not None),
                has_fp16_weights=has_fp16_weights,
                threshold=threshold,
                memory_efficient_backward=memory_efficient_backward,
            )
            # 重み/バイアスをコピー
            with torch.no_grad():
                new_layer.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    new_layer.bias.data.copy_(module.bias.data)

            setattr(parent, attr, new_layer)
    return model


def assert_int8_applied(model: nn.Module):
    """置換確認（nn.Linearが残っていないことを確認）"""
    has_plain = any(isinstance(m, nn.Linear) for m in model.modules())
    has_int8  = any(isinstance(m, bnb.nn.Linear8bitLt) for m in model.modules())
    assert has_int8, "bnb.nn.Linear8bitLt が見つかりません"
    assert not has_plain, "置換後に nn.Linear が残っています"


# --- 使用例 ---
if __name__ == "__main__":
    class SmallMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(784, 256), nn.ReLU(),
                nn.Linear(256, 10)
            )
        def forward(self, x): return self.net(x)

    model = SmallMLP().cuda()
    model = convert_linear_to_int8(model, skip_modules=["net.2"])  # 例: 最終層はFPで保持
    assert_int8_applied(model)
    x = torch.randn(32, 784, device="cuda")
    y = model(x)  # 8bitで推論
    print(y.shape)
