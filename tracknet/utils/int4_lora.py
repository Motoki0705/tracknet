# --- qlora_int4.py ---
from __future__ import annotations
from typing import Iterable, Optional
import torch
import torch.nn as nn

try:
    import bitsandbytes as bnb
except ImportError as e:
    raise ImportError("bitsandbytes が必要です: pip install bitsandbytes") from e

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError as e:
    raise ImportError("peft が必要です: pip install peft") from e


def _get_parent_and_attr(model: nn.Module, name: str):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def convert_linear_to_int4(
    model: nn.Module,
    *,
    skip_modules: Optional[Iterable[str]] = None,
    quant_type: str = "nf4",                 # "nf4" or "fp4"
    compute_dtype: torch.dtype = torch.float16,  # Ampere以降は bfloat16 も可
    compress_statistics: bool = True,        # いわゆる double quant
) -> nn.Module:
    """
    nn.Linear を bnb.nn.Linear4bit に置き換える（QLoRAのベース作成）。
    """
    skip_modules = list(skip_modules or [])
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if any(k in name for k in skip_modules):
                continue

            parent, attr = _get_parent_and_attr(model, name)
            new_layer = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=(module.bias is not None),
                compute_dtype=compute_dtype,
                quant_type=quant_type,
                compress_statistics=compress_statistics,
            )
            with torch.no_grad():
                new_layer.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    new_layer.bias.data.copy_(module.bias.data)

            setattr(parent, attr, new_layer)
    return model


def apply_lora(
    model: nn.Module,
    *,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,  # 例: ["q_proj","k_proj","v_proj","o_proj"]
    bias: str = "none",
    task_type: TaskType = TaskType.FEATURE_EXTRACTION,   # Vision/一般用途
) -> nn.Module:
    """
    LoRA アダプタを付与。trainableはLoRAのみに限定される。
    """
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,  # NoneならPEFTの自動検出に任せる（誤爆がある場合は明示推奨）
        bias=bias,
        task_type=task_type,
    )
    lora_model = get_peft_model(model, lora_cfg)
    # 進捗確認：学習可能パラメータ比率を出力
    try:
        lora_model.print_trainable_parameters()
    except Exception:
        pass
    return lora_model


def assert_int4_lora_ready(model: nn.Module):
    """Linear4bit が含まれることだけ簡易チェック"""
    has_int4 = any(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())
    assert has_int4, "bnb.nn.Linear4bit が見つかりません（置換できていない可能性）"


# --- 使用例 ---
if __name__ == "__main__":
    class TinyViTBlock(nn.Module):
        def __init__(self, d=256):
            super().__init__()
            self.fc1 = nn.Linear(d, 4*d)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(4*d, d)
        def forward(self, x): return self.fc2(self.act(self.fc1(x)))

    model = TinyViTBlock().cuda()
    # 1) ベースを4bit化
    bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute = torch.bfloat16 if bf16_ok else torch.float16
    model = convert_linear_to_int4(model, quant_type="nf4", compute_dtype=compute)
    assert_int4_lora_ready(model)

    # 2) LoRAを付与（対象モジュール名はモデルに合わせて指定推奨）
    #   名前解決が難しい場合は None で自動検出 -> 学習時に print_trainable_parameters で要確認
    model = apply_lora(model, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=None)

    # 3) 学習は通常通り（LoRAアダプタのみ更新）
    x = torch.randn(32, 256, device="cuda")
    y = torch.randn(32, 256, device="cuda")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward(); opt.step(); opt.zero_grad()
    print("step ok, loss:", float(loss))
