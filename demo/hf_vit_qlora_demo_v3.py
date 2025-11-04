#!/usr/bin/env python3
"""
HF ViT (DINOv3) - QLoRA Demo (v3)
---------------------------------
Fixes for stability vs v2:
- Distill CLS features (with final layer norm) by default instead of pooler head
- Cosine loss by default (scale-invariant), MSE optional
- Manual INT4 conversion with skip list (e.g., keep 'pooler.dense' in FP)
- Optionally support HF BitsAndBytesConfig ("hf" mode)
- prepare_model_for_kbit_training() after quantization
"""

from __future__ import annotations

import argparse
import contextlib
import random
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoImageProcessor, AutoModel, BitsAndBytesConfig
from transformers.image_utils import load_image

try:
    import bitsandbytes as bnb

    BNB = True
except Exception:
    BNB = False

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    PEFT = True
except Exception:
    PEFT = False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dev() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _get_parent_and_attr(model: nn.Module, name: str):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def convert_linear_to_int4_manual(
    model: nn.Module,
    *,
    skip_modules: Iterable[str] | None = None,
    quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.bfloat16,
    compress_statistics: bool = True,
) -> nn.Module:
    if not BNB:
        raise ImportError("bitsandbytes is required for manual int4 path")
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


def auto_target_modules(model: nn.Module) -> list[str]:
    cands = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv",
        "attn.proj",
        "attention.query",
        "attention.key",
        "attention.value",
        "attention.output.dense",
        "mlp.fc1",
        "mlp.fc2",
        "intermediate.dense",
        "output.dense",
        "fc1",
        "fc2",
        "proj",
    ]
    names = [n for n, _ in model.named_modules()]
    got = sorted({c for c in cands if any(c in n for n in names)})
    got = [g for g in got if "norm" not in g.lower()]
    return got if got else ["fc1", "fc2"]


def apply_lora(
    model: nn.Module,
    target_modules: list[str] | None = None,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> nn.Module:
    if not PEFT:
        raise ImportError("peft is required")
    if target_modules is None:
        target_modules = auto_target_modules(model)
    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    m = get_peft_model(model, cfg)
    with contextlib.suppress(Exception):
        m.print_trainable_parameters()
    return m


def make_augmented_batch(processor, pil_img, n_samples=256, size=224, seed=123):
    torch.Generator().manual_seed(seed)
    aug = T.Compose(
        [
            T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        ]
    )
    imgs = [aug(pil_img) for _ in range(n_samples)]
    batch = processor(images=imgs, return_tensors="pt")
    return batch["pixel_values"]


@torch.no_grad()
def extract_features(
    model: nn.Module, pixel_values: torch.Tensor, kind: str = "cls"
) -> torch.Tensor:
    model.eval()
    d = next(model.parameters()).device
    outs = []
    bs = 32
    for i in range(0, pixel_values.size(0), bs):
        pv = pixel_values[i : i + bs].to(d)
        out = model(pixel_values=pv, output_hidden_states=True)
        if kind == "pooler":
            feat = out.pooler_output  # (B, D)
        else:
            cls = out.last_hidden_state[:, 0, :]
            if hasattr(model, "layernorm"):
                cls = model.layernorm(cls)
            feat = cls
        outs.append(feat.detach().cpu())
    return torch.cat(outs, dim=0)


class CosineLoss(nn.Module):
    def forward(self, x, y):
        x = nn.functional.normalize(x, dim=-1)
        y = nn.functional.normalize(y, dim=-1)
        return (1 - nn.functional.cosine_similarity(x, y, dim=-1)).mean()


@dataclass
class TrainCfg:
    epochs: int = 1
    batch_size: int = 32
    amp: str = "bf16"
    lr_lora: float = 2e-4
    wd_lora: float = 0.0
    loss: str = "cosine"


@dataclass
class RunResult:
    name: str
    lat_ms: float
    samples_per_s: float
    max_mem_gb: float
    final_train_loss: float
    final_val_loss: float


def run_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainCfg,
    name="run",
) -> RunResult:
    d = dev()
    model.to(d).train()
    use_bf16 = cfg.amp == "bf16" and bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = None if use_bf16 else torch.cuda.amp.GradScaler()
    loss_fn = CosineLoss() if cfg.loss == "cosine" else nn.MSELoss()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_lora,
        weight_decay=cfg.wd_lora,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # warm
    model.eval()
    xb, yb = next(iter(val_loader))
    xb = xb.to(d)
    with torch.autocast(
        device_type=d.type, dtype=amp_dtype, enabled=torch.cuda.is_available()
    ):
        _ = model(pixel_values=xb)
    model.train()

    # train
    for _epoch in range(cfg.epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(d), yb.to(d)
            with torch.autocast(
                device_type=d.type, dtype=amp_dtype, enabled=torch.cuda.is_available()
            ):
                if cfg.loss == "cosine":
                    out = model(
                        pixel_values=xb, output_hidden_states=True
                    ).last_hidden_state[:, 0, :]
                    if hasattr(model, "layernorm"):
                        out = model.layernorm(out)
                else:
                    out = model(
                        pixel_values=xb, output_hidden_states=False
                    ).pooler_output
                loss = loss_fn(out, yb)
            if scaler is None:
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            else:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

    # val
    model.eval()
    vloss = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(d), yb.to(d)
            with torch.autocast(
                device_type=d.type, dtype=amp_dtype, enabled=torch.cuda.is_available()
            ):
                if cfg.loss == "cosine":
                    out = model(
                        pixel_values=xb, output_hidden_states=True
                    ).last_hidden_state[:, 0, :]
                    if hasattr(model, "layernorm"):
                        out = model.layernorm(out)
                else:
                    out = model(
                        pixel_values=xb, output_hidden_states=False
                    ).pooler_output
                vloss += float(loss_fn(out, yb).detach()) * xb.size(0)
                n += xb.size(0)
    vloss /= max(1, n)

    # latency/throughput
    xb, _ = next(iter(val_loader))
    xb = xb.to(d)
    with torch.no_grad():
        for _ in range(10):
            with torch.autocast(
                device_type=d.type, dtype=amp_dtype, enabled=torch.cuda.is_available()
            ):
                _ = model(pixel_values=xb)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    iters = 30
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            with torch.autocast(
                device_type=d.type, dtype=amp_dtype, enabled=torch.cuda.is_available()
            ):
                _ = model(pixel_values=xb)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    lat_ms = dt * 1000.0
    sps = xb.size(0) / dt
    max_mem_gb = (
        (torch.cuda.max_memory_allocated() / 1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    # final train loss
    model.train()
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(d), yb.to(d)
    with torch.autocast(
        device_type=d.type, dtype=amp_dtype, enabled=torch.cuda.is_available()
    ):
        if cfg.loss == "cosine":
            out = model(pixel_values=xb, output_hidden_states=True).last_hidden_state[
                :, 0, :
            ]
            if hasattr(model, "layernorm"):
                out = model.layernorm(out)
        else:
            out = model(pixel_values=xb, output_hidden_states=False).pooler_output
        train_loss = float(loss_fn(out, yb).detach())

    return RunResult(
        name=name,
        lat_ms=lat_ms,
        samples_per_s=sps,
        max_mem_gb=max_mem_gb,
        final_train_loss=train_loss,
        final_val_loss=vloss,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m"
    )
    ap.add_argument(
        "--image_url",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
    )
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--amp", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument(
        "--quant_mode", type=str, default="manual", choices=["manual", "hf"]
    )
    ap.add_argument("--skip_list", type=str, default="pooler.dense")
    ap.add_argument("--target", type=str, default="cls", choices=["cls", "pooler"])
    ap.add_argument("--loss", type=str, default="cosine", choices=["cosine", "mse"])
    args = ap.parse_args()

    set_seed(42)
    use_bf16 = args.amp == "bf16" and bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    processor = AutoImageProcessor.from_pretrained(args.model)
    pil = load_image(args.image_url)
    print(f"[Info] Loaded image: {pil.size}")

    size = processor.size.get("shortest_edge", 224)
    pixel_values = make_augmented_batch(
        processor, pil, n_samples=args.num_samples, size=size
    )

    print("[Info] Loading teacher (float) ...")
    teacher = AutoModel.from_pretrained(args.model, dtype=amp_dtype, device_map=None)
    teacher.to(dev())
    with torch.inference_mode():
        y_train = extract_features(
            teacher, pixel_values[: int(0.8 * len(pixel_values))], kind=args.target
        )
        y_val = extract_features(
            teacher, pixel_values[int(0.8 * len(pixel_values)) :], kind=args.target
        )
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n = pixel_values.size(0)
    n_train = int(0.8 * n)
    train_ds = TensorDataset(pixel_values[:n_train], y_train)
    val_ds = TensorDataset(pixel_values[n_train:], y_val)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Run A
    print("[Run A] LoRA (float backbone)")
    model_fp = AutoModel.from_pretrained(args.model, dtype=amp_dtype, device_map=None)
    model_fp = apply_lora(
        model_fp, target_modules=None, r=args.r, lora_alpha=32, lora_dropout=0.05
    )
    res_a = run_train(
        model_fp,
        train_loader,
        val_loader,
        TrainCfg(
            epochs=args.epochs, batch_size=args.batch_size, amp=args.amp, loss=args.loss
        ),
        name=f"LoRA_{args.amp}_{args.target}_{args.loss}",
    )
    del model_fp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Run B
    print("[Run B] INT4 + LoRA (QLoRA)")
    if args.quant_mode == "hf":
        if not BNB:
            raise SystemExit("bitsandbytes required for --quant_mode hf")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=(torch.bfloat16 if use_bf16 else torch.float16),
            bnb_4bit_quant_type=args.quant_type,
            bnb_4bit_use_double_quant=True,
        )
        model_q = AutoModel.from_pretrained(
            args.model, quantization_config=bnb_cfg, device_map=None
        )
    else:
        skip_list = (
            [s.strip() for s in args.skip_list.split(",")] if args.skip_list else []
        )
        model_q = AutoModel.from_pretrained(
            args.model, dtype=amp_dtype, device_map=None
        )
        model_q = convert_linear_to_int4_manual(
            model_q,
            quant_type=args.quant_type,
            compute_dtype=(torch.bfloat16 if use_bf16 else torch.float16),
            skip_modules=skip_list,
        )
    if not PEFT:
        raise SystemExit("peft is required")
    model_q = prepare_model_for_kbit_training(model_q)
    model_q = apply_lora(
        model_q, target_modules=None, r=args.r, lora_alpha=32, lora_dropout=0.05
    )
    res_b = run_train(
        model_q,
        train_loader,
        val_loader,
        TrainCfg(
            epochs=args.epochs, batch_size=args.batch_size, amp=args.amp, loss=args.loss
        ),
        name=f"INT4+LoRA({args.quant_type},{args.amp},{args.target},{args.loss})",
    )

    # Print + Save
    def fmt(x, nd=3):
        return f"{x:.{nd}f}"

    print("\n=== Results ===")
    print(
        f"{'run':45s}  {'lat_ms':>8s}  {'samples/s':>10s}  {'max_mem(GB)':>11s}  {'train_loss':>10s}  {'val_loss':>8s}"
    )
    for r in [res_a, res_b]:
        print(
            f"{r.name:45s}  {fmt(r.lat_ms):>8s}  {fmt(r.samples_per_s):>10s}  {fmt(r.max_mem_gb):>11s}  {fmt(r.final_train_loss):>10s}  {fmt(r.final_val_loss):>8s}"
        )
    out_csv = "hf_vit_qlora_results_v3.csv"
    pd.DataFrame([asdict(r) for r in [res_a, res_b]]).to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")


if __name__ == "__main__":
    main()
