#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF ViT (DINOv3) - INT4 + LoRA (QLoRA) vs FP16/BF16 LoRA Demo (v2)
------------------------------------------------------------------
Changes vs v1:
- Use Hugging Face BitsAndBytesConfig to load quantized model (robust placement)
- Call prepare_model_for_kbit_training() for stability (LayerNorm dtypes, grads, etc.)
- Free teacher from GPU before training to get fair VRAM numbers
- Use `dtype=` (not deprecated torch_dtype)
- Detach when reading losses to silence warnings
- Keep AMP dtype consistent with BnB compute dtype
"""
from __future__ import annotations
import argparse, time, math, random, os
from dataclasses import dataclass, asdict
from typing import List, Optional, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd

try:
    import bitsandbytes as bnb
    BNB = True
except Exception:
    BNB = False

try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    PEFT = True
except Exception:
    PEFT = False

from transformers import AutoImageProcessor, AutoModel, BitsAndBytesConfig
from transformers.image_utils import load_image
import torchvision.transforms as T


# ------------------ utils ------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dev() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def print_trainable(m: nn.Module):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(1, total)
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")


# --------------- LoRA ---------------

def auto_target_modules(model: nn.Module) -> List[str]:
    candidates = [
        "q_proj","k_proj","v_proj","o_proj","qkv","attn.proj",
        "attention.query","attention.key","attention.value","attention.output.dense",
        "mlp.fc1","mlp.fc2","intermediate.dense","output.dense",
        "fc1","fc2","proj",
    ]
    names = [n for n,_ in model.named_modules()]
    found = sorted({c for c in candidates if any(c in n for n in names)})
    found = [f for f in found if "layernorm" not in f.lower() and "norm" not in f.lower()]
    return found if found else ["fc1","fc2"]

def apply_lora(model: nn.Module, target_modules: Optional[List[str]] = None,
               r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05) -> nn.Module:
    if not PEFT:
        raise ImportError("peft is required: pip install peft")
    if target_modules is None:
        target_modules = auto_target_modules(model)
    cfg = LoraConfig(
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=target_modules, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    m = get_peft_model(model, cfg)
    try:
        m.print_trainable_parameters()
    except Exception:
        print_trainable(m)
    return m


# --------------- Distillation dataset ---------------

def make_augmented_batch(processor, pil_img, n_samples=256, size=224, seed=123):
    g = torch.Generator().manual_seed(seed)
    aug = T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    ])
    imgs = [aug(pil_img) for _ in range(n_samples)]
    batch = processor(images=imgs, return_tensors="pt")
    return batch["pixel_values"]  # (N, 3, H, W)


# --------------- Training / Evaluation ---------------

@dataclass
class TrainCfg:
    epochs: int = 1
    batch_size: int = 32
    amp: str = "bf16"  # or "fp16"
    lr_lora: float = 2e-4
    wd_lora: float = 0.0

@dataclass
class RunResult:
    name: str
    lat_ms: float
    samples_per_s: float
    max_mem_gb: float
    final_train_loss: float
    final_val_loss: float

@torch.no_grad()
def teacher_targets(teacher: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    teacher.eval()
    devc = next(teacher.parameters()).device
    outs = []
    for i in range(0, pixel_values.size(0), 32):
        pv = pixel_values[i:i+32].to(devc)
        out = teacher(pixel_values=pv)
        outs.append(out.pooler_output.detach().cpu())  # (B, D)
    return torch.cat(outs, dim=0)

def run_train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainCfg, name="run") -> RunResult:
    d = dev(); model.to(d).train()
    use_bf16 = (cfg.amp == "bf16" and bf16_supported())
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = None if use_bf16 else torch.cuda.amp.GradScaler()
    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr_lora, weight_decay=cfg.wd_lora)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # warmup
    model.eval()
    xb, yb = next(iter(val_loader))
    xb = xb.to(d)
    with torch.autocast(device_type=d.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
        _ = model(pixel_values=xb)
    model.train()

    # train loop
    for epoch in range(cfg.epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(d), yb.to(d)
            with torch.autocast(device_type=d.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
                out = model(pixel_values=xb).pooler_output
                loss = loss_fn(out, yb)
            if scaler is None:
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            else:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

    # val
    model.eval()
    vloss = 0.0; n = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(d), yb.to(d)
            with torch.autocast(device_type=d.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
                out = model(pixel_values=xb).pooler_output
                vloss += float(nn.functional.mse_loss(out, yb, reduction="sum").detach())
                n += xb.size(0)
    vloss /= max(1, n)

    # latency/throughput
    xb, _ = next(iter(val_loader)); xb = xb.to(d)
    with torch.no_grad():
        for _ in range(10):
            with torch.autocast(device_type=d.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
                _ = model(pixel_values=xb).pooler_output
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    iters = 30
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            with torch.autocast(device_type=d.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
                _ = model(pixel_values=xb).pooler_output
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    lat_ms = dt * 1000.0
    sps = xb.size(0) / dt
    max_mem_gb = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0

    # final train loss on one batch
    model.train()
    xb, yb = next(iter(train_loader)); xb, yb = xb.to(d), yb.to(d)
    with torch.autocast(device_type=d.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
        out = model(pixel_values=xb).pooler_output
        train_loss = float(nn.functional.mse_loss(out, yb).detach())

    return RunResult(name=name, lat_ms=lat_ms, samples_per_s=sps, max_mem_gb=max_mem_gb,
                     final_train_loss=train_loss, final_val_loss=vloss)


# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--image_url", type=str, default="http://images.cocodataset.org/val2017/000000039769.jpg")
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--amp", type=str, default="bf16", choices=["bf16","fp16"])
    ap.add_argument("--quant_type", type=str, default="nf4", choices=["nf4","fp4"])
    ap.add_argument("--r", type=int, default=16)
    args = ap.parse_args()

    set_seed(42)
    use_bf16 = (args.amp == "bf16" and bf16_supported())
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Processor + image
    processor = AutoImageProcessor.from_pretrained(args.model)
    pil = load_image(args.image_url)
    print(f"[Info] Loaded image: {pil.size}")

    # Augmentations
    size = processor.size.get("shortest_edge", 224)
    pixel_values = make_augmented_batch(processor, pil, n_samples=args.num_samples, size=size)

    # Teacher (float) -> targets
    print("[Info] Loading teacher (float) ...")
    teacher = AutoModel.from_pretrained(args.model, dtype=amp_dtype, device_map=None)
    teacher.to(dev())
    with torch.inference_mode():
        y_train = teacher_targets(teacher, pixel_values[: int(0.8 * len(pixel_values))])
        y_val   = teacher_targets(teacher, pixel_values[int(0.8 * len(pixel_values)) :])
    # Free teacher to avoid skewing VRAM metrics
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Dataloaders
    n = pixel_values.size(0)
    n_train = int(0.8 * n)
    train_ds = TensorDataset(pixel_values[:n_train], y_train)
    val_ds   = TensorDataset(pixel_values[n_train:], y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    results = []

    # ---------- Run A: LoRA (no int4) ----------
    print("[Run A] LoRA (float backbone)")
    model_fp = AutoModel.from_pretrained(args.model, dtype=amp_dtype, device_map=None)
    model_fp = apply_lora(model_fp, target_modules=None, r=args.r, lora_alpha=32, lora_dropout=0.05)
    results.append(run_train(model_fp, train_loader, val_loader,
                             TrainCfg(epochs=args.epochs, batch_size=args.batch_size, amp=args.amp),
                             name=f"LoRA_{args.amp}"))

    # ---------- Run B: INT4 + LoRA (QLoRA) ----------
    if not (BNB and PEFT):
        raise SystemExit("bitsandbytes + peft are required for INT4+LoRA")
    print("[Run B] INT4 + LoRA (QLoRA)")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=(torch.bfloat16 if use_bf16 else torch.float16),
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_use_double_quant=True,
    )
    model_q = AutoModel.from_pretrained(args.model, quantization_config=bnb_cfg, device_map=None)
    # important for stability with k-bit training
    model_q = prepare_model_for_kbit_training(model_q)
    model_q = apply_lora(model_q, target_modules=None, r=args.r, lora_alpha=32, lora_dropout=0.05)
    results.append(run_train(model_q, train_loader, val_loader,
                             TrainCfg(epochs=args.epochs, batch_size=args.batch_size, amp=args.amp),
                             name=f"INT4+LoRA({args.quant_type},{args.amp})"))

    # ---------- Print + Save ----------
    def fmt(x, nd=3): return f"{x:.{nd}f}"
    print("\n=== Results ===")
    print(f"{'run':28s}  {'lat_ms':>8s}  {'samples/s':>10s}  {'max_mem(GB)':>11s}  {'train_loss':>10s}  {'val_loss':>8s}")
    for r in results:
        print(f"{r.name:28s}  {fmt(r.lat_ms):>8s}  {fmt(r.samples_per_s):>10s}  {fmt(r.max_mem_gb):>11s}  {fmt(r.final_train_loss):>10s}  {fmt(r.final_val_loss):>8s}")
    df = pd.DataFrame([asdict(r) for r in results])
    out_csv = "hf_vit_qlora_results_v2.csv"
    df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

if __name__ == "__main__":
    main()
