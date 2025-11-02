#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA (INT4 + LoRA) vs FP16 LoRA Demo
-------------------------------------
- Model: TinyViT-like block (fc1 + GELU + fc2)
- Pipeline: model -> (optionally) convert Linear to INT4 -> apply LoRA -> train & benchmark -> print results
- Dependencies: torch, bitsandbytes, peft
"""
from __future__ import annotations
import os, time, math, random, argparse
from dataclasses import dataclass, asdict
from typing import Optional, Iterable, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Soft imports (we check availability at runtime)
try:
    import bitsandbytes as bnb
    BNB = True
except Exception:
    BNB = False
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT = True
except Exception:
    PEFT = False


# ----------------------------- Utilities -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- Tiny Model -----------------------------

class TinyViTBlock(nn.Module):
    """A small MLP-like transformer MLP block (no attention) for demo purposes.
    Input: [N, D], Output: [N, D]
    """
    def __init__(self, d: int = 512, expansion: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d, expansion * d)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(expansion * d, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ---------------------- INT4 conversion & LoRA ------------------------

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
    quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.bfloat16,
    compress_statistics: bool = True,
) -> nn.Module:
    """Replace nn.Linear with bnb.nn.Linear4bit (weight-only INT4)."""
    if not BNB:
        raise ImportError("bitsandbytes is required: pip install bitsandbytes")
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
    target_modules: Optional[List[str]] = None,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    task_type: str = "feature_extraction",
) -> nn.Module:
    """Attach LoRA adapters via PEFT. Only LoRA params will be trainable."""
    if not PEFT:
        raise ImportError("peft is required: pip install peft")
    # TaskType for general modules
    _task = {
        "feature_extraction": TaskType.FEATURE_EXTRACTION,
        "seq2seq_lm": TaskType.SEQ_2_SEQ_LM,
        "causal_lm": TaskType.CAUSAL_LM,
        "token_classification": TaskType.TOKEN_CLS,
    }.get(task_type, TaskType.FEATURE_EXTRACTION)
    cfg = LoraConfig(
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=target_modules, bias="none", task_type=_task,
    )
    m = get_peft_model(model, cfg)
    try:
        m.print_trainable_parameters()
    except Exception:
        pass
    return m


# ---------------------------- Data Setup -----------------------------

def make_toy_regression(n_train=4096, n_val=1024, dim=512, noise=0.05, seed=123):
    """Synthetic regression: y = Wx + b (+ noise)."""
    g = torch.Generator().manual_seed(seed)
    W_true = torch.randn(dim, dim, generator=g) / math.sqrt(dim)
    b_true = torch.randn(dim, generator=g)

    def synth(n):
        X = torch.randn(n, dim, generator=g)
        Y = X @ W_true.T + b_true + noise * torch.randn(n, dim, generator=g)
        return X, Y

    Xtr, Ytr = synth(n_train)
    Xva, Yva = synth(n_val)
    return (Xtr, Ytr), (Xva, Yva)


# -------------------------- Training / Eval --------------------------

@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 2
    lr_lora: float = 2e-4
    lr_head: float = 1e-3
    weight_decay_lora: float = 0.0
    weight_decay_head: float = 1e-2
    amp_dtype: str = "bf16"  # "bf16" or "fp16"

@dataclass
class RunResult:
    name: str
    lat_ms: float
    samples_per_s: float
    max_mem_gb: float
    final_train_loss: float
    final_val_loss: float

def train_one(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainConfig, name="run") -> RunResult:
    dev = device()
    model.to(dev).train()

    # split params: LoRA vs others
    lora_params, other_params = [], []
    for n,p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            lora_params.append(p)
        else:
            other_params.append(p)
    # For this demo, "other" is empty (only LoRA should be trainable), but keep structure
    optim = torch.optim.AdamW([
        {"params": other_params, "lr": cfg.lr_head, "weight_decay": cfg.weight_decay_head},
        {"params": lora_params,  "lr": cfg.lr_lora, "weight_decay": cfg.weight_decay_lora},
    ])

    use_bf16 = (cfg.amp_dtype == "bf16")
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = None if use_bf16 else torch.cuda.amp.GradScaler()

    loss_fn = nn.MSELoss()

    # Warmup forward to stabilize kernels & memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
        for xb, yb in val_loader:
            _ = model(input_ids=xb.to(dev))

    # Training loop
    start = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
                pred = model(input_ids=xb)
                loss = loss_fn(pred, yb)
            if scaler is None:
                loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)
            else:
                scaler.scale(loss).backward()
                scaler.step(optim); scaler.update()
                optim.zero_grad(set_to_none=True)

        # quick val
        model.eval()
        with torch.no_grad():
            vloss = 0.0; n = 0
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
                    pred = model(input_ids=xb)
                    vloss += loss_fn(pred, yb).item() * xb.size(0)
                    n += xb.size(0)
            vloss /= max(1, n)

    elapsed = time.time() - start

    # Throughput measurement on a single batch
    model.eval()
    xb, yb = next(iter(val_loader))
    xb = xb.to(dev)
    iters = 50
    with torch.no_grad():
        for _ in range(10):
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
                _ = model(input_ids=xb)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
                _ = model(input_ids=xb)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    lat_ms = dt * 1000.0
    sps = xb.size(0) / dt

    # final train loss on a batch
    model.train()
    with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
        xb_tr, yb_tr = next(iter(train_loader))
        xb_tr, yb_tr = xb_tr.to(dev), yb_tr.to(dev)
        pred = model(input_ids=xb_tr); train_loss = loss_fn(pred, yb_tr).item()

    max_mem_gb = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0

    return RunResult(
        name=name,
        lat_ms=lat_ms,
        samples_per_s=sps,
        max_mem_gb=max_mem_gb,
        final_train_loss=train_loss,
        final_val_loss=vloss,
    )


# ------------------------------- Main --------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--amp", type=str, default="bf16", choices=["bf16","fp16"])
    parser.add_argument("--quant_type", type=str, default="nf4", choices=["nf4","fp4"])
    parser.add_argument("--skip_int4_fc2", action="store_true")
    args = parser.parse_args()

    set_seed(42)

    dev = device()
    print(f"[Info] Device: {dev}, CUDA: {torch.cuda.is_available()}, BF16 supported: {bf16_supported()}")

    # data
    (Xtr, Ytr), (Xva, Yva) = make_toy_regression(dim=args.dim)
    train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=args.batch_size, shuffle=False)

    # amp dtype
    amp_dtype = "bf16" if (args.amp == "bf16" and bf16_supported()) else "fp16"
    print(f"[Info] AMP dtype: {amp_dtype}")

    results: List[RunResult] = []

    # ---------------- FP16/BF16 LoRA (no int4) ----------------
    if not PEFT:
        raise SystemExit("peft not available. Please: pip install peft")
    model_fp = TinyViTBlock(d=args.dim)
    model_fp = apply_lora(model_fp, target_modules=["fc1","fc2"], r=args.r, lora_alpha=32, lora_dropout=0.05)
    cfg = TrainConfig(batch_size=args.batch_size, epochs=args.epochs, amp_dtype=amp_dtype)
    res_fp = train_one(model_fp, train_loader, val_loader, cfg, name=f"LoRA_{amp_dtype}")
    results.append(res_fp)

    # ---------------- INT4 + LoRA (QLoRA) ----------------
    if not BNB:
        raise SystemExit("bitsandbytes not available. Please: pip install bitsandbytes")
    model_q = TinyViTBlock(d=args.dim)
    skip = ["fc2"] if args.skip_int4_fc2 else []
    compute_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    model_q = convert_linear_to_int4(model_q, quant_type=args.quant_type, compute_dtype=compute_dtype, skip_modules=skip)
    model_q = apply_lora(model_q, target_modules=["fc1","fc2"], r=args.r, lora_alpha=32, lora_dropout=0.05)
    cfg_q = TrainConfig(batch_size=args.batch_size, epochs=args.epochs, amp_dtype=amp_dtype)
    res_q = train_one(model_q, train_loader, val_loader, cfg_q, name=f"INT4+LoRA({args.quant_type},{amp_dtype})")
    results.append(res_q)

    # ---------------- Print results ----------------
    def fmt(x, nd=3):
        return f"{x:.{nd}f}"
    print("\n=== Results ===")
    print(f"{'run':28s}  {'lat_ms':>8s}  {'samples/s':>10s}  {'max_mem(GB)':>11s}  {'train_loss':>10s}  {'val_loss':>8s}")
    for r in results:
        print(f"{r.name:28s}  {fmt(r.lat_ms):>8s}  {fmt(r.samples_per_s):>10s}  {fmt(r.max_mem_gb):>11s}  {fmt(r.final_train_loss):>10s}  {fmt(r.final_val_loss):>8s}")

    # Optional: write CSV
    try:
        import pandas as pd
        df = pd.DataFrame([asdict(r) for r in results])
        out = "qlora_demo_results.csv"
        df.to_csv(out, index=False)
        print(f"\n[Saved] {out}")
    except Exception as e:
        print(f"[Warn] Could not save CSV: {e}")

if __name__ == "__main__":
    main()
