"""ViT backbone wrapper for TrackNet (minimal, HF only).

- 常に Hugging Face の ViT (AutoModel) を使用。
- 入力:  images  … [B, 3, H, W]  (呼び出し側でモデル想定どおりの正規化済みを推奨)
- 出力:  grid    … [B, H//16, W//16, C]
- グリッド復元は常に (H//16, W//16)。H と W は 16 の倍数を想定。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ViTBackboneConfig:
    """Configuration (HF only, minimal)."""
    pretrained_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    device_map: Optional[str] = "auto"
    local_files_only: bool = True
    patch_size: int = 16              # 要件: グリッドは H//16, W//16 に固定


class ViTBackbone(nn.Module):
    """Backbone that outputs patch tokens as a spatial grid.
    Input:  [B, 3, H, W]
    Output: [B, H//ps, W//ps, C]  (ps=16)
    """

    def __init__(self, cfg: ViTBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_size = int(cfg.patch_size)

        try:
            from transformers import AutoModel  # lazy import
        except Exception as e:  # pragma: no cover
            raise RuntimeError("transformers が見つかりません。`pip install transformers` を実行してください。") from e

        # モデル読み込み（ローカルキャッシュ優先）
        try:
            self.model = AutoModel.from_pretrained(
                cfg.pretrained_model_name,
                device_map=cfg.device_map,
                local_files_only=cfg.local_files_only,
            )
        except Exception as e:
            raise RuntimeError(
                "事前学習済み ViT のローカル読み込みに失敗しました。"
                "事前にキャッシュするか、config.local_files_only=False を指定してください。"
            ) from e

        # init で一度だけ取得して保持
        self.hidden_dim: int = int(self.model.config.hidden_size)
        self.num_reg: int = int(getattr(self.model.config, "num_register_tokens", 0))
        self.has_cls: bool = True  # ViT は通常クラス埋め込みを先頭に持つ
        # （任意）モデル側の patch_size と整合性チェック
        model_ps = getattr(self.model.config, "patch_size", None)
        if model_ps is not None:
            model_ps = model_ps if isinstance(model_ps, int) else int(model_ps[0])
            if model_ps != self.patch_size:
                raise ValueError(
                    f"config.patch_size({self.patch_size}) と model.config.patch_size({model_ps}) が不一致です。"
                    " H//16, W//16 を前提とするため、patch_size=16 のモデルを使用してください。"
                )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute patch token grid from input images.

        Args:
            images: [B, 3, H, W] （モデル期待どおりに前処理済みを推奨）

        Returns:
            [B, H//16, W//16, C]
        """
        if images.dim() != 4 or images.size(1) != 3:
            raise ValueError(f"Expected images shape [B,3,H,W], got {tuple(images.shape)}")

        B, _, H, W = images.shape
        ps = self.patch_size
        if (H % ps) != 0 or (W % ps) != 0:
            raise ValueError(f"H({H}) と W({W}) は patch_size({ps}) の倍数である必要があります。")

        Hp, Wp = H // ps, W // ps

        # 余計な処理を避けてダイレクトに入力
        # ※ 呼び出し側で正規化（mean/std）を合わせてください。
        outputs = self.model(pixel_values=images)  # last_hidden_state: [B, 1+reg+N, C]
        last = outputs.last_hidden_state
        if last is None:
            raise RuntimeError("last_hidden_state が None です。")

        # init で取得した num_reg / has_cls を使用
        start_idx = (1 if self.has_cls else 0) + self.num_reg
        patch = last[:, start_idx:, :]  # [B, N, C], N should be Hp*Wp

        expected = Hp * Wp
        if patch.size(1) != expected:
            raise RuntimeError(
                f"トークン数不一致: 期待 {expected} (= {Hp}x{Wp}) / 実際 {patch.size(1)}。"
                " 入力前処理（リサイズ/切り抜き）で形状が変わっていないか確認してください。"
            )

        grid = patch.unflatten(1, (Hp, Wp))  # -> [B, Hp, Wp, C]
        return grid
