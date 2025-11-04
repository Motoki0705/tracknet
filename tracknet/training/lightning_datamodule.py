"""PyTorch Lightning DataModule for TrackNet.

This module wraps the existing dataset and collate implementations to provide
train/val DataLoaders managed by Lightning.

Design goals:
- Reuse the project's dataset, preprocess, and collate utilities as-is.
- Keep batch, worker, and memory settings configurable via OmegaConf.
- Avoid side effects in ``__init__``; construct datasets in ``setup``.
"""

from __future__ import annotations

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from tracknet.datasets import (
    PreprocessConfig,
    TrackNetFrameDataset,
    TrackNetFrameDatasetConfig,
    TrackNetSequenceDataset,
    TrackNetSequenceDatasetConfig,
    collate_frames,
    collate_sequences,
)


class TrackNetDataModule(pl.LightningDataModule):
    """Lightning DataModule that builds TrackNet DataLoaders.

    Attributes are derived from ``cfg.data`` and ``cfg.training`` sections.

    Args:
        cfg: Unified OmegaConf configuration.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None
        self._heatmap_size: tuple[int, int] | None = None
        self._sigma: float | None = None

    def setup(self, stage: str | None = None) -> None:
        """Build datasets for the given stage.

        Args:
            stage: One of ``fit``/``validate``/``test``/``predict`` or ``None``.
        """

        dcfg = self.cfg.data
        mcfg = self.cfg.model

        pp = PreprocessConfig(
            resize=(
                None
                if dcfg.preprocess.get("resize", None) in (None, "null")
                else tuple(dcfg.preprocess.resize)
            ),
            normalize=bool(dcfg.preprocess.get("normalize", True)),
            flip_prob=float(dcfg.preprocess.get("flip_prob", 0.0)),
        )
        heatmap_size = (int(mcfg.heatmap.size[0]), int(mcfg.heatmap.size[1]))
        sigma = float(mcfg.heatmap.sigma)
        self._heatmap_size = heatmap_size
        self._sigma = sigma

        if bool(dcfg.sequence.get("enabled", False)):
            length = int(dcfg.sequence.get("length", 3))
            stride = int(dcfg.sequence.get("stride", 1))
            self.train_ds = TrackNetSequenceDataset(
                TrackNetSequenceDatasetConfig(
                    root=str(dcfg.root),
                    games=list(dcfg.split.train_games),
                    length=length,
                    stride=stride,
                    preprocess=pp,
                )
            )
            self.val_ds = (
                TrackNetSequenceDataset(
                    TrackNetSequenceDatasetConfig(
                        root=str(dcfg.root),
                        games=list(dcfg.split.val_games),
                        length=length,
                        stride=stride,
                        preprocess=pp,
                    )
                )
                if dcfg.split.get("val_games")
                else None
            )
            self._collate_fn = lambda b: collate_sequences(
                b, heatmap_size=heatmap_size, sigma=sigma
            )
        else:
            self.train_ds = TrackNetFrameDataset(
                TrackNetFrameDatasetConfig(
                    root=str(dcfg.root),
                    games=list(dcfg.split.train_games),
                    preprocess=pp,
                )
            )
            self.val_ds = (
                TrackNetFrameDataset(
                    TrackNetFrameDatasetConfig(
                        root=str(dcfg.root),
                        games=list(dcfg.split.val_games),
                        preprocess=pp,
                    )
                )
                if dcfg.split.get("val_games")
                else None
            )
            self._collate_fn = lambda b: collate_frames(
                b, heatmap_size=heatmap_size, sigma=sigma
            )

    def _dataloader_common_kwargs(self) -> dict:
        tcfg = self.cfg.training
        dl_cfg = tcfg.get("data_loader", {})
        bs = int(tcfg.batch_size)
        num_workers = int(dl_cfg.get("num_workers", 0))
        pin_memory = bool(dl_cfg.get("pin_memory", False))
        persistent_workers = (
            bool(dl_cfg.get("persistent_workers", False)) and num_workers > 0
        )
        prefetch_factor = (
            int(dl_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None
        )
        drop_last = bool(dl_cfg.get("drop_last", False))

        kwargs: dict = dict(
            batch_size=bs,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=self._collate_fn,
            drop_last=drop_last,
        )
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
        return kwargs

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "Call setup() before requesting dataloaders"
        return DataLoader(
            self.train_ds, shuffle=True, **self._dataloader_common_kwargs()
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds, shuffle=False, **self._dataloader_common_kwargs()
        )
