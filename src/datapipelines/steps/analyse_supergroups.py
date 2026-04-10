from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .base import BaseStep
from .util import EmbeddingCache


class AnalyseSuperGroupStep(BaseStep):
    """Visualise sampled supergroups to inspect geographic and visual properties.

    For each sampled supergroup produces:

    1. **UTM scatter** — place centroids coloured by place_id.
    2. **Geographic distance histogram** — pairwise distances between place
       centroids, with a vertical line at the decision threshold.
    3. **Embedding similarity histogram** — pairwise cosine similarities
       between place embeddings.
    4. **Image grid** — sample images from each place in the supergroup.

    Outputs are written to
    ``$PLACEFORGE_PROCESSED_DIR/train/<name>/analysis/supergroups/``.

    Parameters
    ----------
    name:
        Dataset name (determines the output sub-directory).
    place_embedding_name:
        Name of the place embedding cache.
    num_supergroups:
        Number of supergroups to sample for visualisation.
    images_per_place:
        Max images to show per place in the image grid.
    max_places_in_grid:
        Max places to include in the image grid (for very large supergroups).
    decision_threshold:
        Geographic distance threshold (metres) drawn on histograms.
    seed:
        Random seed for reproducible sampling.
    """

    show_pbar = False

    def __init__(
        self,
        name: str,
        place_embedding_name: str,
        num_supergroups: int = 8,
        images_per_place: int = 4,
        max_places_in_grid: int = 20,
        decision_threshold: float = 25.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.name = name
        self.place_embedding_name = place_embedding_name
        self.num_supergroups = num_supergroups
        self.images_per_place = images_per_place
        self.max_places_in_grid = max_places_in_grid
        self.decision_threshold = decision_threshold
        self.seed = seed
        self.place_cache = EmbeddingCache(
            Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"])
            / "embedding"
            / "place"
            / place_embedding_name
        )

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = context["traindataset"]
        raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])
        out_dir = (
            Path(os.environ["PLACEFORGE_PROCESSED_DIR"])
            / "train"
            / self.name
            / "analysis"
            / "supergroups"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        has_utm = "utm_east" in df.columns and "utm_north" in df.columns
        has_emb = self.place_cache.exists

        # Load place embeddings if available
        place_embs = None
        place_emb_index = None
        if has_emb:
            place_embs = self.place_cache.mmap()
            place_emb_index = (
                self.place_cache.load_index().set_index("id")["row"]
            )

        # Sample supergroups
        rng = random.Random(self.seed)
        sg_ids = df["supergroup_id"].unique().tolist()
        n_sample = min(self.num_supergroups, len(sg_ids))
        sampled_sgs = rng.sample(sg_ids, n_sample)

        # Aggregate stats across all supergroups for a summary plot
        all_geo_spreads = []
        all_mean_sims = []
        all_min_dists = []

        for sg_id in sampled_sgs:
            sg_df = df[df["supergroup_id"] == sg_id]
            sg_dir = out_dir / f"supergroup_{sg_id}"
            sg_dir.mkdir(exist_ok=True)

            # Build place-level dataframe for this supergroup
            place_df = self._build_place_df(sg_df, place_embs, place_emb_index)
            n_places = len(place_df)

            # -- 1. UTM scatter --
            if has_utm and n_places > 1:
                self._plot_utm_scatter(
                    place_df, sg_id, sg_dir / "utm_scatter.png", plt
                )

            # -- 2. Geographic distance histogram --
            geo_dists = None
            if has_utm and n_places > 1:
                geo_dists = self._pairwise_geo_distances(place_df)
                self._plot_geo_hist(
                    geo_dists, sg_id, sg_dir / "geo_distance_hist.png", plt
                )
                spread = float(np.std(place_df["utm_east"].values)) + float(
                    np.std(place_df["utm_north"].values)
                )
                all_geo_spreads.append(spread)
                if len(geo_dists) > 0:
                    all_min_dists.append(float(geo_dists.min()))

            # -- 3. Embedding similarity histogram --
            if has_emb and n_places > 1 and "emb" in place_df.columns:
                cos_sims = self._pairwise_cosine_sims(place_df)
                self._plot_sim_hist(
                    cos_sims, sg_id, sg_dir / "embedding_sim_hist.png", plt
                )
                all_mean_sims.append(float(cos_sims.mean()))

            # -- 4. Image grid --
            self._plot_image_grid(
                sg_df, place_df, sg_id, raw_dir, sg_dir / "image_grid.png", plt
            )

            # -- 5. Per-supergroup stats text --
            self._write_sg_stats(
                sg_id, place_df, sg_df, geo_dists, sg_dir / "stats.txt"
            )

        # -- Summary plot across sampled supergroups --
        self._plot_summary(
            all_geo_spreads, all_mean_sims, all_min_dists, out_dir, plt
        )

        print(
            f"  Saved {n_sample} supergroup analyses → {out_dir}/"
        )
        return context

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _build_place_df(
        sg_df: pd.DataFrame,
        place_embs: np.ndarray | None,
        place_emb_index: pd.Series | None,
    ) -> pd.DataFrame:
        """One row per place with centroid UTM and optional embedding."""
        agg: dict[str, Any] = {"image_path": "count"}
        if "utm_east" in sg_df.columns:
            agg["utm_east"] = "mean"
            agg["utm_north"] = "mean"
        if "source" in sg_df.columns:
            agg["source"] = "first"

        place_df = sg_df.groupby("place_id").agg(agg).reset_index()
        place_df.rename(columns={"image_path": "num_images"}, inplace=True)

        if place_embs is not None and place_emb_index is not None:
            rows = []
            for pid in place_df["place_id"]:
                if pid in place_emb_index.index:
                    rows.append(place_embs[int(place_emb_index.loc[pid])])
                else:
                    rows.append(None)
            place_df["emb"] = rows

        return place_df

    @staticmethod
    def _pairwise_geo_distances(place_df: pd.DataFrame) -> np.ndarray:
        east = place_df["utm_east"].values
        north = place_df["utm_north"].values
        coords = np.stack([east, north], axis=1)
        diff = coords[:, None, :] - coords[None, :, :]
        dists = np.sqrt((diff**2).sum(axis=-1))
        # Upper triangle only (no diagonal)
        idx = np.triu_indices(len(dists), k=1)
        return dists[idx]

    @staticmethod
    def _pairwise_cosine_sims(place_df: pd.DataFrame) -> np.ndarray:
        embs = np.stack([e for e in place_df["emb"] if e is not None])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-8)
        sims = embs @ embs.T
        idx = np.triu_indices(len(sims), k=1)
        return sims[idx]

    def _plot_utm_scatter(
        self, place_df: pd.DataFrame, sg_id: int, path: Path, plt
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(
            place_df["utm_east"],
            place_df["utm_north"],
            c=range(len(place_df)),
            cmap="tab20",
            s=30,
            edgecolors="k",
            linewidths=0.3,
        )
        ax.set_xlabel("UTM East (m)")
        ax.set_ylabel("UTM North (m)")
        ax.set_title(f"Supergroup {sg_id} — {len(place_df)} places")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def _plot_geo_hist(
        self, dists: np.ndarray, sg_id: int, path: Path, plt
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(dists, bins=60, edgecolor="black", linewidth=0.3, alpha=0.8)
        ax.axvline(
            self.decision_threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"threshold = {self.decision_threshold}m",
        )
        ax.set_xlabel("Pairwise geographic distance (m)")
        ax.set_ylabel("Count")
        ax.set_title(f"Supergroup {sg_id} — pairwise place distances")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def _plot_sim_hist(
        self, sims: np.ndarray, sg_id: int, path: Path, plt
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(sims, bins=60, edgecolor="black", linewidth=0.3, alpha=0.8, color="steelblue")
        ax.set_xlabel("Pairwise cosine similarity")
        ax.set_ylabel("Count")
        ax.set_title(f"Supergroup {sg_id} — pairwise embedding similarity")
        ax.set_xlim(-0.1, 1.05)
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def _plot_image_grid(
        self,
        sg_df: pd.DataFrame,
        place_df: pd.DataFrame,
        sg_id: int,
        raw_dir: Path,
        path: Path,
        plt,
    ) -> None:
        rng = random.Random(self.seed)
        place_ids = place_df["place_id"].tolist()
        if len(place_ids) > self.max_places_in_grid:
            place_ids = rng.sample(place_ids, self.max_places_in_grid)

        n_rows = len(place_ids)
        n_cols = self.images_per_place
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 2.5, n_rows * 2.0),
            squeeze=False,
        )

        for row_i, pid in enumerate(place_ids):
            imgs = sg_df[sg_df["place_id"] == pid]["image_path"].tolist()
            if len(imgs) > n_cols:
                imgs = rng.sample(imgs, n_cols)

            for col_i in range(n_cols):
                ax = axes[row_i][col_i]
                if col_i < len(imgs):
                    try:
                        img = Image.open(raw_dir / imgs[col_i]).convert("RGB")
                        ax.imshow(img)
                    except Exception:
                        ax.text(
                            0.5, 0.5, "load error",
                            ha="center", va="center", transform=ax.transAxes,
                        )
                else:
                    ax.set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                if col_i == 0:
                    ax.set_ylabel(f"p{pid}", fontsize=7, rotation=0, labelpad=25)

        fig.suptitle(
            f"Supergroup {sg_id} — {n_rows} places  "
            f"({self.images_per_place} imgs/place)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _write_sg_stats(
        sg_id: int,
        place_df: pd.DataFrame,
        sg_df: pd.DataFrame,
        geo_dists: np.ndarray | None,
        path: Path,
    ) -> None:
        lines = [
            f"Supergroup {sg_id}",
            f"  places:          {len(place_df)}",
            f"  images:          {len(sg_df)}",
            f"  images/place:    {len(sg_df) / len(place_df):.1f}",
        ]
        if "source" in place_df.columns:
            counts = place_df["source"].value_counts()
            for src, cnt in counts.items():
                lines.append(f"  source {src}: {cnt} places")
        if geo_dists is not None and len(geo_dists) > 0:
            lines.append(f"  geo dist min:    {geo_dists.min():.1f} m")
            lines.append(f"  geo dist median: {float(np.median(geo_dists)):.1f} m")
            lines.append(f"  geo dist mean:   {geo_dists.mean():.1f} m")
            lines.append(f"  geo dist max:    {geo_dists.max():.1f} m")
            lines.append(f"  geo dist std:    {geo_dists.std():.1f} m")
        path.write_text("\n".join(lines) + "\n")

    def _plot_summary(
        self,
        geo_spreads: list[float],
        mean_sims: list[float],
        min_dists: list[float],
        out_dir: Path,
        plt,
    ) -> None:
        n_plots = sum(
            1 for lst in [geo_spreads, mean_sims, min_dists] if lst
        )
        if n_plots == 0:
            return

        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), squeeze=False)
        col = 0

        if geo_spreads:
            ax = axes[0][col]
            ax.hist(geo_spreads, bins=20, edgecolor="black", linewidth=0.3, alpha=0.8)
            ax.set_xlabel("UTM spread (std_east + std_north, m)")
            ax.set_title("Geographic spread per SG")
            col += 1

        if mean_sims:
            ax = axes[0][col]
            ax.hist(mean_sims, bins=20, edgecolor="black", linewidth=0.3, alpha=0.8, color="steelblue")
            ax.set_xlabel("Mean pairwise cosine similarity")
            ax.set_title("Visual coherence per SG")
            col += 1

        if min_dists:
            ax = axes[0][col]
            ax.hist(min_dists, bins=20, edgecolor="black", linewidth=0.3, alpha=0.8, color="coral")
            ax.axvline(
                self.decision_threshold,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"threshold = {self.decision_threshold}m",
            )
            ax.set_xlabel("Min pairwise distance (m)")
            ax.set_title("Closest negative per SG")
            ax.legend()

        fig.suptitle(f"Summary across {len(geo_spreads)} sampled supergroups", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "summary.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
