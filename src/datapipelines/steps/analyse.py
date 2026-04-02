from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .base import BaseStep


class AnalyseTrainDatasetStep(BaseStep):
    """Produce basic statistics and sample place visualisations.

    Outputs are written to
    ``$PLACEFORGE_PROCESSED_DIR/train/<name>/analysis/``:

    - ``stats.json`` — dataset-level statistics.
    - ``sample_places/place_<id>.png`` — image grids for sampled places.

    Parameters
    ----------
    name:
        Dataset name (determines the output sub-directory).
    num_places:
        Number of places to sample for visualisation.
    seed:
        Random seed for reproducible sampling.
    """

    show_pbar = False

    def __init__(
        self,
        name: str,
        num_places: int = 16,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.name = name
        self.num_places = num_places
        self.seed = seed

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
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Stats ────────────────────────────────────────────────────────
        stats = self._compute_stats(df)
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

        self._print_stats(stats)

        # ── Sample place figures ─────────────────────────────────────────
        figures_dir = out_dir / "sample_places"
        figures_dir.mkdir(exist_ok=True)

        place_ids = df["place_id"].unique()
        rng = random.Random(self.seed)
        sampled = rng.sample(
            list(place_ids), min(self.num_places, len(place_ids))
        )

        for place_id in sampled:
            place_df = df[df["place_id"] == place_id]
            image_paths = place_df["image_path"].tolist()

            n_images = len(image_paths)
            n_cols = min(n_images, 4)
            n_rows = (n_images + n_cols - 1) // n_cols

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(n_cols * 2.5, n_rows * 2.5),
                squeeze=False,
            )

            sg_id = place_df["supergroup_id"].iloc[0] if "supergroup_id" in place_df.columns else "?"

            for i, rel_path in enumerate(image_paths):
                row, col = divmod(i, n_cols)
                ax = axes[row][col]
                try:
                    img = Image.open(raw_dir / rel_path).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "load error", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

            for i in range(n_images, n_rows * n_cols):
                row, col = divmod(i, n_cols)
                axes[row][col].set_visible(False)

            fig.suptitle(
                f"place {place_id}  (sg={sg_id}, {len(place_df)} images)",
                fontsize=10,
            )
            fig.tight_layout()
            fig.savefig(figures_dir / f"place_{place_id}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

        print(f"  Saved {len(sampled)} place figures → {figures_dir}/")

        return context

    @staticmethod
    def _compute_stats(df: pd.DataFrame) -> dict[str, Any]:
        num_images = len(df)
        images_per_place = df.groupby("place_id").size()
        num_places = len(images_per_place)

        stats: dict[str, Any] = {
            "num_images": num_images,
            "num_places": num_places,
            "images_per_place": {
                "mean": float(images_per_place.mean()),
                "median": float(images_per_place.median()),
                "min": int(images_per_place.min()),
                "max": int(images_per_place.max()),
                "std": float(images_per_place.std()),
            },
        }

        if "supergroup_id" in df.columns:
            places_per_sg = df.groupby("supergroup_id")["place_id"].nunique()
            images_per_sg = df.groupby("supergroup_id").size()
            stats["num_supergroups"] = int(df["supergroup_id"].nunique())
            stats["places_per_supergroup"] = {
                "mean": float(places_per_sg.mean()),
                "median": float(places_per_sg.median()),
                "min": int(places_per_sg.min()),
                "max": int(places_per_sg.max()),
                "std": float(places_per_sg.std()),
            }
            stats["images_per_supergroup"] = {
                "mean": float(images_per_sg.mean()),
                "median": float(images_per_sg.median()),
                "min": int(images_per_sg.min()),
                "max": int(images_per_sg.max()),
                "std": float(images_per_sg.std()),
            }

        if "source" in df.columns:
            source_stats = {}
            for source, source_df in df.groupby("source"):
                src_ipp = source_df.groupby("place_id").size()
                entry: dict[str, Any] = {
                    "num_images": int(len(source_df)),
                    "num_places": int(source_df["place_id"].nunique()),
                    "pct_images": float(len(source_df) / num_images * 100),
                    "pct_places": float(source_df["place_id"].nunique() / num_places * 100),
                    "images_per_place": {
                        "mean": float(src_ipp.mean()),
                        "median": float(src_ipp.median()),
                        "min": int(src_ipp.min()),
                        "max": int(src_ipp.max()),
                        "std": float(src_ipp.std()) if len(src_ipp) > 1 else 0.0,
                    },
                }
                if "supergroup_id" in source_df.columns:
                    entry["num_supergroups"] = int(source_df["supergroup_id"].nunique())
                source_stats[str(source)] = entry
            stats["sources"] = source_stats

        return stats

    @staticmethod
    def _print_stats(stats: dict[str, Any]) -> None:
        W = 60
        print()
        print("=" * W)
        print("  Train Dataset Summary")
        print("-" * W)
        print(f"    images:       {stats['num_images']:>8,}")
        print(f"    places:       {stats['num_places']:>8,}")
        ipp = stats["images_per_place"]
        print(
            f"    images/place: avg={ipp['mean']:.1f}  "
            f"min={ipp['min']}  max={ipp['max']}  "
            f"std={ipp['std']:.1f}"
        )
        if "num_supergroups" in stats:
            print(f"    supergroups:  {stats['num_supergroups']:>8,}")
            pps = stats["places_per_supergroup"]
            print(
                f"    places/sg:    avg={pps['mean']:.1f}  "
                f"min={pps['min']}  max={pps['max']}  "
                f"std={pps['std']:.1f}"
            )
            ips = stats["images_per_supergroup"]
            print(
                f"    images/sg:    avg={ips['mean']:.1f}  "
                f"min={ips['min']}  max={ips['max']}  "
                f"std={ips['std']:.1f}"
            )

        if "sources" in stats:
            print("-" * W)
            print("  Per-Source Breakdown")
            print("-" * W)
            # Header
            print(
                f"    {'source':<20} {'images':>8} {'places':>8} "
                f"{'%img':>6} {'%plc':>6} {'img/plc':>8}"
            )
            print(f"    {'-'*20} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*8}")
            for src_name, src in stats["sources"].items():
                src_ipp = src["images_per_place"]
                print(
                    f"    {src_name:<20} {src['num_images']:>8,} {src['num_places']:>8,} "
                    f"{src['pct_images']:>5.1f}% {src['pct_places']:>5.1f}% "
                    f"{src_ipp['mean']:>7.1f}"
                )

        print("=" * W)
        print()
