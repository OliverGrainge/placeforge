#!/usr/bin/env python3
"""Debug utilities for placeforge."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def cmd_dataloader(args: argparse.Namespace) -> int:
    """Load one batch from the train dataloader and visualize it."""
    from datamodules.datasets.train import PlaceImageTrainDataset, SupergroupBatchSampler

    dataset = PlaceImageTrainDataset(
        args.index_path,
        images_per_place=args.images_per_place,
        load_images=True,
        seed=args.seed,
    )

    sampler = SupergroupBatchSampler(
        dataset.valid_supergroup_ids,
        places_per_batch=args.places_per_batch,
        shuffle=True,
        drop_last=True,
        seed=args.seed,
    )

    batch_indices = next(iter(sampler))
    places = [dataset[idx] for idx in batch_indices]

    _visualize_batch(places, output=args.output)
    return 0


def _visualize_batch(places: list[dict[str, Any]], output: Path | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Visualization requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc

    n_places = len(places)
    images_per_place = len(places[0]["items"])

    fig, axes = plt.subplots(
        n_places,
        images_per_place,
        figsize=(2.5 * images_per_place, 2.5 * n_places),
        squeeze=False,
    )

    for row, place_data in enumerate(places):
        place_id = place_data["items"][0].get("place_id", place_data["anchor_image_id"])
        for col, item in enumerate(place_data["items"]):
            ax = axes[row][col]
            ax.imshow(item["image"])
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(
                    f"place {place_id}",
                    fontsize=7,
                    rotation=0,
                    labelpad=55,
                    va="center",
                )
            if row == 0:
                label = "anchor" if col == 0 else f"positive {col}"
                ax.set_title(label, fontsize=8)

    fig.suptitle(
        f"{n_places} places × {images_per_place} images/place  —  "
        "columns: intra-class variation,  rows: inter-class variation",
        fontsize=9,
    )
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="placeforge-debug")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dl_parser = subparsers.add_parser(
        "dataloader",
        help="Visualize one batch from the train dataloader",
    )
    dl_parser.add_argument("index_path", type=Path, help="Path to training index (.parquet)")
    dl_parser.add_argument(
        "--places-per-batch",
        type=int,
        default=4,
        help="Number of places (rows) to show (default: 4)",
    )
    dl_parser.add_argument(
        "--images-per-place",
        type=int,
        default=4,
        help="Number of images per place (columns) to show (default: 4)",
    )
    dl_parser.add_argument("--seed", type=int, default=0)
    dl_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save figure to this path instead of displaying it",
    )
    dl_parser.set_defaults(handler=cmd_dataloader)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
