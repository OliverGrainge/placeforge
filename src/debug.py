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
        place_id = place_data.get("place_id")
        if place_id is None and place_data["items"]:
            place_id = place_data["items"][0].get("place_id")
        if place_id is None:
            place_id = place_data.get("anchor_image_id", "<unknown>")
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


class _SimilarityModel:
    """Lazily-loaded DINOv2 backbone for computing intraclass similarities."""

    _instance: "_SimilarityModel | None" = None

    @classmethod
    def get(cls) -> "_SimilarityModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading DINOv2 backbone...", end=" ", flush=True)
        self._backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=True, verbose=False
        )
        self._backbone.eval().to(self.device)
        print("done")

    def _embed(self, images: list[Any]) -> "Any":
        """Return L2-normalised embeddings as a numpy array of shape (N, D)."""
        import torch
        import torch.nn.functional as F
        from torchvision import transforms

        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensors = torch.stack([preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            features = self._backbone.forward_features(tensors)
            patch_tokens = features["x_norm_patchtokens"]
            embeddings = F.normalize(patch_tokens.mean(dim=1), dim=-1)
        return embeddings.cpu().numpy()

    def similarity_matrix(self, images: list[Any]) -> "Any":
        """Return an (N, N) pairwise cosine similarity matrix."""
        embs = self._embed(images)
        return (embs @ embs.T).astype("float32")


def _browse_places(dataset: Any, start_index: int = 0, show_sim: bool = False) -> None:
    """Interactive place browser — keyboard navigation through every place."""
    try:
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Browsing requires matplotlib and Pillow. "
            "Install them with `pip install matplotlib pillow`."
        ) from exc

    if show_sim:
        _SimilarityModel.get()  # load model before drawing anything

    n_places = len(dataset)
    state = {"idx": start_index % n_places}


    def _load_all_images(idx: int) -> tuple[str, int, list[Any]]:
        place_id, supergroup_id, image_ids = dataset._places[idx]
        images = []
        for iid in image_ids:
            record = dataset._image_id_to_record[iid]
            path = record[dataset.image_path_column]
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        return str(place_id), int(supergroup_id), images

    def _render(idx: int) -> None:
        place_id, supergroup_id, images = _load_all_images(idx)
        n = len(images)
        cols = min(n, 6)
        img_rows = (n + cols - 1) // cols

        fig.clf()

        if show_sim and n >= 2:
            gs = gridspec.GridSpec(
                img_rows, cols + 1,
                figure=fig,
                width_ratios=[1.0] * cols + [0.6],
                hspace=0.08, wspace=0.08,
            )
            img_axes = [[fig.add_subplot(gs[r, c]) for c in range(cols)] for r in range(img_rows)]
            heatmap_ax = fig.add_subplot(gs[:, cols])
        else:
            gs = gridspec.GridSpec(img_rows, cols, figure=fig, hspace=0.08, wspace=0.04)
            img_axes = [[fig.add_subplot(gs[r, c]) for c in range(cols)] for r in range(img_rows)]
            heatmap_ax = None

        for i, img in enumerate(images):
            r, c = divmod(i, cols)
            img_axes[r][c].imshow(img)
            img_axes[r][c].axis("off")

        for i in range(n, img_rows * cols):
            r, c = divmod(i, cols)
            img_axes[r][c].set_visible(False)

        if heatmap_ax is not None:
            _render_sim_heatmap(images, heatmap_ax)

        fig.suptitle(
            f"place {place_id}  |  supergroup {supergroup_id}  |  {n} images  |  "
            f"[{idx + 1} / {n_places}]          "
            "← / → navigate    r random    q quit",
            fontsize=9,
        )
        fig.tight_layout()
        fig.canvas.draw_idle()

    def _on_key(event: Any) -> None:
        import random as _random

        if event.key in ("right", "n"):
            state["idx"] = (state["idx"] + 1) % n_places
        elif event.key in ("left", "p"):
            state["idx"] = (state["idx"] - 1) % n_places
        elif event.key == "r":
            state["idx"] = _random.randrange(n_places)
        elif event.key in ("q", "escape"):
            plt.close(fig)
            return
        else:
            return
        _render(state["idx"])

    fig = plt.figure(figsize=(14, 8))
    fig.canvas.mpl_connect("key_press_event", _on_key)
    _render(state["idx"])
    plt.show()


def _render_sim_heatmap(
    images: list[Any],
    ax: Any,
    *,
    facecolor: str = "#1a1a1a",
    label_color: str = "#aaa",
) -> None:
    """Draw a pairwise cosine similarity heatmap onto an existing Axes."""
    n = len(images)
    sim_matrix = _SimilarityModel.get().similarity_matrix(images)

    im = ax.imshow(sim_matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="equal")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=max(5, 8 - n // 3), color="black")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(i) for i in range(n)], fontsize=7, color=label_color)
    ax.set_yticklabels([str(i) for i in range(n)], fontsize=7, color=label_color)
    ax.tick_params(length=0)
    ax.set_facecolor(facecolor)
    for spine in ax.spines.values():
        spine.set_visible(False)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = ax.get_figure().colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, colors=label_color)
    cb.outline.set_visible(False)


def _render_sim_heatmap_b64(images: list[Any]) -> str:
    """Render the pairwise similarity heatmap to a base64 PNG string."""
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    size = max(2.5, 0.45 * n)
    fig, ax = plt.subplots(figsize=(size + 0.6, size))
    fig.patch.set_facecolor("#1a1a1a")
    _render_sim_heatmap(images, ax)
    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _browse_places_web(dataset: Any, start_index: int = 0, port: int = 8765, show_sim: bool = False) -> None:
    """Interactive place browser served over HTTP — works on headless / SSH machines."""
    import base64
    import io
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Browsing requires Pillow. Install it with `pip install pillow`."
        ) from exc

    if show_sim:
        _SimilarityModel.get()  # load model before serving

    n_places = len(dataset)

    def _load_images(idx: int) -> tuple[str, int, list[Any]]:
        place_id, supergroup_id, image_ids = dataset._places[idx]
        images = []
        for iid in image_ids:
            record = dataset._image_id_to_record[iid]
            path = record[dataset.image_path_column]
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        return str(place_id), int(supergroup_id), images

    def _pil_to_b64(img: Any) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            idx = int(params.get("idx", [start_index])[0]) % n_places

            place_id, supergroup_id, images = _load_images(idx)
            prev_idx = (idx - 1) % n_places
            next_idx = (idx + 1) % n_places

            img_tags = "".join(
                f'<img src="data:image/jpeg;base64,{_pil_to_b64(img)}">'
                for img in images
            )

            sim_block = ""
            if show_sim and len(images) >= 2:
                heatmap_b64 = _render_sim_heatmap_b64(images)
                sim_block = (
                    f'<div class="sim">'
                    f'<img src="data:image/png;base64,{heatmap_b64}" style="max-width:480px">'
                    f"</div>"
                )

            html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>placeforge browse</title>
<style>
  body {{ margin: 0; background: #111; color: #ddd; font-family: monospace; text-align: center; }}
  h2 {{ margin: 14px 0 4px; font-size: 1em; font-weight: normal; color: #aaa; }}
  .nav {{ margin: 10px; }}
  .nav a {{ color: #6af; text-decoration: none; margin: 0 14px; font-size: 1.3em; }}
  .grid {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 6px; padding: 12px; }}
  .grid img {{ max-height: 280px; max-width: 280px; object-fit: cover; border-radius: 3px; }}
  .sim {{ margin: 4px auto 16px; }}
</style>
<script>
document.addEventListener("keydown", e => {{
  if (e.key === "ArrowRight" || e.key === "n") location.href = "/?idx={next_idx}";
  if (e.key === "ArrowLeft"  || e.key === "p") location.href = "/?idx={prev_idx}";
  if (e.key === "r") location.href = "/?idx=" + Math.floor(Math.random() * {n_places});
}});
</script>
</head><body>
<h2>place <b>{place_id}</b> &nbsp;|&nbsp; supergroup <b>{supergroup_id}</b>
    &nbsp;|&nbsp; <b>{len(images)}</b> images
    &nbsp;|&nbsp; [{idx + 1} / {n_places}]</h2>
<div class="nav">
  <a href="/?idx={prev_idx}">&#8592; prev</a>
  <a href="/?idx={next_idx}">next &#8594;</a>
</div>
<div class="grid">{img_tags}</div>
{sim_block}
</body></html>"""

            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:  # suppress request logs
            pass

    server = HTTPServer(("127.0.0.1", port), _Handler)
    print(f"  http://localhost:{port}/?idx={start_index}")
    print("  ← / → keys or prev/next links to navigate    Ctrl-C to quit")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


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
