from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from datapipelines import get_pipeline, list_pipelines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="placeforge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("config", type=Path, help="Path to a YAML config file")
    train_parser.set_defaults(handler=_handle_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.set_defaults(handler=_handle_eval)

    datapipeline_parser = subparsers.add_parser(
        "datapipeline",
        help="Run a registered data pipeline",
    )
    datapipeline_parser.add_argument(
        "name",
        nargs="?",
        help="Registered pipeline name",
    )
    datapipeline_parser.add_argument(
        "--context",
        type=Path,
        help="Optional path to a Python file defining INITIAL_CONTEXT as a dict",
    )
    datapipeline_parser.add_argument(
        "--list",
        action="store_true",
        help="List registered pipelines and exit",
    )
    datapipeline_parser.set_defaults(handler=_handle_datapipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


def _handle_train(args: argparse.Namespace) -> int:
    import pytorch_lightning as pl

    from datamodules import get_datamodule
    from modules import get_module, get_transform

    config = yaml.safe_load(args.config.read_text())

    trainer_kwargs: dict[str, Any] = config.get("trainer", {})
    module_kwargs: dict[str, Any] = config.get("module", {})
    datamodule_kwargs: dict[str, Any] = config.get("datamodule", {})

    module_name = module_kwargs.pop("name")
    datamodule_name = datamodule_kwargs.pop("name")

    wandb_kwargs: dict[str, Any] | None = config.get("wandb")
    if wandb_kwargs is not None:
        from pytorch_lightning.loggers import WandbLogger
        trainer_kwargs["logger"] = WandbLogger(**wandb_kwargs)

    transform_kwargs: dict[str, Any] | None = datamodule_kwargs.pop("transform", None)
    if transform_kwargs is not None:
        transform_name = transform_kwargs.pop("name")
        datamodule_kwargs["transform"] = get_transform(transform_name, **transform_kwargs)

    module = get_module(module_name, **module_kwargs)
    datamodule = get_datamodule(datamodule_name, **datamodule_kwargs)
    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(module, datamodule=datamodule)
    return 0


def _handle_eval(args: argparse.Namespace) -> int:
    raise NotImplementedError("`eval` is not implemented yet")


def _handle_datapipeline(args: argparse.Namespace) -> int:
    if args.list:
        available = list_pipelines()
        print("Available datapipelines:")
        for pipeline_name in available:
            print(f"- {pipeline_name}")
        if not available:
            print("- <none>")
        return 0

    if args.name is None:
        raise SystemExit("datapipeline name is required unless --list is used")

    pipeline = get_pipeline(args.name)
    context = _load_initial_context(args.context) if args.context else {}
    pipeline.run(context)
    return 0


def _load_initial_context(path: Path) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    source = path.read_text()
    exec(compile(source, str(path), "exec"), namespace)

    initial_context = namespace.get("INITIAL_CONTEXT")
    if initial_context is None:
        raise ValueError(f"{path} does not define INITIAL_CONTEXT")
    if not isinstance(initial_context, dict):
        raise TypeError(f"INITIAL_CONTEXT in {path} must be a dict")

    return initial_context


if __name__ == "__main__":
    raise SystemExit(main())
