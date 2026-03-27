from __future__ import annotations

import unittest
from unittest import mock

from datamodules.datamodule import DataModule


class _DummyTrainDataset:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def get_batch_sampler(self, _batch_size: int, drop_last: bool = False):
        assert drop_last is True
        return [[0]]

    def __getitem__(self, index: int):
        return {"index": index}

    def __len__(self) -> int:
        return 1

    @staticmethod
    def collate_fn(batch):
        return batch


class DataLoaderOrderTests(unittest.TestCase):
    def test_train_dataloader_disables_in_order_delivery(self) -> None:
        with mock.patch("datamodules.datamodule.ContrastiveTrainDataset", _DummyTrainDataset):
            datamodule = DataModule(
                train_dataset_name="ignored",
                val_dataset_names=[],
                batch_size=16,
                num_workers=2,
            )
            datamodule.setup("fit")

            loader = datamodule.train_dataloader()

        self.assertFalse(loader.in_order)


if __name__ == "__main__":
    unittest.main()
