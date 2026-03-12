from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from datapipelines.base import Pipeline
from datapipelines.steps.base import BaseStep
from datapipelines.steps.filtermatches import RemoveUnmatchedImagesStep


class _RecordingStep(BaseStep):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.seen: list[tuple[str, int]] = []

    def run(self, context: dict[str, object]) -> dict[str, object]:
        self.seen.append((self._pipeline_name, self._progress_position))
        return context


class DatapipelineProgressTests(unittest.TestCase):
    def test_pipeline_assigns_a_stable_progress_row_to_each_step(self) -> None:
        first = _RecordingStep("first")
        second = _RecordingStep("second")
        pipeline = Pipeline("demo", [first, second])

        pipeline.run({})

        self.assertEqual(first.seen, [("demo", 0)])
        self.assertEqual(second.seen, [("demo", 1)])

    def test_progress_stays_on_screen_after_completion(self) -> None:
        step = _RecordingStep("example")
        step._pipeline_name = "demo"
        step._progress_position = 3

        with patch("datapipelines.steps.base.tqdm") as tqdm_mock:
            step.progress(total=10, desc="work")

        tqdm_mock.assert_called_once_with(
            total=10,
            desc="demo | work",
            dynamic_ncols=True,
            leave=True,
            position=0,
        )

    def test_steps_without_inner_loops_still_emit_a_progress_bar(self) -> None:
        step = RemoveUnmatchedImagesStep()
        step._pipeline_name = "demo"
        step._progress_position = 2

        with patch("datapipelines.steps.base.tqdm") as tqdm_mock:
            progress = tqdm_mock.return_value.__enter__.return_value
            result = step.run({"index": pd.DataFrame([{"matches": ["a"]}])})

        self.assertEqual(result["index"]["matches"].tolist(), [["a"]])
        tqdm_mock.assert_called_once_with(
            total=1,
            desc="demo | remove unmatched images",
            dynamic_ncols=True,
            leave=True,
            position=0,
        )
        progress.update.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
