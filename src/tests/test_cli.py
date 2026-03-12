from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch

import cli


class CliDatapipelineTests(unittest.TestCase):
    def test_datapipeline_handler_injects_min_images_per_place_into_context(self) -> None:
        args = argparse.Namespace(
            name="sf_xl_small",
            context=None,
            min_images_per_place=4,
        )

        with patch("cli.get_pipeline") as get_pipeline:
            pipeline = get_pipeline.return_value

            exit_code = cli._handle_datapipeline(args)

        self.assertEqual(exit_code, 0)
        pipeline.run.assert_called_once_with({"min_images_per_place": 4})


if __name__ == "__main__":
    unittest.main()
