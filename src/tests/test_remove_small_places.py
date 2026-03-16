from __future__ import annotations

import unittest

import pandas as pd

from datapipelines.steps.filterplaces import RemoveSmallPlacesStep


class RemoveSmallPlacesStepTests(unittest.TestCase):
    def test_removes_all_images_from_places_below_threshold(self) -> None:
        dataframe = pd.DataFrame(
            [
                {
                    "image_id": "a1",
                    "place_id": "place_a",
                    "matches": ["a2", "b1", "c1"],
                },
                {"image_id": "a2", "place_id": "place_a", "matches": ["a1", "b1"]},
                {"image_id": "b1", "place_id": "place_b", "matches": ["a1", "b2"]},
                {"image_id": "b2", "place_id": "place_b", "matches": ["b1"]},
                {"image_id": "c1", "place_id": "place_c", "matches": ["a1"]},
            ]
        )
        step = RemoveSmallPlacesStep(min_images_per_place=2)

        result = step.run({"index": dataframe})

        filtered = result["index"]
        self.assertEqual(filtered["image_id"].tolist(), ["a1", "a2", "b1", "b2"])
        self.assertEqual(
            filtered["place_id"].tolist(), ["place_a", "place_a", "place_b", "place_b"]
        )
        self.assertEqual(
            filtered["matches"].tolist(),
            [["a2", "b1"], ["a1", "b1"], ["a1", "b2"], ["b1"]],
        )

    def test_context_threshold_can_disable_removal(self) -> None:
        dataframe = pd.DataFrame(
            [
                {"image_id": "a1", "place_id": "place_a"},
                {"image_id": "b1", "place_id": "place_b"},
            ]
        )
        step = RemoveSmallPlacesStep(
            min_images_per_place_context_key="min_images_per_place"
        )

        result = step.run({"index": dataframe, "min_images_per_place": 1})

        self.assertEqual(result["index"]["image_id"].tolist(), ["a1", "b1"])


if __name__ == "__main__":
    unittest.main()
