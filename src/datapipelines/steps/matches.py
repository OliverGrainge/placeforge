from __future__ import annotations

from typing import Any
from sklearn.neighbors import BallTree

from .base import BaseStep


class ComputeValMatchesStep(BaseStep):
    def __init__(self, radius_meters: int) -> None:
        self.radius_meters = radius_meters

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["valdataset"].copy()

        coords = df[["utm_east", "utm_north"]].values
        image_ids = df["image_id"].values

        tree = BallTree(coords, metric="euclidean")
        indices = tree.query_radius(coords, r=self.radius_meters)

        df["matches"] = None
        query_mask = df["is_query"]

        for i in df.index[query_mask]:
            neighbors = indices[i]
            db_neighbors = neighbors[neighbors != i]
            # Further filter to only database image indices
            db_neighbors = db_neighbors[~df["is_query"].iloc[db_neighbors].values]
            df.at[i, "matches"] = image_ids[db_neighbors].tolist()

        # Drop query rows with no database matches
        no_match_mask = query_mask & df["matches"].apply(
            lambda m: m is not None and len(m) == 0
        )
        df = df[~no_match_mask].reset_index(drop=True)

        return {**context, "valdataset": df}


class ComputeTestMatchesStep(BaseStep):
    def __init__(self, radius_meters: int) -> None:
        self.radius_meters = radius_meters

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["testdataset"].copy()

        coords = df[["utm_east", "utm_north"]].values
        image_ids = df["image_id"].values

        tree = BallTree(coords, metric="euclidean")
        indices = tree.query_radius(coords, r=self.radius_meters)

        df["matches"] = None
        query_mask = df["is_query"]

        for i in df.index[query_mask]:
            neighbors = indices[i]
            db_neighbors = neighbors[neighbors != i]
            db_neighbors = db_neighbors[~df["is_query"].iloc[db_neighbors].values]
            df.at[i, "matches"] = image_ids[db_neighbors].tolist()

        no_match_mask = query_mask & df["matches"].apply(
            lambda m: m is not None and len(m) == 0
        )
        df = df[~no_match_mask].reset_index(drop=True)

        return {**context, "testdataset": df}
