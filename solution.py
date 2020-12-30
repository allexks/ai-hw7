#!python3

"""62136."""

from dataclasses import dataclass
from typing import List
from random import random
from math import sqrt, inf

@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance_to(self, other) -> float:
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class KMeans:
    @dataclass(frozen=True)
    class Mean:
        id: int
        pos: Point

    def __init__(self, clusters: int, points: List[Point]):
        self.clusters_count = clusters
        self.points = points
        self.points_to_clusters_dict = {p: 0 for p in points}
        self.clusters_to_points_dict = {
            i: set(points) if i == 0 else set()
            for i in range(clusters)
        }
        self._max_x = max(map(lambda p: p.x, points))
        self._max_y = max(map(lambda p: p.y, points))

    def fit(self):
        while True: # loops until no 0-size cluster exist
            means = self._random_means_init()
            self._assignment_step(means)

            if not any(map(
                lambda i: len(self.clusters_to_points_dict[i]) == 0,
                range(self.clusters_count)
            )):
                break

        iteration = 0
        is_over = False
        while not is_over:
            if iteration > 0:
                self._assignment_step(means)

            is_over = self._update_step(means)

            debug = "-".join(str(len(self.clusters_to_points_dict[i])) for i in range(self.clusters_count))
            print(f"({iteration}) {debug}")
            iteration += 1

    def _random_means_init(self):
        return [
            KMeans.Mean(
                id,
                Point(
                    random() * self._max_x,
                    random() * self._max_y
                )
            )
             for id in range(self.clusters_count)
        ]

    def _assignment_step(self, means):
        for point in self.points:
            nearest_mean = min(means, key=lambda m: point.distance_to(m.pos))
            prev_cluster_id = self.points_to_clusters_dict[point]
            if prev_cluster_id != nearest_mean.id:
                self.points_to_clusters_dict[point] = nearest_mean.id
                self.clusters_to_points_dict[prev_cluster_id].remove(point)
                self.clusters_to_points_dict[nearest_mean.id].add(point)

    def _update_step(self, means):
        is_over = True
        for mean in means:
            cluster_points = self.clusters_to_points_dict[mean.id]
            new_x = sum(p.x for p in cluster_points) / len(cluster_points)
            new_y = sum(p.y for p in cluster_points) / len(cluster_points)
            if new_x == mean.pos.x and new_y == mean.pos.y:
                continue  # No update to this mean
            else:
                is_over = False  # Update this mean which means another iteration should come.
                means[mean.id] = KMeans.Mean(mean.id, Point(new_x, new_y))
        return is_over


if __name__ == "__main__":
    from argparse import ArgumentParser, FileType
    parser = ArgumentParser(description="Apply kMeans to clusterize a dataset.")
    parser.add_argument(
        "-f",
        "--file",
        help="The file with the dataset.",
        required=True,
        type=FileType("r"),
        action="store",
        dest="dataset_file"
    )
    parser.add_argument(
        "-c",
        "--clusters",
        help="Clusters count",
        required=True,
        type=int,
        action="store",
        dest="clusters"
    )

    args = parser.parse_args()

    points = [
        Point(p[0], p[1])
        for p in (
            [*map(float, line.split())]
            for line in args.dataset_file.readlines()
            if len(line.split()) == 2
        )
    ]

    args.dataset_file.close()

    kMeans = KMeans(args.clusters, points)
    kMeans.fit()

    import matplotlib.pyplot as plt
    x = [p.x for p in points]
    y = [p.y for p in points]
    c = [kMeans.points_to_clusters_dict[p] for p in points]
    plt.scatter(x=x, y=y, c=c)
    plt.show()
