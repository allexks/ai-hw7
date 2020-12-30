#!python3

"""62136."""

from dataclasses import dataclass
from typing import List, Tuple
from random import random
from math import sqrt, inf
import matplotlib.pyplot as plt


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
        means = self._random_means_init()

        is_over = False
        iteration = 0
        while not is_over:
            self._assignment_step(means)

            is_over = self._update_step(means)

            debug = "-".join(str(len(self.clusters_to_points_dict[i])) for i in range(self.clusters_count))
            print(f"({iteration}) {debug}")
            iteration += 1

    def evaluate_clusters(self) -> Tuple[float, float]:
        """
        Use single-linkage intercluster distance evaluation.
        Return (minimum_distance, average_distance).
        """
        cl_eval = []
        for cl1_id in range(self.clusters_count):
            for cl2_id in filter(lambda i: i != cl1_id, range(self.clusters_count)):
                cl_eval.append(min(
                    p1.distance_to(p2)
                    for p1 in self.clusters_to_points_dict[cl1_id]
                    for p2 in self.clusters_to_points_dict[cl2_id]
                    if p1 != p2
                ))

        return min(cl_eval), (sum(cl_eval) / len(cl_eval))

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

            if len(cluster_points) == 0:
                # Give it a new random location.
                new_x = random() * self._max_x
                new_y = random() * self._max_y
            else:
                new_x = sum(p.x for p in cluster_points) / len(cluster_points)
                new_y = sum(p.y for p in cluster_points) / len(cluster_points)

            if new_x == mean.pos.x and new_y == mean.pos.y:
                continue  # No update to this mean
            else:
                is_over = False  # Update this mean which means another iteration should come.
                means[mean.id] = KMeans.Mean(mean.id, Point(new_x, new_y))

        return is_over


def plot_solution(solution: KMeans):
        x = [p.x for p in solution.points]
        y = [p.y for p in solution.points]
        c = [solution.points_to_clusters_dict[p] for p in solution.points]
        plt.scatter(x=x, y=y, c=c)
        plt.show()


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

    solution = None
    solution_eval = None

    for _ in range(3):
        kMeans = KMeans(args.clusters, points)
        kMeans.fit()
        print("Evaluating clusters...")
        current = kMeans.evaluate_clusters()
        print("Current solution: min={0}, avg={1}".format(*current))
        if solution is None:
            solution = kMeans
            solution_eval = current
        elif solution_eval[0] < current[0] or (solution_eval[0] == current[0] and solution_eval[1] < current[1]):
            print("Current is better.")
            solution = kMeans
            solution_eval = current
        else:
            print("Current is not better.")

        plot_solution(kMeans)

    assert solution is not None, "No solution."

    plot_solution(solution)


