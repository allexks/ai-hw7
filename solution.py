#!python3

"""62136."""

from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Point:
    x: float
    y: float


class KMeans:
    def __init__(self, clusters: int, points: List[Point]):
        self.clusters = clusters
        self.points_to_clusters_dict = {p: 0 for p in points}
        self.clusters_to_points_dict = {0: set(points)}

    def fit(self):
        pass #TODO

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
