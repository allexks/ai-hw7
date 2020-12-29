#!python3

"""62136."""


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
    print(args.dataset_file)
    print(args.clusters)
