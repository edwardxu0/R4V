#!/usr/bin/env python
import argparse
import contextlib
import multiprocessing as mp
import os
import pandas as pd

from pathlib import Path
from verifiers import neurify


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("property_csv", type=Path)
    return parser.parse_args()


@contextlib.contextmanager
def lock(filename: Path, *args, **kwargs):
    lock_filename = filename.with_suffix(".lock")
    try:
        while True:
            try:
                lock_fd = os.open(lock_filename, os.O_CREAT | os.O_WRONLY | os.O_EXCL)
                break
            except IOError as e:
                pass
        yield
    finally:
        os.close(lock_fd)
        os.remove(lock_filename)


def main(args):
    with lock(args.results_csv):
        if not args.results_csv.exists():
            with open(args.results_csv, "w+") as f:
                f.write("Network,Property,Result,Time\n")
    network_property_pairs = set()
    property_df = pd.read_csv(args.property_csv)
    for network in args.model_dir.iterdir():
        for prop in property_df["id"]:
            print(network, prop)
            network_property_pairs.add((network.name, prop))

    pool = []
    while len(network_property_pairs) > 0:
        with lock(args.results_csv):
            df = pd.read_csv(args.results_csv)
            for row in df[["Network", "Property"]].itertuples():
                network = row.Network
                prop = row.Property
                # print(network, prop)
                network_property_pairs.discard((network, prop))
            network, prop = network_property_pairs.pop()
            df = df.append({"Network": network, "Property": prop}, ignore_index=True)
            df.to_csv(args.results_csv, index=False)
        print("VERIFYING", network, prop)
        prop_x = Path(
            property_df[property_df["id"] == prop]["image_filename"]
        ).with_suffix("neurify")
        prop_e = None
        prop_lb = np.tan(
            min(
                -np.pi / 2,
                property_df[property_df["id"] == prop]["steering_angle_lb"] / 2.0,
            )
        )
        prop_ub = np.tan(
            max(
                np.pi / 2,
                property_df[property_df["id"] == prop]["steering_angle_ub"] / 2.0,
            )
        )
        pool.append(
            mp.Process(
                neurify.run,
                args=(args.model_dir / network, prop_x, prob_e, prop_lb, prop_ub),
            )
        )
        with lock(args.results_csv):
            df = pd.read_csv(args.results_csv)
            df.at[
                (df["Network"] == network) & (df["Property"] == prop), "Result"
            ] = "done"
            df.at[(df["Network"] == network) & (df["Property"] == prop), "Time"] = 0.0
            df.to_csv(args.results_csv, index=False)


if __name__ == "__main__":
    main(_parse_args())
