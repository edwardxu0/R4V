#!/usr/bin/env python
import argparse
import contextlib
import os
import pandas as pd
import shlex
import subprocess as sp
import time

from pathlib import Path


def memory_t(value):
    if isinstance(value, int):
        return value
    elif value.lower().endswith("g"):
        return int(value[:-1]) * 1_000_000_000
    elif value.lower().endswith("m"):
        return int(value[:-1]) * 1_000_000
    elif value.lower().endswith("k"):
        return int(value[:-1]) * 1000
    else:
        return int(value)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("property_csv", type=Path)
    parser.add_argument("verifier", choices=["eran", "neurify", "planet", "reluplex"])

    parser.add_argument(
        "-n",
        "--ntasks",
        type=int,
        default=float("inf"),
        help="The max number of running verification tasks.",
    )

    parser.add_argument(
        "-T", "--time", default=-1, type=float, help="The max running time in seconds."
    )
    parser.add_argument(
        "-M",
        "--memory",
        default=-1,
        type=memory_t,
        help="The max allowed memory in bytes.",
    )
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


def wait(pool, timeout=float("inf")):
    start_t = time.time()
    while time.time() - start_t < timeout:
        for index, task in enumerate(pool):
            if task.poll() is not None:
                return pool.pop(index)


def parse_verification_output(stdout, stderr):
    time = None
    if "finished successfully" in stderr:
        try:
            stdout_lines = stdout.split("\n")
            result = stdout_lines[-3].split()[-1]
            time = float(stdout_lines[-2].split()[-1])
        except:
            result = "error"
    elif "Out of Memory" in stderr:
        result = "outofmemory"
        stderr_lines = stderr.split("\n")
        time = float(stderr_lines[-3].split()[-3][:-2])
    elif "Timeout" in stderr:
        result = "timeout"
        stderr_lines = stderr.split("\n")
        time = float(stderr_lines[-3].split()[-3][:-2])
    else:
        result = "!"
    print("  result:", result)
    print("  time:", time)
    return result, time


def update_results(results_csv, network, prop, result, time):
    with lock(results_csv):
        df = pd.read_csv(results_csv)
        df.at[(df["Network"] == network) & (df["Property"] == prop), "Result"] = result
        df.at[(df["Network"] == network) & (df["Property"] == prop), "Time"] = time
        df.to_csv(results_csv, index=False)


def main(args):
    with lock(args.results_csv):
        if not args.results_csv.exists():
            with open(args.results_csv, "w+") as f:
                f.write("Network,Property,Result,Time\n")
    network_property_pairs = set()
    property_df = pd.read_csv(args.property_csv)
    for network in args.model_dir.iterdir():
        for prop in property_df["id"]:
            network_property_pairs.add((network.name, prop))

    pool = []
    while len(network_property_pairs) > 0:
        with lock(args.results_csv):
            df = pd.read_csv(args.results_csv)
            for row in df[["Network", "Property"]].itertuples():
                network = row.Network
                prop = row.Property
                network_property_pairs.discard((network, prop))
            if len(network_property_pairs) == 0:
                break
            network, prop = network_property_pairs.pop()
            df = df.append({"Network": network, "Property": prop}, ignore_index=True)
            df.to_csv(args.results_csv, index=False)

        property_filename = property_df[property_df["id"] == prop][
            "property_filename"
        ].values.item()
        resmonitor = "python ./tools/resmonitor.py"
        resmonitor_args = f"{resmonitor} -M {args.memory} -T {args.time}"
        verifier_args = f"python -m dnnv {args.model_dir / network} {property_filename} --{args.verifier}"
        run_args = f"{resmonitor_args} {verifier_args}"
        print(run_args)

        proc = sp.Popen(
            shlex.split(run_args), stdout=sp.PIPE, stderr=sp.PIPE, encoding="utf8"
        )
        proc.network = network
        proc.prop = prop
        pool.append(proc)

        while len(pool) >= args.ntasks:
            finished_task = wait(pool)
            print("FINISHED:", " ".join(proc.args))
            stdout = finished_task.stdout.read()
            stderr = finished_task.stderr.read()
            result, time = parse_verification_output(stdout, stderr)
            update_results(
                args.results_csv,
                finished_task.network,
                finished_task.prop,
                result,
                time,
            )
    while len(pool):
        finished_task = wait(pool)
        print("FINISHED:", " ".join(proc.args))
        stdout = finished_task.stdout.read()
        stderr = finished_task.stderr.read()
        result, time = parse_verification_output(stdout, stderr)
        update_results(
            args.results_csv, finished_task.network, finished_task.prop, result, time
        )


if __name__ == "__main__":
    main(_parse_args())
