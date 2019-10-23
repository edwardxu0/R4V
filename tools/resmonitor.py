import argparse
import os
import psutil
import shlex
import subprocess as sp
import time

import d4v.logging as logging


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
    parser = argparse.ArgumentParser(description="resmonitor - monitor resource usage")

    parser.add_argument(
        "-T", "--time", default=-1, type=int, help="The max running time in seconds."
    )
    parser.add_argument(
        "-M",
        "--memory",
        default=-1,
        type=memory_t,
        help="The max allowed memory in seconds.",
    )

    parser.add_argument("prog", nargs=argparse.REMAINDER, help="TEST")

    return parser.parse_args()


def get_memory_usage():
    try:
        p = psutil.Process(os.getpid())
        children = p.children(recursive=True)
        memory = 0
        for child in children:
            memory += child.memory_info().rss
    except psutil.NoSuchProcess:
        memory = 0
    return memory


def dispatch(prog, max_memory=-1, timeout=-1):
    logger = logging.getLogger(__name__)
    proc = sp.Popen(prog)

    start_t = time.time()
    try:
        while proc.poll() is None:
            time.sleep(0.01)
            now_t = time.time()
            duration_t = now_t - start_t
            mem_usage = get_memory_usage()
            if max_memory >= 0 and mem_usage >= max_memory:
                logger.error(
                    "Out of Memory (killing process): %d > %d", mem_usage, max_memory
                )
                proc.terminate()
                break
            if timeout >= 0 and duration_t >= timeout:
                logger.error("Timeout (killing process): %d > %d", duration_t, timeout)
                proc.terminate()
                break
        else:
            logger.info("Process finished successfully.")
            proc.terminate()
    except KeyboardInterrupt:
        logger.error("Received keyboard interupt (killing process)")
        proc.terminate()
    return proc.wait()


def main(args):
    logger = logging.initialize(
        __name__, argparse.Namespace(debug=False, quiet=False, verbose=True)
    )
    return dispatch(args.prog, max_memory=args.memory, timeout=args.time)


if __name__ == "__main__":
    main(_parse_args())
