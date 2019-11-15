#!/usr/bin/env python
import os
import pandas as pd
import sys

from datetime import datetime
from pathlib import Path


def main():
    original_dir = os.getcwd()
    results_dir = "."
    csv_name = "results.distillation"
    if len(sys.argv) >= 2:
        results_dir = sys.argv[1]
        csv_name = Path(sys.argv[1]).name
    re_extract = False
    if len(sys.argv) >= 3:
        re_extract = True
    os.chdir(results_dir)
    if not Path("tmp").exists() or re_extract:
        os.system("ls *.model.tar.gz | xargs -I {} tar -xzf {}")
    if not Path("models").exists() or re_extract:
        Path("models").mkdir(exist_ok=True)
        os.system(
            "for model in tmp/*/*/model.onnx;"
            "do"
            "  cp $model models/$(basename $(dirname $(dirname $model))).$(basename $(dirname $model)).onnx;"
            "done"
        )

    results = {
        "transform": [],
        "id": [],
        "best_epoch": [],
        "true_error": [],
        "relative_error": [],
        "num_neurons": [],
        "num_parameters": [],
        "distillation_time": [],
    }
    for model_path in Path("models").iterdir():
        identifier = model_path.stem.split(".")[-1]
        transform = model_path.stem[: -len(identifier) - 1]
        with open(f"{identifier}.out") as distillation_log:
            cmd = distillation_log.readline()
            date_str = distillation_log.readline().strip()
            start_datetime = datetime.strptime(date_str, "%a %b %d %H:%M:%S %Z %Y")
            best_epoch = -1
            best_epoch_true_err = float("inf")
            best_epoch_relative_err = float("inf")
            current_epoch = 0
            num_neurons = -1
            num_parameters = -1
            improved = False
            for line in distillation_log:
                if "number of neurons" in line:
                    num_neurons = int(line.split()[-2].split("=")[-1])
                elif "number of parameters" in line:
                    num_parameters = int(line.split()[-2].split("=")[-1])
                elif "validation error" in line:
                    current_epoch += 1
                    true_err_str, _, relative_err_str = line.split()[-3:]
                    true_err = float(true_err_str.split("=")[-1][:-1])
                    relative_err = float(relative_err_str.split("=")[-1][:-1])
                    if relative_err < best_epoch_relative_err:
                        best_epoch = current_epoch
                        best_epoch_true_err = true_err
                        best_epoch_relative_err = relative_err
            end_datetime = datetime.strptime(
                " ".join(line.split()[1:3]).split(",")[0], "%Y-%m-%d %H:%M:%S"
            )
        results["transform"].append(transform)
        results["id"].append(identifier)
        results["best_epoch"].append(best_epoch)
        results["true_error"].append(best_epoch_true_err)
        results["relative_error"].append(best_epoch_relative_err)
        results["num_neurons"].append(num_neurons)
        results["num_parameters"].append(num_parameters)
        results["distillation_time"].append(
            (end_datetime - start_datetime).total_seconds()
        )

    os.chdir(original_dir)
    df = pd.DataFrame(results)
    print("saving csv:", f"{csv_name}.csv")
    df.to_csv(f"{csv_name}.csv", index=False)


if __name__ == "__main__":
    main()
