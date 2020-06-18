import argparse
import numpy as np
import toml
import torch
import torch.nn as nn

from pathlib import Path

from r4v.distillation.config import DataConfiguration
from r4v.distillation.data import get_data_loader
from r4v.nn import load_network
from r4v.nn.pytorch import Flatten, Relu, Transpose


def Linear_forward(self, X):
    x, x_err, x_lb, x_ub = X
    y = torch.matmul(self.weight.data, x)
    y[:, :, -1] += self.bias.data

    y_err = torch.matmul(self.weight.data, x_err)
    return y, y_err, x_lb, x_ub


def Flatten_forward(self, X):
    x, x_err, x_lb, x_ub = X
    assert self.axis == 1 and x.shape[0] == 1
    size = np.product(x_err.shape[1:-1])
    new_shape = (x.shape[0], size, x.shape[-1])
    y = x.reshape(*new_shape)
    y_err = x_err.reshape(x_err.shape[0], size, x_err.shape[-1])
    return y, y_err, x_lb, x_ub


def Relu_forward(self, X):
    x, x_err, x_lb, x_ub = X
    x_pos = x.clamp(0, None)
    x_neg = x.clamp(None, 0)

    lb = ((x_pos @ x_lb) + (x_neg @ x_ub)).flatten() + x_err.clamp(None, 0).sum(
        -1
    ).flatten()
    ub = ((x_pos @ x_ub) + (x_neg @ x_lb)).flatten() + x_err.clamp(0, None).sum(
        -1
    ).flatten()
    contains_zero = (lb < 0) & (ub > 0)

    y_err = x_err.clone()
    y_err[:, ub <= 0, :] = 0
    y_err[:, contains_zero] = (
        y_err[:, contains_zero]
        * (ub[contains_zero] / (ub[contains_zero] - lb[contains_zero]))[:, None]
    )
    new_err = torch.zeros(x_err.shape[0], len(ub), contains_zero.sum().item())
    new_err[:, contains_zero] = torch.diag_embed(
        (
            -lb[contains_zero]
            * ub[contains_zero]
            / (ub[contains_zero] - lb[contains_zero])
        )
    )
    y_err = torch.cat([y_err, new_err], dim=-1)

    y = x.clone()
    assert ((ub <= 0) & contains_zero).sum() == 0
    y[:, ub <= 0] = 0
    y[:, contains_zero] = (
        y[:, contains_zero]
        * (ub[contains_zero] / (ub[contains_zero] - lb[contains_zero]))[:, None]
    )

    # print("PRE-RELU")
    # sign_bias = abs(lb) > abs(ub)
    # sign_weight = torch.ones_like(lb)
    # sign_weight[sign_bias] = lb[sign_bias] / ub[sign_bias]
    # sign_weight[~sign_bias] = ub[~sign_bias] / -lb[~sign_bias]
    # L = lb[torch.sign(lb) != torch.sign(ub)]
    # U = ub[torch.sign(lb) != torch.sign(ub)]
    # print(L)
    # print(U)
    # # print(sign_weight[torch.sign(lb) != torch.sign(ub)])
    # print(
    #     [
    #         f"{i}:{(abs(sign_weight[torch.sign(lb) != torch.sign(ub)]) > i).sum().item()}"
    #         for i in range(21)
    #     ]
    # )
    # D = torch.min(
    #     abs(lb[torch.sign(lb) == torch.sign(ub)]),
    #     abs(ub[torch.sign(lb) == torch.sign(ub)]),
    # ) * torch.sign(lb[torch.sign(lb) == torch.sign(ub)])
    # print(D)
    # print([f"{i}:{(abs(D) > i).sum().item()}" for i in range(21)])
    # print()

    return y, y_err, x_lb, x_ub


def Transpose_forward(self, X):
    x, x_err, x_lb, x_ub = X
    y = x.permute(*(self.dims + (len(self.dims),)))
    y_err = x_err.permute(*(self.dims + (len(self.dims),)))
    return y, y_err, x_lb, x_ub


def compute_output_intervals(model, x, eps, x_min=0, x_max=1):
    old_Linear_forward = nn.Linear.forward
    old_Flatten_forward = Flatten.forward
    old_Relu_forward = Relu.forward
    old_Transpose_forward = Transpose.forward

    nn.Linear.forward = Linear_forward
    Flatten.forward = Flatten_forward
    Relu.forward = Relu_forward
    Transpose.forward = Transpose_forward

    input_shape = x.shape
    input_size = np.product(input_shape)
    I = torch.eye(input_size, input_size + 1).reshape(input_shape + (input_size + 1,))
    err = torch.zeros(input_shape + (0,))

    lb = (x - eps / 255.0).reshape(input_size, 1).clamp(x_min, x_max)
    ub = (x + eps / 255.0).reshape(input_size, 1).clamp(x_min, x_max)
    lb = torch.cat([lb, torch.ones(1, 1)])
    ub = torch.cat([ub, torch.ones(1, 1)])

    Y = model((I, err, lb, ub))
    y, y_err, y_lb, y_ub = Y

    y_pos = y.clamp(0, None)
    y_neg = y.clamp(None, 0)
    y_lb = (y_pos @ lb + y_neg @ ub).flatten() + y_err.clamp(None, 0).sum(-1).flatten()
    y_ub = (y_pos @ ub + y_neg @ lb).flatten() + y_err.clamp(0, None).sum(-1).flatten()

    nn.Linear.forward = old_Linear_forward
    Flatten.forward = old_Flatten_forward
    Relu.forward = old_Relu_forward
    Transpose.forward = old_Transpose_forward

    return y_lb, y_ub


def compute_active_ranges(model, data_loader):
    layer = {}

    old_Relu_forward = Relu.forward

    def Relu_forward(self, x):
        layer_id = Relu.layer_id
        if layer_id not in layer:
            layer[layer_id] = {"lb": torch.min(x, 0)[0], "ub": torch.max(x, 0)[0]}
        else:
            layer[layer_id]["lb"] = torch.min(layer[layer_id]["lb"], torch.min(x, 0)[0])
            layer[layer_id]["ub"] = torch.max(layer[layer_id]["ub"], torch.max(x, 0)[0])
        Relu.layer_id += 1
        return old_Relu_forward(self, x)

    Relu.forward = Relu_forward

    for _, x, _, _ in data_loader:
        Relu.layer_id = 0
        model(x)

    # for layer_id, bounds in layer.items():
    #     print("Layer:", layer_id)
    #     print("LB:", bounds["lb"].detach().numpy().tolist())
    #     print("UB:", bounds["ub"].detach().numpy().tolist())
    #     print()
    return layer


def compute_active_ranges_per_class(model, data_loader):
    bounds = {}
    sign_counts = {}

    old_Relu_forward = Relu.forward

    def Relu_forward(self, x):
        layer_id = Relu.layer_id
        for x_, y_ in zip(x, y):
            if layer_id not in bounds:
                bounds[layer_id] = {}
                sign_counts[layer_id] = {}
            c = y_.item()
            if c not in bounds[layer_id]:
                bounds[layer_id][c] = {"lb": x_, "ub": x_}
                sign_counts[layer_id][c] = {
                    "-": (x_ < 0).long(),
                    "0": (x_ == 0).long(),
                    "+": (x_ > 0).long(),
                }
            else:
                bounds[layer_id][c]["lb"] = torch.min(bounds[layer_id][c]["lb"], x_)
                bounds[layer_id][c]["ub"] = torch.max(bounds[layer_id][c]["ub"], x_)
                sign_counts[layer_id][c]["-"] += x_ < 0
                sign_counts[layer_id][c]["+"] += x_ > 0
                sign_counts[layer_id][c]["0"] += x_ == 0
        Relu.layer_id += 1
        return old_Relu_forward(self, x)

    Relu.forward = Relu_forward

    for _, x, _, y in data_loader:
        Relu.layer_id = 0
        model(x)

    for layer_id, layer_bounds in bounds.items():
        print("Layer:", layer_id)
        for i in range(10):
            print(f"LB[{i}]:", layer_bounds[i]["lb"].detach().numpy().tolist())
            print(f"UB[{i}]:", layer_bounds[i]["ub"].detach().numpy().tolist())
        print()

    for layer_id, layer_sign_counts in sign_counts.items():
        print("Layer:", layer_id)
        for i in range(10):
            print(
                f"#-[{i}]:",
                ", ".join(
                    [f"{c:6d}" for c in layer_sign_counts[i]["-"].detach().numpy()]
                ),
            )
            print(
                f"#+[{i}]:",
                ", ".join(
                    [f"{c:6d}" for c in layer_sign_counts[i]["+"].detach().numpy()]
                ),
            )
        print()
    return bounds


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("network", type=Path)
    parser.add_argument("data_config", type=str, help="path to the dataset")
    parser.add_argument("input", type=str)
    parser.add_argument("-e", "--epsilon", type=float, default=10)
    return parser.parse_args()


def main(args):
    config = DataConfiguration(toml.load(args.data_config))
    config.config["_STAGE"] = "train"
    data_loader = get_data_loader(config)

    dnn = load_network({"model": args.network})
    print(dnn.layers)
    # dnn.layers = dnn.layers[:40]
    model = dnn.as_pytorch(maintain_weights=True)
    print(model)
    print()

    x = torch.from_numpy(np.load(args.input)[None]).float()
    y = model(x)
    print(y)
    print()

    # _ = compute_active_ranges(model, data_loader)

    _ = compute_active_ranges_per_class(model, data_loader)

    y_lb, y_ub = compute_output_intervals(model, x, args.epsilon)
    print(y_lb)
    print(y_ub)

    glb = y >= y_lb
    lub = y <= y_ub
    assert glb.all()
    assert lub.all()


if __name__ == "__main__":
    main(_parse_args())
