import numpy as np
import torch
import torch.nn as nn


class Data:
    def __init__(self, initializations, path, model_type):
        self.initializations = initializations
        X_list = []
        Y_list = []
        for init_type in self.initializations:
            data = np.load(path + "/" + init_type + ".npz")
            X_list.append(data["input"].astype(np.float32))
            Y_list.append(data["output"].astype(np.float32))
        self.X_numpy = np.concatenate(X_list, axis=0)
        self.Y_numpy = np.concatenate(Y_list, axis=0)
        self.Nx = self.X_numpy.shape[1]
        if model_type == "CNN":
            self.X_numpy = self.X_numpy[:, np.newaxis, :]
            self.Y_numpy = self.Y_numpy[:, np.newaxis, :]
        self.X = torch.from_numpy(self.X_numpy)
        self.Y = torch.from_numpy(self.Y_numpy)


def weights_init(layer_in, nonlinearity):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_normal_(
            layer_in.weight,
            nonlinearity=nonlinearity
        )
        layer_in.bias.data.fill_(0.0)


def compute_l2_error(u, u_pred):
    diff = u_pred - u
    return np.linalg.norm(diff) / (np.linalg.norm(u) + 1e-10)
