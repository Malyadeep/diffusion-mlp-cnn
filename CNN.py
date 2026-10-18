import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser


class Data:
    def __init__(self, initializations, path):
        self.initializations = initializations
        X_list = []
        Y_list = []
        for init_type in self.initializations:
            data = np.load(path + "/" + init_type + ".npz")
            X_list.append(data["input"].astype(np.float32))
            Y_list.append(data["output"].astype(np.float32))
        self.X_numpy = np.concatenate(X_list, axis=0)
        self.Y_numpy = np.concatenate(Y_list, axis=0)
        self.X_numpy = self.X_numpy[:, np.newaxis, :]
        self.Y_numpy = self.Y_numpy[:, np.newaxis, :]
        self.Nx = self.X_numpy.shape[1]
        self.X = torch.from_numpy(self.X_numpy)
        self.Y = torch.from_numpy(self.Y_numpy)


class CNNDiffusionRegressor(nn.Module):
    def __init__(self, Nx):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 1, 3, padding=1, padding_mode="circular")
        )

    def forward(self, u):
        return self.network(u)


def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_normal_(
            layer_in.weight,
            nonlinearity="relu"
        )
        layer_in.bias.data.fill_(0.0)


def compute_l2_error(u, u_pred):
    diff = u_pred - u
    return np.linalg.norm(diff) / (np.linalg.norm(u) + 1e-10)


class CNN:
    def __init__(
        self,
        config: dict,
        name="cnn_state",
        activation="relu"
    ):
        try:
            self.epochs = config["epochs"]
            self.learning_rate = config["learning_rate"]
            self.width = config["network_width"]
            self.print_interval = config["print_interval"]
            self.batch_size = config["batch_size"]
        except KeyError as e:
            print("Missing key in config dict: " + str(e))
        self.name = name
        self.activation = activation

    def load_data(self, initializations: list):
        print(80 * "-")
        print("Loading training data...\n")
        self.training_data = Data(
            initializations,
            path="training_data"
        )
        print(f"Input training data shape (N_samples, Nx): "
              f"{self.training_data.X_numpy.shape}")
        print(f"Output training data shape (N_samples, Nx): "
              f"{self.training_data.Y_numpy.shape}")
        print(f"Training data dtype (input, output): "
              f"({self.training_data.X.dtype}, {self.training_data.Y.dtype})")
        print("\nLoading training data done!")
        print(80 * "-")

        print(80 * "-")
        print("Loading single-step validation data...\n")
        self.validation_data = Data(
            initializations,
            path="validation_data"
        )
        print(f"Input single-step validation data shape (N_samples, Nx): "
              f"{self.validation_data.X_numpy.shape}")
        print(f"Output single-step validation data shape (N_samples, Nx): "
              f"{self.validation_data.Y_numpy.shape}")
        print(f"Single-step validation data dtype (input, output): "
              f"({self.validation_data.X.dtype}, "
              f"{self.validation_data.Y.dtype})")
        print("\nLoading single-step validation data done!")
        print(80 * "-")

    def init_model(self):
        print(80 * "-")
        print("Initialize model...\n")
        print(f"Network width: {self.width}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        self.model = CNNDiffusionRegressor(
            self.training_data.Nx
        )
        self.model.apply(weights_init)
        self.opt = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        self.criterion = nn.MSELoss()
        self.data_loader = DataLoader(
            TensorDataset(self.training_data.X, self.training_data.Y),
            batch_size=self.batch_size,
            shuffle=True
        )
        self.n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total no. of parameters: {self.n_params}")
        print("\nInitializing model done!")
        print(80 * "-")

    def train(self):
        print(80 * "-")
        print("Training begins...\n")
        print(f"Epochs: {self.epochs} {'\n'}")
        best_val_error = float("inf")
        perf_file = open(
            self.name + "_training_perf.txt",
            "w"
        )
        perf_file.write(
            f"{'# epoch':<10} "
            f"{'training_MSE_loss':<20} "
            f"{'training_L2_error':<20} "
            f"{'validation_L2_error':<20}\n"
        )
        perf_file.close()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_itr, batch_data in enumerate(self.data_loader):
                X_batch, Y_batch = batch_data
                self.opt.zero_grad()
                pred = self.model.forward(X_batch)
                loss = self.criterion(pred, Y_batch)
                loss.backward()
                self.opt.step()
                epoch_loss += loss.item()
            epoch_loss /= len(self.data_loader)
            Y_pred_training = self.model(
                self.training_data.X
            ).detach().numpy()
            l2_error_training = compute_l2_error(
                self.training_data.Y_numpy,
                Y_pred_training
            )
            Y_pred_validation = self.model(
                self.validation_data.X
            ).detach().numpy()
            l2_error_validation = compute_l2_error(
                self.validation_data.Y_numpy,
                Y_pred_validation
            )
            if l2_error_validation < best_val_error:
                best_val_error = l2_error_validation
                torch.save(
                    self.model.state_dict(),
                    self.name + ".pt"
                )
            if epoch % self.print_interval == 0:
                print(
                    f"Epoch: {epoch:5d} | "
                    f"Training MSE Loss: {epoch_loss:.5f} | "
                    f"Training L2 Error: {l2_error_training:.5f} | "
                    f"Validation L2 Error: {l2_error_validation:.5f} | "
                )
                perf_file = open(
                    self.name + "_training_perf.txt",
                    "a"
                )
                perf_file.write(
                    f"{epoch:<10d} "
                    f"{epoch_loss:<10.5f} "
                    f"{l2_error_training:<10.5f} "
                    f"{l2_error_validation:<10.5f}\n"
                )
                perf_file.close()
        print("\nTraining complete!")
        print(80 * "-")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="A Convolutional Neural Network for diffusion operator"
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="CNN name to be used to save"
    )
    parser.add_argument(
        "-a",
        "--activation",
        choices=["relu", "tanh"],
        help="Type of activation between linear layers"
    )
    args = parser.parse_args()
    config = {
        "epochs": 2001,
        "learning_rate": 1e-3,
        "print_interval": 100,
        "network_width": 128,
        "batch_size": 100
    }
    initializations = ["gaussian", "uniform", "cosine"]
    cnn = CNN(
        config,
        name=args.name
    )
    cnn.load_data(initializations)
    cnn.init_model()
    cnn.train()
