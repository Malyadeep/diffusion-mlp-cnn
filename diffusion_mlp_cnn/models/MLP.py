import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from diffusion_mlp_cnn.models.utils import Data, weights_init, compute_l2_error


class MLPDiffusionRegressor(nn.Module):
    def __init__(self, Nx, width, activation="relu"):
        super().__init__()
        if activation == "relu":
            activation_layer = nn.ReLU()
        elif activation == "tanh":
            activation_layer = nn.Tanh()
        self.network = nn.Sequential(
            nn.Linear(Nx, width),
            activation_layer,
            nn.Linear(width, width),
            activation_layer,
            nn.Linear(width, Nx)
        )

    def forward(self, u):
        return self.network(u)


class MLP:
    def __init__(
        self,
        data,
        config
    ):
        try:
            self.epochs = config["epochs"]
            self.learning_rate = config["learning_rate"]
            self.width = config["network_width"]
            self.print_interval = config["print_interval"]
            self.batch_size = config["batch_size"]
            self.name = config["name"]
            self.activation = config["activation"]
        except KeyError as e:
            print("Missing key in config dict: " + str(e))
        self.data = data

    def init_model(self):
        print(80 * "-")
        print("Initialize model...\n")
        print(f"Network width: {self.width}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        self.model = MLPDiffusionRegressor(
            self.data.training.Nx,
            self.width,
            activation=self.activation
        )
        self.model.apply(lambda layer: weights_init(layer, self.activation))
        self.opt = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        self.criterion = nn.MSELoss()
        self.data_loader = DataLoader(
            TensorDataset(self.data.training.X, self.data.training.Y),
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
                self.data.training.X
            ).detach().numpy()
            l2_error_training = compute_l2_error(
                self.data.training.Y_numpy,
                Y_pred_training
            )
            Y_pred_validation = self.model(
                self.data.validation.X
            ).detach().numpy()
            l2_error_validation = compute_l2_error(
                self.data.validation.Y_numpy,
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
