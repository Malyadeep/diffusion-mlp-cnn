import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from diffusion_mlp_cnn.models.utils import weights_init, compute_l2_error


class CNNDiffusionRegressor(nn.Module):
    def __init__(self, Nx):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 1, 3, padding=1, padding_mode="circular")
        )

    def forward(self, u):
        return self.network(u)


class CNN:
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
            self.save_path = config["save_path"]
            self.activation = config["activation"]
        except KeyError as e:
            print("Missing key in config dict: " + str(e), flush=True)
        self.data = data

    def init_model(self):
        print(80 * "-")
        print("Initialize model...\n", flush=True)
        print(f"Network width: {self.width}", flush=True)
        print(f"Learning rate: {self.learning_rate}", flush=True)
        print(f"Batch size: {self.batch_size}", flush=True)
        self.model = CNNDiffusionRegressor(
            self.data.training.Nx
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
        print(f"Total no. of parameters: {self.n_params}", flush=True)
        print("\nInitializing model done!", flush=True)
        print(80 * "-")

    def train(self):
        print(80 * "-", flush=True)
        print("Training begins...\n", flush=True)
        print(f"Epochs: {self.epochs} {'\n'}", flush=True)
        best_val_error = float("inf")
        perf_file_dir = self.save_path + "training_performance/"
        models_dir = self.save_path + "trained_models/"
        os.makedirs(perf_file_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        perf_file = open(perf_file_dir + self.name + ".txt", "w")
        perf_file.write(
            f"{'# epoch':<10} "
            f"{'training_MSE_loss':<20} "
            f"{'training_L2_error':<20} "
            f"{'validation_L2_error':<20}\n"
        )
        perf_file.close()
        start_time = time.perf_counter()
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
                    models_dir + self.name + ".pt"
                )
            if epoch % self.print_interval == 0:
                print(
                    f"Epoch: {epoch:5d} | "
                    f"Training MSE Loss: {epoch_loss:.5f} | "
                    f"Training L2 Error: {l2_error_training:.5f} | "
                    f"Validation L2 Error: {l2_error_validation:.5f} | ",
                    flush=True
                )
                perf_file = open(perf_file_dir + self.name + ".txt", "a")
                perf_file.write(
                    f"{epoch:<10d} "
                    f"{epoch_loss:<10.5f} "
                    f"{l2_error_training:<10.5f} "
                    f"{l2_error_validation:<10.5f}\n"
                )
                perf_file.close()
        run_time = time.perf_counter() - start_time
        print("\nTraining complete! Run time = " + str(run_time) + " s",
              flush=True)
        print(80 * "-", flush=True)
