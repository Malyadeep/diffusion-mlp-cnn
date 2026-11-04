from types import SimpleNamespace
import copy
from argparse import ArgumentParser
from diffusion_mlp_cnn.models.utils import Data
from diffusion_mlp_cnn.training.train import train


def load_data(initializations, model_type, dataset_path):
    print(80 * "-")
    print("Loading training data...\n")
    training_data = Data(
        initializations,
        path=dataset_path + "training_data",
        model_type=model_type
    )
    print(f"Input training data shape (N_samples, Nx): "
          f"{training_data.X_numpy.shape}")
    print(f"Output training data shape (N_samples, Nx): "
          f"{training_data.Y_numpy.shape}")
    print(f"Training data dtype (input, output): "
          f"({training_data.X.dtype}, {training_data.Y.dtype})")
    print("\nLoading training data done!")
    print(80 * "-")

    print(80 * "-")
    print("Loading single-step validation data...\n")
    validation_data = Data(
        initializations,
        path=dataset_path + "validation_data",
        model_type=model_type
    )
    print(f"Input single-step validation data shape (N_samples, Nx): "
          f"{validation_data.X_numpy.shape}")
    print(f"Output single-step validation data shape (N_samples, Nx): "
          f"{validation_data.Y_numpy.shape}")
    print(f"Single-step validation data dtype (input, output): "
          f"({validation_data.X.dtype}, "
          f"{validation_data.Y.dtype})")
    print("\nLoading single-step validation data done!")
    print(80 * "-")
    data = SimpleNamespace(
        training=training_data,
        validation=validation_data
    )
    return data


CONFIG = {
    "model_type": "MLP",
    "name": "network",
    "save_path": "models/",
    "epochs": 2001,
    "learning_rate": 1e-3,
    "print_interval": 100,
    "network_width": 128,
    "network_depth": 2,
    "batch_size": 100,
    "activation": "relu"
}


def train_baseline_mlp_cnn():
    config = copy.deepcopy(CONFIG)
    initializations = ["gaussian", "uniform", "cosine"]
    dataset_path = "datasets/training_size_1925/"
    data = load_data(initializations, config["model_type"], dataset_path)
    config["save_path"] = "baseline_mlp_cnn/"
    # Train MLP with ReLU and Tanh activations - baseline dataset
    config["network_width"] = 128
    config["network_depth"] = 2
    for activation in ["relu", "tanh"]:
        config["activation"] = activation
        config["name"] = "mlp_" + activation + "_baseline"
        train(data, config)
    # Train CNN - baseline dataset
    config["model_type"] = "CNN"
    data = load_data(initializations, config["model_type"], dataset_path)
    config["name"] = "cnn_baseline"
    train(data, config)


def train_mlp_depth_width():
    config = copy.deepcopy(CONFIG)
    initializations = ["gaussian", "uniform", "cosine"]
    dataset_path = "datasets/training_size_1925/"
    data = load_data(initializations, config["model_type"], dataset_path)
    config["save_path"] = "mlp_depth_width/"
    # Train MLP with ReLU and Tanh activations - baseline dataset
    # Vary width and depth of the neural network to see
    # if auto-regressive accuracy improves
    for width in [128, 256, 512]:
        for depth in [2, 4, 8]:
            for activation in ["relu", "tanh"]:
                config["network_width"] = width
                config["network_depth"] = depth
                config["activation"] = activation
                config["name"] = "mlp_" + activation +\
                    "_width_" + str(width) +\
                    "_depth_" + str(depth)
                train(data, config)


def train_mlp_vary_dataset():
    config = copy.deepcopy(CONFIG)
    config["save_path"] = "increased_training_examples/"
    initializations = ["gaussian", "uniform", "cosine"]
    # Train MLP with ReLU and Tanh activations
    # Check if adding more training data improves
    # autoregressive test accuracy
    config["network_width"] = 128
    config["network_depth"] = 2
    for size in [1925, 14700, 48825]:
        dataset_path = "datasets/training_size_" + str(size) + "/"
        data = load_data(initializations, config["model_type"], dataset_path)
        for activation in ["relu", "tanh"]:
            config["activation"] = activation
            config["name"] = "mlp_" + activation + "_size_" + str(size)
            train(data, config)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Training script for MLP and CNN for learning" +
        " diffusion operator"
    )
    parser.add_argument(
        "-opt",
        "--option",
        choices=["all", "baseline", "capacity", "dataset"]
    )
    args = parser.parse_args()
    if args.option == "all":
        train_baseline_mlp_cnn()
        train_mlp_depth_width()
        train_mlp_vary_dataset()
    elif args.option == "baseline":
        train_baseline_mlp_cnn()
    elif args.option == "capacity":
        train_mlp_depth_width()
    elif args.option == "dataset":
        train_mlp_vary_dataset()
