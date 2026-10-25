from types import SimpleNamespace
from diffusion_mlp_cnn.models.utils import Data
from diffusion_mlp_cnn.training.train import train


def load_data(initializations: list, model_type):
    print(80 * "-")
    print("Loading training data...\n")
    training_data = Data(
        initializations,
        path="training_data",
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
        path="validation_data",
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


config = {
    "model_type": "MLP",
    "name": "mlp_tanh_512",
    "epochs": 2001,
    "learning_rate": 1e-3,
    "print_interval": 100,
    "network_width": 512,
    "batch_size": 100,
    "activation": "tanh"
}


def main():
    initializations = ["gaussian", "uniform", "cosine"]
    data = load_data(initializations, config["model_type"])
    train(data, config)


if __name__ == "__main__":
    main()
