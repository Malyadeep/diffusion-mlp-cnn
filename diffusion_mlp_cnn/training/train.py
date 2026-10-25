from diffusion_mlp_cnn.models.MLP import MLP
from diffusion_mlp_cnn.models.CNN import CNN


def train(
    data: object,
    config: dict
):
    if "model_type" not in config:
        raise ValueError("model_type is not present in config")
    model_type = config["model_type"]
    if model_type not in ("MLP", "CNN"):
        raise ValueError(
            model_type + " not supported, choose between: (MLP, CNN)"
        )
    if model_type == "MLP":
        model = MLP(
            data,
            config
        )
    elif model_type == "CNN":
        model = CNN(
            data,
            config
        )
    model.init_model()
    model.train()
