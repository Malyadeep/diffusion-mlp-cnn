"""
Generates training, validation and test dataset for
1D unsteady diffusion equation using various initial conditions
"""
import numpy as np
from itertools import product
from types import SimpleNamespace

from diffusion_mlp_cnn.fdm.solver import State


def create_params(params_dict):
    params_list = []
    for item in params_dict:
        range_data = params_dict[item]
        params_list.append(
            np.linspace(range_data[0], range_data[1], range_data[2])
        )
    params = np.array(list(product(*params_list)))
    return params


def generate_data(
    func,
    params,
    D,
    L,
    iterations,
    training_range,
    validation_range
):
    if not isinstance(training_range, tuple):
        raise ValueError(
            "Training range must be a tuple containing"
            " initial and final step for training data: (t_i, t_f)"
        )
    if training_range[0] >= iterations or training_range[1] >= iterations:
        raise ValueError(
            "Initial and final step for training range "
            "must be less than total number of iterations"
        )
    if not isinstance(validation_range, tuple):
        raise ValueError(
            "Validation range must be a tuple containing"
            " initial and final step for training data: (t_i, t_f)"
        )
    if validation_range[0] >= iterations or validation_range[1] >= iterations:
        raise ValueError(
            "Initial and final step for validation range "
            "must be less than total number of iterations"
        )
    state = State(L)
    training_snaps = training_range[1] - training_range[0] + 1
    validation_snaps = validation_range[1] - validation_range[0] + 1
    training_data = SimpleNamespace(
        input=np.zeros(
            (training_snaps * params.shape[0], state.Nx),
            dtype=np.float32
        ),
        output=np.zeros(
            (training_snaps * params.shape[0], state.Nx),
            dtype=np.float32
        )
    )
    validation_data = SimpleNamespace(
        input=np.zeros(
            (validation_snaps * params.shape[0], state.Nx),
            dtype=np.float32
        ),
        output=np.zeros(
            (validation_snaps * params.shape[0], state.Nx),
            dtype=np.float32
        )
    )
    for itr, param in enumerate(params):
        state.initialize(func, param)
        state.evolve(iterations, D, verbose=False)
        training_data.input[
            itr * training_snaps:(itr + 1) * training_snaps, :
        ] = np.copy(
                state.u[training_range[0]:training_range[1] + 1, :]
            )
        training_data.output[
            itr * training_snaps:(itr + 1) * training_snaps, :
        ] = np.copy(
                state.u[training_range[0] + 1:training_range[1] + 2, :]
            )
        validation_data.input[
            itr * validation_snaps:(itr + 1) * validation_snaps, :
        ] = np.copy(
                state.u[validation_range[0]:validation_range[1] + 1, :]
            )
        validation_data.output[
            itr * validation_snaps:(itr + 1) * validation_snaps, :
        ] = np.copy(
                state.u[validation_range[0] + 1:validation_range[1] + 2, :]
            )
    return training_data, validation_data
