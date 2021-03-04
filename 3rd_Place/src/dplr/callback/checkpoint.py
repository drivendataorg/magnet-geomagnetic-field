from dplr.callback import CallBack
import copy
from collections import OrderedDict
from typing import List, Dict
import torch


def average_model_weights(model_parameters: List[Dict[str, torch.tensor]]):
    new_parameters = OrderedDict()
    n_models = len(model_parameters)
    for param_dict in model_parameters:
        for key, tensor in param_dict.items():
            if key in new_parameters:
                new_parameters[key] += tensor/n_models
            else:
                new_parameters[key] = tensor/n_models
    return new_parameters


class ModelCheckpointCallBack(CallBack):
    def __init__(self):
        self.model_params = {}

    def load_averaged_model(self, epochs: List[int]):
        parameters = [param for e, param in self.model_params.items()
                      if e in epochs]
        parameters_to_load = average_model_weights(parameters)
        self.model.load_state_dict(parameters_to_load)

    def load(self, epoch: int):
        self.model.load_state_dict(self.model_params[epoch])

    def after_epoch(self):
        self.model_params[self.epoch] = copy.deepcopy(self.model.state_dict())
