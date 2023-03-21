import inspect
from abc import ABC
from typing import Sequence, Dict, Any, Iterator, Tuple, List, Union

import torch
from tango.common import Registrable, Tqdm
from tango.common.det_hash import DetHashWithVersion

from catwalk.task import Task


Instance = Dict[str, Any]


def tensor_args(args: Tuple[Any]) -> Tuple[Any, ...]:
    """
    Annoyingly, torchmetrics only supports tensors as input, not raw values. So we have to convert raw values
    into tensors.
    """
    fixed_args: List[Any] = []
    for arg in args:
        if isinstance(arg, (float, int)):
            fixed_args.append(torch.tensor(arg))
        else:
            fixed_args.append(arg)
    return tuple(fixed_args)


def unsqueeze_args(args: Tuple[Any]) -> Tuple[Any, ...]:
    """
    Further, torchmetrics can't handle single-instance calls when given tensors. It always needs the first
    dimension of the tensors to be the instance dimension. So we add one.
    """
    fixed_args: List[Any] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            fixed_args.append(arg.unsqueeze(0))
        else:
            fixed_args.append(arg)
    return tuple(fixed_args)


_TorchmetricsResult = Union[torch.Tensor, Dict[str, '_TorchmetricsResult']]
_CatwalkResult = Union[float, Dict[str, '_CatwalkResult']]


def recursive_tolist(args: _TorchmetricsResult) -> _CatwalkResult:
    if isinstance(args, dict):
        return { key: recursive_tolist(value) for key, value in args.items() }
    else:
        return args.tolist()


class Model(Registrable, DetHashWithVersion, ABC):
    VERSION = "002lst"

    def prepare(self, task: Task, instances: Sequence[Dict[str, Any]], **kwargs):
        raise NotImplementedError()

    def predict(self, task: Task, instances: Sequence[Dict[str, Any]], **kwargs) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError()

    def calculate_metrics(self, task: Task, predictions: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        metrics = task.make_metrics()
        with Tqdm.tqdm(predictions, desc="Calculating metrics") as predictions_tqdm:
            for prediction in predictions_tqdm:
                for metric_name, metric_args in prediction.items():
                    try:
                        metric = metrics[metric_name]
                    except KeyError:
                        continue
                    metric_args = tensor_args(metric_args)
                    metric_args = unsqueeze_args(metric_args)
                    metric.update(*metric_args)
        return {
            metric_name: recursive_tolist(metric.compute())
            for metric_name, metric in metrics.items()
        }

    @property
    def supports_fewshot(self) -> bool:
        return "num_shots" in inspect.signature(self.predict).parameters


class UnsupportedTaskError(Exception):
    """Thrown when the model doesn't support the task it's given."""

    def __init__(self, model: Model, task: Task):
        super().__init__(f"Model {model} does not support task {task}.")
        self.model = model
        self.task = task
