import time
from collections import defaultdict
from random import Random
import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import numpy as np
from efficiency_benchmark.efficiency.profiler import Profiler
from efficiency_benchmark.stdio_wrapper import StdioWrapper
from efficiency_benchmark.tango_utils import MappedSequence
from efficiency_benchmark.task import Task
from efficiency_benchmark.tasks import TASKS, InstanceFormat
import more_itertools
import csv


EXPECTED_BATCH_SIZE = 32
NUM_BATCHES = 1000


class PredictStep():
    def __init__(
        self,
        *,
        cmd: List[str],
        task: Union[str, Task],
        scenario: str,
        split: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs
    ):
        np.random.seed(42)
        self.task = TASKS[task] if isinstance(task, str) else task
        self.split = split if split is not None else self.task.default_split
        self.scenario = scenario
        self.limit = limit
        self.cmd = cmd
        self.predictor = StdioWrapper(cmd=cmd)
        self._profiler = Profiler(interval=0.1)
        self._get_batches()

    def _get_batches(self) -> List[List[str]]:
        instances = self.task.get_split(self.split)
        instances = self._convert_instances(
            instances, InstanceFormat.EFFICIENCY_BENCHMARK, self.task)
        instances = list(instances)
        if self.scenario == "single_stream":
            batches = list(more_itertools.chunked(instances, 1))
        elif self.scenario == "random_batch":
            num_instances_per_batch = np.random.poisson(
                lam=EXPECTED_BATCH_SIZE, size=NUM_BATCHES)
            batches = []
            for n in num_instances_per_batch:
                if n == 0:
                    continue
                batch = np.random.choice(instances, size=n, replace=False).tolist()
                batches.append(batch)
        elif self.scenario == "offline":
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown scenario: {self._scenario}. Choose from 'single_stream', 'random_batch', 'offline'")

        if self.limit is not None and len(batches) > self.limit:
            indices = np.random.choice(
                list(range(len(batches))), 
                size=self.limit, 
                replace=False
            )
            batches = [ batches[i] for i in indices]
        self._num_batches = len(batches)
        self._num_instances = sum(len(batch) for batch in batches)
        self._input_batches = [ [instance.input for instance in batch] for batch in batches]
        target_batches = [ [instance.target for instance in batch] for batch in batches]
        self._targets = list(itertools.chain(*target_batches))
        assert len(self._targets) == self._num_instances

    @classmethod
    def _convert_instances(
        self,
        instances: Sequence[Dict[str, Any]],
        instance_format,
        task
    ) -> MappedSequence:
        return MappedSequence(task.instance_conversions[instance_format], instances)

    def massage_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(kwargs["task"], str):
            kwargs["task"] = TASKS[kwargs["task"]]
        if kwargs["split"] is None:
            kwargs["split"] = kwargs["task"].default_split
        return kwargs

    def tabulate_efficiency_metrics(
        self,
        efficiency_metrics: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        print(f"Time Elapsed: {efficiency_metrics['time']:.2f} s")
        # print(f"Max DRAM Memory Usage: {max_mem_util * total_memory: .2f} GiB")
        # print(f"Number of Parameters: {efficiency_metrics['num_params'] / 1e6: .2f} M")
        print(f"Max GPU Memory Usage: {efficiency_metrics['max_gpu_mem']: .2f} GiB")
        # print(f"GPU Energy: {efficiency_metrics['gpu_energy']:.2e} Wh")
        # print(f"CPU Energy: {efficiency_metrics['cpu_energy']: .2e} Wh")
        # print(f"Memory Energy: {efficiency_metrics['dram_energy']: .2e} Wh")
        print(f"Total Energy: {efficiency_metrics['total_energy']: .2e} Wh")
        print(f"CO2 emission: {efficiency_metrics['carbon']: .2e} grams.")
        print(f"Throughput: {efficiency_metrics['throughput']: .2f} instances / s.")
        print(f"Throughput: {efficiency_metrics['throughput_words']: .2f} words / s.")
        print(f"Latency: {efficiency_metrics['latency'] * 1000: .2f} ms / batch.")

    def run(self) -> Sequence[Any]:
        output_batches = []
        self.predictor.start(dummy_inputs=self._input_batches[-1])

        self._profiler.start()
        for output_batch in self.predictor.predict(batches=self._input_batches):
            output_batches.append(output_batch)
        efficiency_metrics = self._profiler.stop()
        efficiency_metrics["throughput"] = self._num_instances / efficiency_metrics["time"]
        efficiency_metrics["latency"] = efficiency_metrics["time"] / self._num_batches
        results, num_output_words = self.process(output_batches)
        efficiency_metrics["throughput_words"] = num_output_words / efficiency_metrics["time"]
        # self.tabulate_efficiency_metrics(efficiency_metrics)
        return results, efficiency_metrics

    def process(
        self,
        output_batches: Iterable[str]
    ) -> Tuple[Sequence[Dict[str, Any]], int]:
        yielded_label_index = -1
        results = []
        num_output_words = 0
        for output in output_batches:
            yielded_label_index += 1
            output = output.rstrip()
            target = self._targets[yielded_label_index]

            result = {metric_name: (output, target) for metric_name in self.task.metrics.keys()}
            result.update({
                "target": self._targets[yielded_label_index] if self._targets is not None else None,
                "output": output,
            })
            num_output_words += len(output.split())
            results.append(result)
        return results, num_output_words
            

class CalculateMetricsStep():

    _TorchmetricsResult = Union[torch.Tensor, Dict[str, '_TorchmetricsResult']]
    _CatwalkResult = Union[float, Dict[str, '_CatwalkResult']]

    def __init__(self, task: Union[str, Task]):
        self.task = TASKS[task] if isinstance(task, str) else task

    def massage_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(kwargs["task"], str):
            kwargs["task"] = TASKS[kwargs["task"]]
        return kwargs

    def _tensor_args(self, args: Tuple[Any]) -> Tuple[Any, ...]:
        """
        Annoyingly, torchmetrics only supports tensors as input, not raw values. So we have to convert raw values
        into tensors.
        
        From catwalk.
        https://github.com/allenai/catwalk/blob/main/catwalk/model.py
        """
        fixed_args: List[Any] = []
        for arg in args:
            if isinstance(arg, (float, int)):
                fixed_args.append(torch.tensor(arg))
            else:
                fixed_args.append(arg)
        return tuple(fixed_args)

    def _unsqueeze_args(self, args: Tuple[Any]) -> Tuple[Any, ...]:
        """
        Further, torchmetrics can't handle single-instance calls when given tensors. It always needs the first
        dimension of the tensors to be the instance dimension. So we add one.

        From catwalk.
        https://github.com/allenai/catwalk/blob/main/catwalk/model.py
        """
        fixed_args: List[Any] = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                fixed_args.append(arg.unsqueeze(0))
            else:
                fixed_args.append(arg)
        return tuple(fixed_args)


    def _recursive_tolist(self, args: _TorchmetricsResult) -> _CatwalkResult:
        """From catwalk.
        https://github.com/allenai/catwalk/blob/main/catwalk/model.py
        """
        if isinstance(args, dict):
            return { key: self._recursive_tolist(value) for key, value in args.items() }
        else:
            return args.tolist()
        
    def calculate_metrics(self, predictions: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """From catwalk.
        https://github.com/allenai/catwalk/blob/main/catwalk/model.py
        """
        metrics = self.task.make_metrics()
        for prediction in predictions:
            for metric_name, metric_args in prediction.items():
                try:
                    metric = metrics[metric_name]
                except KeyError:
                    continue
                metric_args = self._tensor_args(metric_args)
                metric_args = self._unsqueeze_args(metric_args)
                metric.update(*metric_args)
        return {
            metric_name: self._recursive_tolist(metric.compute())
            for metric_name, metric in metrics.items()
        }


class TabulateMetricsStep():

    def __init__(self):
        pass

    def run(self, metrics: Dict[str, Dict[str, float]], format: str = "text") -> Iterable[str]:
        flattend_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        for task_name, task_metrics in metrics.items():
            for metric_name, metric_value in task_metrics.items():
                # if metric_value is a dict, then it's a nested metric
                if isinstance(metric_value, dict):
                    for nested_metric_name, nested_metric_value in metric_value.items():
                        flattend_metrics[task_name][f"{metric_name}.{nested_metric_name}"] = nested_metric_value.item() if isinstance(nested_metric_value, torch.Tensor) else nested_metric_value
                else:
                    flattend_metrics[task_name][metric_name] = metric_value

        if format == "text":
            for task_name, task_metrics in flattend_metrics.items():
                for metric_name, metric_value in task_metrics.items():
                    yield f"{task_name}\t{metric_name}\t{metric_value}"
        elif format == "latex":
            raise NotImplementedError()
        else:
            raise AttributeError("At the moment, only the 'text' format is supported.")
        

class LogOutputStep():

    def __init__(self, task: Union[str, Task], output_file: Optional[str] = None):
        self.task = TASKS[task] if isinstance(task, str) else task
        self.output_file = output_file

    def run(self, predictions: Sequence[Dict[str, Any]]) -> None:
        predictions = self.remove_metrics(predictions)
        if self.output_file is None:
            # Log to stdout if no output file is specified.
            for prediction in predictions:
                print(prediction)
        else:
            field_names = predictions[0].keys()
            with open(self.output_file, "w") as fout:
                writer = csv.DictWriter(fout, fieldnames=field_names, delimiter="\t")
                writer.writeheader()
                for prediction in predictions:
                    writer.writerow(prediction)

    def remove_metrics(
            self, 
            predictions: Sequence[Dict[str, Any]]
    ) -> Sequence[Dict[str, Any]]:
        # Remove metrics from the output.
        for prediction in predictions:
            for metric_name in self.task.metrics.keys():
                prediction.pop(metric_name)
        return predictions

