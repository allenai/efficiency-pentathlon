import time
from collections import defaultdict
from random import Random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from efficiency_benchmark.efficiency.profiler import (NUM_LATENCY_INSTANCES,
                                                      Profiler)
from efficiency_benchmark.stdio_wrapper import StdioWrapper
from efficiency_benchmark.tango_utils import MappedSequence
from efficiency_benchmark.task import Task
from efficiency_benchmark.tasks import TASKS, InstanceFormat


class PredictStep():
    def __init__(
        self,
        *,
        cmd: List[str],
        task: Union[str, Task],
        split: Optional[str] = None,
        limit: Optional[int] = None,
        random_subsample_seed: Optional[int] = None,
        **kwargs
    ):
        self.task = TASKS[task] if isinstance(task, str) else task
        self.split = split if split is not None else self.task.default_split
        self.limit = limit
        self.random_subsample_seed = random_subsample_seed
        self.cmd = cmd
        self.predictor = StdioWrapper(cmd=cmd)
        self._eval_inputs, self._targets = self._get_instances()
        num_latency_instances = min(NUM_LATENCY_INSTANCES, len(self._eval_inputs))
        self._latency_inputs = Random(random_subsample_seed).sample(
                self._eval_inputs, num_latency_instances)
        self._profiler = Profiler(interval=0.1)

    def _get_instances(self) -> Tuple[Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]]:
        instances = self.task.get_split(self.split)
        # TODO
        instances = self._convert_instances(
            instances, InstanceFormat.EFFICIENCY_BENCHMARK, self.task)

        random_subsample_seed = 0 if self.random_subsample_seed is None else self.random_subsample_seed
        if self.limit is not None and len(instances) > self.limit:
            instances = instances[:self.limit] if random_subsample_seed is None else Random(random_subsample_seed).sample(instances, self.limit)

        return [i.input for i in instances], \
               [i.target for i in instances]

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
    ):
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
        print(f"Latency: {efficiency_metrics['latency'] * 1000: .2f} ms.")

    def run(
        self,
        **kwargs
    ) -> Sequence[Any]:
        output_batches = []
        self.predictor.start(dummy_inputs=self._eval_inputs[:1])

        self._profiler.start()
        for output_batch in self.predictor.predict(instances=self._eval_inputs, **kwargs):
            output_batches.append(output_batch)
        efficiency_metrics = self._profiler.stop()
        efficiency_metrics["throughput"] = len(self._latency_inputs) / efficiency_metrics["time"]

        ### Latency ###
        start_time = time.time()
        for output_batch in self.predictor.predict(instances=self._latency_inputs, batch_size=1):
            pass
        self.predictor.stop()
        elapsed_time = time.time() - start_time
        efficiency_metrics["latency"] = elapsed_time / len(self._latency_inputs)
        self.tabulate_efficiency_metrics(efficiency_metrics)
        results = self.process(output_batches)
        return results

    def process(
        self,
        output_batches: Iterable[str]
    ) -> Iterable[Dict[str, Any]]:
        yielded_label_index = -1
        for output in output_batches:
            yielded_label_index += 1
            output = output.rstrip()
            target = self._targets[yielded_label_index]
            yield {
                "target": target,
                "output": output,
                "bleu": (output, target),   # TODO
            }


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
