import time
from collections import defaultdict
from random import Random
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
from tango.common.sequences import MappedSequence

from catwalk.efficiency.profiler import NUM_LATENCY_INSTANCES, Profiler
from catwalk.model import Model
from catwalk.models import MODELS
from catwalk.task import Task
from catwalk.tasks import TASKS, InstanceFormat


class PredictStep():
    def __init__(
        self,
        *,
        model: Union[str, Model],
        task: Union[str, Task],
        split: Optional[str] = None,
        limit: Optional[int] = None,
        random_subsample_seed: Optional[int] = None,
        **kwargs
    ):
        self.task = TASKS[task] if isinstance(task, str) else task
        self.split = split if split is not None else task.default_split
        self.limit = limit
        self.limit = 64
        self.random_subsample_seed = random_subsample_seed
        self.model = MODELS[model] if isinstance(model, str) else model
        self._eval_inputs, self._targets = self._get_instances()
        num_latency_instances = min(NUM_LATENCY_INSTANCES, len(self._eval_inputs))
        self._latency_inputs = Random(random_subsample_seed).sample(
                self._eval_inputs, num_latency_instances)
        self._profiler = Profiler(
            interval=0.1,
        )

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
        if isinstance(kwargs["model"], str):
            kwargs["model"] = MODELS[kwargs["model"]]
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
        # try:
        self.model.start(dummy_input=self._eval_inputs[:1])

        self._profiler.start()
        for output_batch in self.model.predict(instances=self._eval_inputs, **kwargs):
            output_batches.append(output_batch)
        efficiency_metrics = self._profiler.stop()
        efficiency_metrics["throughput"] = len(self._latency_inputs) / efficiency_metrics["time"]
        # except:
        #     self._profiler.stop()
        #     self.model.stop()

        ### Latency ###
        start_time = time.time()
        for output_batch in self.model.predict(instances=self._latency_inputs, batch_size=1):
            pass
        self.model.stop()
        elapsed_time = time.time() - start_time
        efficiency_metrics["latency"] = elapsed_time / len(self._latency_inputs)
        # TODO
        # efficiency_metrics["num_params"] = self.model.model.num_parameters()
        self.tabulate_efficiency_metrics(efficiency_metrics)
        results = self.process(output_batches)
        return results

    def process(
        self,
        output_batches: Iterable[str]
    ) -> Iterable[Dict[str, Any]]:
        yielded_label_index = -1
        # output_batch = json.loads(output_batch.rstrip())
        for output in output_batches:
            yielded_label_index += 1
            output = output.rstrip()
            target = self._targets[yielded_label_index]
            yield {
                "target": target,
                "output": output,
                "bleu": (output, target),
            }


class CalculateMetricsStep():

    def __init__(
        self,
        *,
        model: Union[str, Model],
        task: Union[str, Task]
    ):
        self.model = MODELS[model] if isinstance(model, str) else model
        self.task = TASKS[task] if isinstance(task, str) else task

    def massage_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(kwargs["model"], str):
            kwargs["model"] = MODELS[kwargs["model"]]
        if isinstance(kwargs["task"], str):
            kwargs["task"] = TASKS[kwargs["task"]]
        return kwargs

    def run(
        self,
        predictions: Sequence[Any]
    ) -> Dict[str, float]:
        return self.model.calculate_metrics(self.task, predictions)


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
