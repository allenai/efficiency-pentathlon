import csv
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from collections import defaultdict
from random import Random
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
from tango.common.sequences import MappedSequence

import torch
from catwalk.model import Model
from catwalk.models import MODELS
from catwalk.task import Task
from catwalk.tasks import TASKS
from catwalk.tasks import InstanceFormat
from codecarbon import EmissionsTracker


EFFICIENCY_DIR = f"{pathlib.Path(__file__).parent.resolve()}/efficiency"  # TODO
NUM_LATENCY_INSTANCES = 100


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
        self.random_subsample_seed = random_subsample_seed
        self.model = MODELS[model] if isinstance(model, str) else model
        self._eval_instances, self._targets = self._get_instances()
        num_latency_instances = min(NUM_LATENCY_INSTANCES, len(self._eval_instances))
        self._latency_instances = Random(random_subsample_seed).sample(
                self._eval_instances, num_latency_instances)
        self._emission_tracker = EmissionsTracker(
            measure_power_secs=0.1,
            log_level="warning"
        )

    def _get_instances(self) -> Tuple[Sequence[Dict[str, Any]], Sequence[str]]:
        instances = self.task.get_split(self.split)
        # TODO
        instances = self._convert_instances(
            instances, InstanceFormat.CONDITIONAL_GENERATION, self.task)

        random_subsample_seed = 0 if self.random_subsample_seed is None else self.random_subsample_seed
        if self.limit is not None and len(instances) > self.limit:
            instances = instances[:self.limit] if random_subsample_seed is None else Random(random_subsample_seed).sample(instances, self.limit)
        # TODO
        return [i.source for i in instances], [i.target for i in instances]

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

    def start_profiling(
        self,
    ):
        self._p_gpu = subprocess.Popen(
            [f"{sys.executable}", f"{EFFICIENCY_DIR}/profile_gpu.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False
        )
        self._emission_tracker.start()
        self._start_time = time.time()

    def end_profiling(self, num_instances: int):
        time_elapsed = time.time() - self._start_time
        self._emission_tracker.stop()

        os.kill(self._p_gpu.pid, signal.SIGTERM)
        if not self._p_gpu.poll():
            print("GPU monitor correctly halted")

        self._p_gpu.wait()
        gpu_results = (self._p_gpu
                       .communicate()[0]
                       .strip()
                       .decode('UTF-8')
                       .replace("\'", "\""))
        reader = csv.DictReader(gpu_results.split("\n"), delimiter=",")
        gpu_results = [r for r in reader]
        gpu_energy, max_gpu_mem = 0, 0
        try:
            for g in gpu_results:
                gpu_energy += float(g["energy"])
                max_gpu_mem += float(g["max_mem"])
            gpu_energy = gpu_energy / 3600.0
        except KeyError:
            gpu_energy, max_gpu_mem = -999.0, -999.0

        emission_data = self._emission_tracker.final_emissions_data
        return {
            "time": time_elapsed,
            "max_gpu_mem": max_gpu_mem,
            "gpu_energy": emission_data.gpu_energy,  # kWh
            "rprof_gpu_energy": gpu_energy,
            "cpu_energy": emission_data.cpu_energy,  # kWh
            "dram_energy": emission_data.ram_energy,  # kWh
            "total_energy": emission_data.gpu_energy + emission_data.cpu_energy + emission_data.ram_energy,
            "carbon": emission_data.emissions,  # kg
            "throughput": num_instances / time_elapsed
        }

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

        print(f"{efficiency_metrics['rprof_gpu_energy']: .2f}, {efficiency_metrics['gpu_energy'] * 1000.: .2f}")

    def run(
        self,
        **kwargs
    ) -> Sequence[Any]:
        output_batches = []
        self.start_profiling()
        try:
            for output_batch in self.model.predict(instances=self._latency_instances, **kwargs):
                output_batches.append(output_batch)
            efficiency_metrics = self.end_profiling(num_instances=len(self._eval_instances))
        except:
            self.end_profiling(num_instances=len(self._eval_instances))
        ### Latency ###
        start_time = time.time()
        for output_batch in self.model.predict(instances=self._latency_instances, batch_size=1):
            pass
        elapsed_time = time.time() - start_time
        efficiency_metrics["latency"] = elapsed_time / len(self._latency_instances)
        # TODO
        # efficiency_metrics["num_params"] = self.model.model.num_parameters()
        self.tabulate_efficiency_metrics(efficiency_metrics)
        results = self.process(output_batches)
        for r in results:
            print(r)
        return results

    def process(
        self,
        output_batches: Iterable[str]
    ) -> Iterable[Dict[str, Any]]:
        yielded_label_index = -1
        for output_batch in output_batches:
            print(output_batch)
            output_batch = json.loads(output_batch.rstrip())
            for output in output_batch:
                yielded_label_index += 1
                prediction = output["output"]
                target = self._targets[yielded_label_index]
                yield {
                    "target": target,
                    "prediction": prediction,
                    "acc": (prediction, target),
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
