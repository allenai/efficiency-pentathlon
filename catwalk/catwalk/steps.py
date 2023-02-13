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
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import torch
from tango import JsonFormat, Step
from tango.common.sequences import SqliteSparseSequence
from tango.format import SqliteSequenceFormat, TextFormat

import docker
from catwalk.efficiency.carbon import get_realtime_carbon  # (TODO)
from catwalk.model import Model
from catwalk.models import MODELS
from catwalk.task import Task
from catwalk.tasks import TASKS


EFFICIENCY_DIR = f"{pathlib.Path(__file__).parent.resolve()}/efficiency"  # TODO


class PredictStep(Step):
    VERSION = "001"
    SKIP_ID_ARGUMENTS = {"batch_size"}
    FORMAT = SqliteSequenceFormat

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
        client = docker.from_env()
        self._container = client.containers.run(
            "test:latest",
            "python3 profile_cpu.py",
            name="test",
            privileged=True,
            tty=True,
            remove=True,
            detach=True,
            stdout=True,
            stderr=True
        )
        self._p_gpu = subprocess.Popen(
            [f"{sys.executable}", f"{EFFICIENCY_DIR}/profile_gpu.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False
        )
        self._start_time = time.time()

    def end_profiling(self):
        time_elapsed = time.time() - self._start_time
        os.kill(self._p_gpu.pid, signal.SIGTERM)
        # if not self._p_gpu.poll():
        #     print("GPU monitor correctly halted")
        self._container.kill(signal.SIGINT)
        cpu_results = json.loads(self._container
                                .logs()
                                .strip()
                                .decode('UTF-8')
                                .replace("\'", "\""))
        self._p_gpu.wait()
        self._container.stop()
        gpu_results = (self._p_gpu
                       .communicate()[0]
                       .strip()
                       .decode('UTF-8')
                       .replace("\'", "\""))
        reader = csv.DictReader(gpu_results.split("\n"), delimiter=",")
        gpu_results = [r for r in reader]

        gpu_energy, max_gpu_mem = 0, 0
        for g in gpu_results:
            gpu_energy += float(g["energy"])
            max_gpu_mem += float(g["max_mem"])
        gpu_energy = gpu_energy / 3600.0
        cpu_energy = cpu_results["cpu_energy"] / 3600.0  # Wh
        dram_energy = cpu_results["dram_energy"] / 3600.0  # Wh

        total_energy = gpu_energy + cpu_energy + dram_energy
        carbon = get_realtime_carbon(total_energy)  # in g
        return {
            "time": time_elapsed,
            "max_gpu_mem": max_gpu_mem,
            "gpu_energy": gpu_energy,  # Wh
            "cpu_energy": cpu_energy,  # Wh
            "dram_energy": dram_energy,  # Wh
            "total_energy": total_energy,
            "carbon": carbon
        }

    def tabulate_efficiency_metrics(
        self,
        efficiency_metrics: Dict[str, Any]
    ):
        print(f"Time Elapsed: {efficiency_metrics['time']:.2f} s")
        # print(f"Max DRAM Memory Usage: {max_mem_util * total_memory: .2f} GiB")
        print(f"Max GPU Memory Usage: {efficiency_metrics['max_gpu_mem']: .2f} GiB")
        print(f"GPU Energy: {efficiency_metrics['gpu_energy']:.2e} Wh")
        print(f"CPU Energy: {efficiency_metrics['cpu_energy']: .2e} Wh")
        print(f"Memory Energy: {efficiency_metrics['dram_energy']: .2e} Wh")
        print(f"Total Energy: {efficiency_metrics['total_energy']: .2e} Wh")
        print(f"CO2 emission: {efficiency_metrics['carbon']: .2e} grams.")

    def run(
        self,
        model: Union[str, Model],
        task: Union[str, Task],
        split: Optional[str] = None,
        limit: Optional[int] = None,
        random_subsample_seed: Optional[int] = None,
        **kwargs
    ) -> Sequence[Any]:
        if isinstance(model, str):
            model = MODELS[model]
        if isinstance(task, str):
            task = TASKS[task]
        if split is None:
            split = task.default_split

        results = SqliteSparseSequence(self.work_dir_for_run / "result.sqlite")
        instances = task.get_split(split)
        if limit is not None and len(instances) > limit:
            instances = instances[:limit] if random_subsample_seed is None else Random(random_subsample_seed).sample(instances, limit)
        instances = instances[len(results):]
        model.prepare(task, instances)

        self.start_profiling()
        for result in model.predict(task, **kwargs):
            results.append(result)
        efficiency_metrics = self.end_profiling()
        self.tabulate_efficiency_metrics(efficiency_metrics)
        return results


class CalculateMetricsStep(Step):
    VERSION = "001"
    FORMAT = JsonFormat

    def massage_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(kwargs["model"], str):
            kwargs["model"] = MODELS[kwargs["model"]]
        if isinstance(kwargs["task"], str):
            kwargs["task"] = TASKS[kwargs["task"]]
        return kwargs

    def run(
        self,
        model: Union[str, Model],
        task: Union[str, Task],
        predictions: Sequence[Any]
    ) -> Dict[str, float]:
        if isinstance(model, str):
            model = MODELS[model]
        if isinstance(task, str):
            task = TASKS[task]

        return model.calculate_metrics(task, predictions)


class TabulateMetricsStep(Step):
    VERSION = "001"
    FORMAT = TextFormat

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