import time
from typing import Any, Dict, List, Optional

from codecarbon import EmissionsTracker
from codecarbon.core.gpu import get_gpu_details, is_gpu_details_available
from codecarbon.external.scheduler import PeriodicScheduler

NUM_LATENCY_INSTANCES = 100


"""A wrapper of codecarbon EmissionsTracker aiming to provide GPU memory and untilization data."""
class Profiler():

    def __init__(
            self,
            interval: float = 0.1,
            # gpu_ids: Optional[Iterable[int]] = None,
            **kwargs):
        # self.gpu_ids = gpu_ids
        self._start_time: Optional[float] = None
        self._emission_tracker = EmissionsTracker(
            measure_power_secs=interval,
            log_level="warning",
            # gpu_ids=gpu_ids
            **kwargs
        )
        self._gpu_details_available: bool = is_gpu_details_available()
        self._gpu_scheduler: Optional[PeriodicScheduler] = None
        self._max_used_gpu_memory: Optional[float] = None
        self._gpu_utilization: Optional[float] = None
        self._gpu_utilization_count: Optional[int] = None

        if self._gpu_details_available:
            self._gpu_scheduler = PeriodicScheduler(
                function=self._profile_gpu,
                interval=interval,
            )
            self._max_used_gpu_memory = -1.0
            self._gpu_utilization = 0.0
            self._gpu_utilization_count = 0

    def _profile_gpu(self):
        all_gpu_details: List[Dict] = get_gpu_details()
        used_memory = sum(
            [
                gpu_details["used_memory"]
                for idx, gpu_details in enumerate(all_gpu_details)
                # if idx in self.gpu_ids
             ]
        )
        gpu_utilization = sum(
            [
                gpu_details["gpu_utilization"]
                for idx, gpu_details in enumerate(all_gpu_details)
                # if idx in self.gpu_ids
             ]
        )
        self._max_used_gpu_memory = max(self._max_used_gpu_memory, used_memory)
        self._gpu_utilization += gpu_utilization
        self._gpu_utilization_count += 1

    def start(self) -> None:
        self._emission_tracker.start()
        if self._gpu_details_available:
            self._gpu_scheduler.start()
        self._start_time = time.time()

    def stop(self) -> Dict[str, Any]:
        time_elapsed = time.time() - self._start_time
        self._emission_tracker.stop()
        try:
            self._gpu_scheduler.stop()
        except:
            raise RuntimeError("Failed to stop gpu scheduler.")
        self._profile_gpu()
        self._max_used_gpu_memory = self._max_used_gpu_memory / 2 ** 30
        self._gpu_utilization /= self._gpu_utilization_count
        codecarbon_data = self._emission_tracker.final_emissions_data
        self.efficiency_metrics: Dict[str, Any] = {
            "time": time_elapsed,
            "max_gpu_mem": self._max_used_gpu_memory,
            "gpu_energy": codecarbon_data.gpu_energy,  # kWh
            "cpu_energy": codecarbon_data.cpu_energy,  # kWh
            "dram_energy": codecarbon_data.ram_energy,  # kWh
            "total_energy": codecarbon_data.gpu_energy + codecarbon_data.cpu_energy + codecarbon_data.ram_energy,
            "carbon": codecarbon_data.emissions
        }
        return self.efficiency_metrics
