import time
import numpy as np
import threading
from typing import Any, Dict, List, Optional

from efficiency_benchmark.efficiency.power_monitor import PowerMonitor
from efficiency_benchmark.efficiency.power_monitor import POWER_FIELDS
from efficiency_benchmark.efficiency.power_monitor import IDLE_POWER
from codecarbon import EmissionsTracker
from codecarbon.core.gpu import get_gpu_details, is_gpu_details_available
from codecarbon.external.scheduler import PeriodicScheduler
from codecarbon.core.units import Energy, Power, Time


"""A wrapper of codecarbon EmissionsTracker aiming to provide GPU memory and untilization data."""
class Profiler():

    def __init__(
            self,
            interval: float = 0.1,
            is_submission: bool = False,
            gpu_ids: Optional[List[int]] = [],
            **kwargs):
        self.gpu_ids = gpu_ids
        self.interval = interval
        is_submission = True
        self._is_submission = is_submission
        self._start_time: Optional[float] = None
        self._emission_tracker = EmissionsTracker(
            measure_power_secs=interval,
            log_level="error",
            gpu_ids=self.gpu_ids,
            **kwargs
        )
        self._try_power_monitor()
        self._gpu_details_available: bool = is_gpu_details_available()
        self._gpu_scheduler: Optional[PeriodicScheduler] = None
        self._max_used_gpu_memory: Optional[float] = None
        self._gpu_utilization: Optional[float] = None
        self._gpu_reads: Optional[int] = None
        self._gpu_power: Optional[float] = None
        self._idle_power: float = self.get_idle_power()

        if self._gpu_details_available:
            self._gpu_scheduler = PeriodicScheduler(
                function=self._profile_gpu,
                interval=interval,
            )
            self._max_used_gpu_memory = -1.0
            self._gpu_utilization = 0.0
            self._gpu_power = 0.0
            self._gpu_reads = 0

    def _try_power_monitor(self):
        if self._is_submission:
            self._power_monitor_ser = PowerMonitor.try_open_serial_port()
            if self._power_monitor_ser is not None:
                self._power_monitor_reads: List = []
                stop_event = threading.Event()
                self._power_monitor = PowerMonitor(
                    stop_event=stop_event,
                    target=PowerMonitor.read_power_monitor,
                    args=(self._power_monitor_ser, stop_event, self._power_monitor_reads)
                )
                self._use_power_monitor = True
                print("Power monitor is available.")
                return
        self._use_power_monitor = False
        print("Power monitor is not available; using codecarbon estimation.")

    def _profile_gpu(self):
        all_gpu_details: List[Dict] = get_gpu_details()
        used_memory = sum(
            [
                gpu_details["used_memory"]
                for idx, gpu_details in enumerate(all_gpu_details)
                if self.gpu_ids is None or idx in self.gpu_ids
             ]
        )
        gpu_utilization = sum(
            [
                gpu_details["gpu_utilization"]
                for idx, gpu_details in enumerate(all_gpu_details)
                if self.gpu_ids is None or idx in self.gpu_ids
             ]
        )
        gpu_power = sum(
            [
                gpu_details["power_usage"] * 1e-3
                for idx, gpu_details in enumerate(all_gpu_details)
                if self.gpu_ids is None or idx in self.gpu_ids
             ]
        )
        self._max_used_gpu_memory = max(self._max_used_gpu_memory, used_memory)
        self._gpu_utilization += gpu_utilization

        self._gpu_power += gpu_power
        self._gpu_reads += 1

    def get_idle_power(self) -> float:
        # if self._is_submission:
        #     # We know the Idle power of the dedicated machine.
        #     return IDLE_POWER
        # else:
            # Evaluating locally. Profile the idling power.
        print("Profiling the idling power ...")
        if self._use_power_monitor:
            power_monitor_ser = PowerMonitor.try_open_serial_port()
            power_monitor_reads: List = []
            stop_event = threading.Event()
            power_monitor = PowerMonitor(
                stop_event=stop_event,
                target=PowerMonitor.read_power_monitor,
                args=(power_monitor_ser, stop_event, power_monitor_reads)
            )
            power_monitor.start()
            time.sleep(10)
            power_monitor.stop()
            power_monitor.join()
            PowerMonitor.try_close_serial_port(power_monitor_ser)
            powers = []
            for r in power_monitor_reads:
                powers.append(np.array([ float(r[f]) for f in POWER_FIELDS ]).sum())
            idle_power: Power = np.array(powers).mean()
        else:
            idle_power_profiler = EmissionsTracker(
                measure_power_secs=self.interval,
                log_level="error",
                gpu_ids=self.gpu_ids
            )
            idle_power_profiler.start()
            time.sleep(10)
            idle_power_profiler.stop()
            data = idle_power_profiler.final_emissions_data
            idle_power = data.cpu_power + data.gpu_power + data.ram_power
        return idle_power

    def start(self) -> None:
        self._emission_tracker.start()
        if self._gpu_details_available:
            self._gpu_scheduler.start()
        if self._use_power_monitor:
            self._power_monitor.start()
        self._start_time = time.monotonic()

    def stop(self) -> Dict[str, Any]:
        time_elapsed = Time.from_seconds(time.monotonic() - self._start_time)
        self._emission_tracker.stop()

        if self._use_power_monitor:
            self._power_monitor.stop()
            self._power_monitor.join()
            PowerMonitor.try_close_serial_port(self._power_monitor_ser)
            powers = []
            for r in self._power_monitor_reads:
                powers.append(np.array([ float(r[f]) for f in POWER_FIELDS ]).sum())
            avg_power: Power = Power.from_watts(np.array(powers).mean() - self._idle_power)
            total_energy: Energy = Energy.from_power_and_time(power=avg_power, time=time_elapsed)
            self._emission_tracker.final_emissions_data.energy_consumed = total_energy
            self._emission_tracker.final_emissions_data = self._emission_tracker._prepare_emissions_data()
        if self._gpu_details_available:
            try:
                self._gpu_scheduler.stop()
            except:
                pass
                # raise RuntimeError("Failed to stop gpu scheduler.")
            self._profile_gpu()
            self._max_used_gpu_memory = self._max_used_gpu_memory / 2 ** 30
            self._gpu_utilization /= self._gpu_reads
            self._gpu_power = Power.from_watts(self._gpu_power / self._gpu_reads)
        codecarbon_data = self._emission_tracker.final_emissions_data

        self.efficiency_metrics: Dict[str, Any] = {
            "time": time_elapsed.seconds,  # seconds
            "max_gpu_mem": self._max_used_gpu_memory,
            "gpu_energy": codecarbon_data.gpu_energy,  # kWh
            "cpu_energy": codecarbon_data.cpu_energy,  # kWh
            "dram_energy": codecarbon_data.ram_energy,  # kWh
            "avg_gpu_power": self._gpu_power.W,  # W
            "avg_power": avg_power.W if self._use_power_monitor else codecarbon_data.cpu_power + codecarbon_data.gpu_power,  # W
            "total_energy": codecarbon_data.energy_consumed,  # kWh
            "carbon": codecarbon_data.emissions  # kg
        }
        return self.efficiency_metrics
