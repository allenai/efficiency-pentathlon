from typing import Dict
from catwalk.model import Model
from catwalk.models.efficiency_benchmark import BenchmarkModel

MODELS: Dict[str, Model] = {"efficiency-benchmark": BenchmarkModel("path")}
