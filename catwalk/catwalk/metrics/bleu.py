import math
from typing import Union, Optional, Any, Dict

import torch
from torchmetrics.aggregation import BaseAggregator


class BLEUMetric(BaseAggregator):
    def __init__(
        self,
        base: int = 2,  # Does anyone ever use anything but 2 here?
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Dict[str, Any],
    ):
        super().__init__("sum", [], nan_strategy, **kwargs)
        self.base = base
        # TODO

    def update(
        self,
    ) -> None:  # type: ignore
        # TODO
        pass

    def compute(self) -> torch.Tensor:
        # TODO
        pass