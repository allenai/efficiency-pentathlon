from typing import Any, Dict, List


class EfficiencyBenchmarkSubmission:
    def __init__(
        self, 
        *,
        device: str,
        args: Dict[str, Any]):
        self.device = device
        self.args = args

        ### TODO(participants): additional initilization code below. ###
        
        ### End ###

    def load(self):
        ### TODO(participants): load models and necessary tools. ###
        raise NotImplementedError
        ### End ###

    def inference(self, inputs: List[str]) -> Dict[int , str]:
        """Predict the outputs. 
        Each instance is an element in the `inputs` list.
        The outputs are stored in a dictionary mapping
        the index of an instance to its output.
        Args:
            inputs: List[str]. Each instance is an element in the list.
        Return: 
            outputs: Dict[int , str]. 
        """

        ### TODO(participants): inference code block below. ###
        raise NotImplementedError
        ### End ###
    
    ### TODO(participants): other tools below. ###

    ### End ###
