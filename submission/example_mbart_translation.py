import numpy as np
import itertools
from typing import Any, Dict, List
from transformers import MBartForConditionalGeneration
from transformers import MBartTokenizer
from submission import EfficiencyBenchmarkSubmission


class MBartSubmission(EfficiencyBenchmarkSubmission):
    def __init__(
        self, 
        *,
        device: str,
        args: Dict[str, Any]):
        super().__init__(device=device, args=args)

        ### TODO(participants): additional initilization code below. ###
        self.load()
        ### End ###

    def load(self):
        ### TODO(participants): load models and necessary tools. ###
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50")
        try:
            self.tokenizer = self.tokenizer.to(self.device)
        except:
            pass
        self.model = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50", 
            forced_bos_token_id=0
        ).to(self.device)
        ### End ###

    def inference(self, inputs: List[str]) -> List[str]:
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
        num_instances = len(inputs)

        indices = MBartSubmission.argsort(inputs)
        inputs = list(np.array(inputs)[indices])
        batch_size = self.args["batch_size"]
        input_batches = self.tokenizer(
            inputs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )["input_ids"]
        num_batches = (num_instances - 1) // batch_size + 1
        output_batches = []
        for i in range(num_batches):
            input_batch = input_batches[i * batch_size : (i + 1) * batch_size].to(self.device)
            output_batch = self.model.generate(input_batch)
            output_batch = self.tokenizer.batch_decode(output_batch, skip_special_tokens=True)
            output_batches.append(output_batch)
        output_batches = list(itertools.chain.from_iterable(output_batches))
        outputs = {}
        for i in range(num_instances):
            outputs[indices[i]] = output_batches[i]
        assert len(outputs) == len(inputs), \
            f"# outputs does not match # inputs: {len(outputs)} vs. {len(inputs)}."
        return outputs
        ### End ###

    ### TODO(participants): other tools below. ###
    @staticmethod
    def argsort(seq):
        lengths = [len(x.split()) for x in seq]
        return sorted(range(len(lengths)), key=lengths.__getitem__)
    ### End ###
