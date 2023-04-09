import json
import sys
import argparse

import torch
import transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer


# Submission
class MBART():
    def __init__(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro").to(device)
        print(device)
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX")

        # TODO
        #self._convert_fn = lambda text: text["input"]

    def predict(self, inputs):
        # inputs = [self._convert_fn(i) for i in inputs]
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            padding=True,
            truncation="only_first",
            return_tensors="pt",
            pad_to_multiple_of=8,
        ).input_ids
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(inputs, max_length=10)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for output in outputs:
            yield output.strip()
