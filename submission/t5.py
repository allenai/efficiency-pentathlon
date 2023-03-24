import argparse
from typing import Any, Dict, Optional, Sequence

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import resolve_device, stdio_predictor_wrapper


class T5():
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            task: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.task = task
        self.load_model()

    def load_model(self):
        device = resolve_device()
        ### TODO(participants): load models and necessary tools. ###
        self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.pretrained_model_name_or_path).half().to(device)

        self.instructions: Optional[Dict[str, str]] = {
            "rte": "You're given a pair of sentences: a Text and a Hypothesis. " \
                + "Your job is to determine the relation between them based on your inference from the statement " \
                + "and your commonsense knowledge. " \
                + f"Answer 'entailment' if the Hypothesis can be inferred from the Text; " \
                + f"Answer 'not_entailment' if the Hypothesis disagrees with the Text.\n",
            "nli": "You're given a pair of sentences: a Premise and a Hypothesis. "\
                + "Your job is to determine the relation between them based on your inference from the statement " \
                + "and your commonsense knowledge. " \
                + f"Answer 'entailment' if the Hypothesis can be inferred from the Premise; " \
                + f"Answer 'contradiction' if the Hypothesis disagrees with the Premise; " \
                + f"Answer 'neutral' if the Hypothesis can neither be inferred from the Premise or disagrees with the Premise.\n",
            "qqp": "You're given a pair of questions. Your job is to determine whether they are duplicate " \
                + f"Answer 'duplicate' if they bear the same meaning; " \
                + f"Answer 'not_duplicate' if they have different meanings.\n"

        }
        # {"sentence1": "111", "sentence2": 222}
        if "flan" in self.pretrained_model_name_or_path:
            self.convert_fns = {
                "rte": lambda text: f"{self.instructions['rte']}Text: {text['sentence1']}\nHypothesis: {text['sentence2']}\n",
                "mnli": lambda text: f"{self.instructions['nli']}Premise: {text['premise']}\nHypothesis: {text['hypothesis']}\n",
                "snli": lambda text: f"{self.instructions['nli']}Premise: {text['premise']}\nHypothesis: {text['hypothesis']}\n",
                "qqp": lambda text: f"{self.instructions['qqp']}Question1: {text['question1']}\nQuestion2: {text['question2']}\n"
            }
        else:
            self.convert_fns = {
                "rte": lambda text: f"rte sentence1: {text['sentence1']} sentence2: {text['sentence2']} ",
                "mnli": lambda text: f"mnli hypothesis: {text['hypothesis']} premise: {text['premise']} ",
                "snli": lambda text: f"snli hypothesis: {text['hypothesis']} premise: {text['premise']} ",
                "qqp": lambda text: f"qqp question1: {text['question1']} question2: {text['question2']} "
            }

    def predict(  # type: ignore
        self,
        inputs: Sequence[Dict[str, Any]]
    ) -> Sequence[str]:
        convert_fn = self.convert_fns[self.task]
        inputs = [convert_fn(input) for input in inputs]
        with torch.inference_mode():
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
            return outputs
            # for output in outputs:
            #     yield output.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()

    classifier = T5(
        pretrained_model_name_or_path=args.model,
        task=args.task
    )
    stdio_predictor_wrapper(classifier)