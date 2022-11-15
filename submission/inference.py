import csv
from typing import List, Tuple
from datasets import load_dataset
from example_mbart_translation import MBartSubmission


DEVICE = "cuda:0"


def get_dataset() -> Tuple[List[str], List[str]]:
    data = load_dataset("wmt14", "de-en")["validation"]["translation"]
    inputs, references = [], []
    for instance in data:
        inputs.append(instance["en"])
        references.append(instance["de"])
    return inputs, references

    
if __name__ == "__main__":
    submission = MBartSubmission(
        device=DEVICE,
        ### TODO(participants): additional args can be added here. ###
        args={
            "batch_size": 32,
        }
        ### End ###
    )

    inputs, references = get_dataset()
    outputs = submission.inference(inputs)
    with open("output.csv", "w") as fout:
        fieldnames = ["id", "input", "prediction", "reference"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()
        for i in range(len(inputs)):
            writer.writerow({
                "id": i, 
                "input": inputs[i],
                "prediction": outputs[i], 
                "reference": references[i]
            })
    
    # TODO(haop@) below
    def eval():
        pass
    eval()
