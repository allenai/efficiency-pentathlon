from typing import Optional, Tuple

import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup

from gantry import run as gantry_run
from efficiency_benchmark.steps import (CalculateMetricsStep, PredictStep,
                                        TabulateMetricsStep)

_CLICK_GROUP_DEFAULTS = {
    "cls": HelpColorsGroup,
    "help_options_color": "green",
    "help_headers_color": "yellow",
    "context_settings": {"max_content_width": 115},
}

_CLICK_COMMAND_DEFAULTS = {
    "cls": HelpColorsCommand,
    "help_options_color": "green",
    "help_headers_color": "yellow",
    "context_settings": {"max_content_width": 115},
}


@click.group(**_CLICK_GROUP_DEFAULTS)
def main():
    pass


@main.command(**_CLICK_COMMAND_DEFAULTS)
@click.argument("cmd", nargs=-1)
@click.option(
    "-t",
    "--task",
    type=str,
    nargs=1,
    help="""Tasks.""",
)
@click.option(
    "--split",
    type=str,
    help="""Split.""",
)
@click.option(
    "-s",
    "--scenario",
    type=str,
    default="single_stream",
    help="""Evaluation scenario [single_stream, batched, offline].""",
)
@click.option(
    "-o",
    "--output_file",
    type=str,
    nargs=1,
    help="""Output file.""",
)
@click.option(
    "-l",
    "--limit",
    type=int,
    default=None,
    help="""Limit.""",
)
def run(
    cmd: Tuple[str, ...],
    task: str,
    split: str = "validation",
    scenario: str = "single_stream",
    output_file: Optional[str] = None,
    limit: Optional[int] = None,
):
    batch_size=1 if scenario == "single_stream" else 32  # TODO
    prediction_step = PredictStep(
        cmd=cmd,
        task=task,
        split=split,
        batch_size=batch_size,
        limit=limit,
    )
    predictions = prediction_step.run(batch_size=batch_size)

    metric_task_dict = {}
    metric_step = CalculateMetricsStep(task=task)
    metrics = metric_step.calculate_metrics(predictions=predictions)
    metric_task_dict[task] = metrics

    table_step = TabulateMetricsStep()
    table_step_result = table_step.run(metrics=metric_task_dict)
    print("\n".join(table_step_result))
    # Logging results
    # gantry saves `/results` to Beaker. We output to this folder if runnning on Beaker
    try:
        with open("/results/outputs", "w") as fout:
            for p in predictions:
                fout.write(p["output"] + "\n")
    except:
        # Running locally.
        if output_file is not None:
            with open(output_file, "w") as fout:
                for p in predictions:
                    fout.write(p["output"] + "\n")
        else:
            pass
            # for p in predictions:
            #     print(p)


@main.command(**_CLICK_COMMAND_DEFAULTS)
@click.argument("cmd", nargs=-1)
@click.option(
    "-t",
    "--task",
    type=str,
    nargs=1,
    help="""Tasks.""",
)
@click.option(
    "--split",
    type=str,
    help="""Split.""",
)
@click.option(
    "-b",
    "--batch_size",
    type=int,
    default=32,
    help="""Batch size.""",
)
@click.option(
    "-l",
    "--limit",
    type=int,
    default=None,
    help="""Limit.""",
)
@click.option(
    "--gpus",
    type=int,
    help="""Minimum number of GPUs (e.g. 1).""",
)
def submit(
    cmd: Tuple[str, ...],
    task: str,
    split: str = "validation",
    batch_size: int = 32,
    limit: int = None,
    gpus: int = 1
):
    gantry_run(
        arg=cmd,
        task=task,
        # name="efficiency-benchmark-submission",
        cluster=["efficiency-benchmark/elanding-rtx-8000"], # TODO
        beaker_image="haop/efficiency-benchmark",  # TODO
        workspace="efficiency-benchmark/efficiency-benchmark",
        cpus= None,
        gpus=gpus,
        allow_dirty=True
    )


if __name__ == "__main__":
    main()
