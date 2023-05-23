import os
from typing import Optional, Tuple

import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup

from efficiency_benchmark.steps import (CalculateMetricsStep, LogOutputStep,
                                        PredictStep, TabulateMetricsStep)
from gantry import run as gantry_run

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
    help="""Evaluation scenario [single_stream, random_batch, offline].""",
)
@click.option(
    "-b",
    "--max_batch_size",
    type=int,
    default=32,
    help="""Maximum batch size.""",
)
@click.option(
    "-o",
    "--offline_dir",
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
    scenario: str = "accuracy",
    max_batch_size: int = 32,
    offline_dir: str = f"{os.getcwd()}/datasets/efficiency-beenchmark",
    limit: Optional[int] = None,
):
    metric_task_dict = {}
    prediction_step = PredictStep(
        cmd=cmd,
        task=task,
        scenario=scenario,
        max_batch_size=max_batch_size,
        offline_dir=offline_dir,
        split=split,
        limit=limit,
    )
    predictions, efficiency_metrics = prediction_step.run()
    if scenario == "accuracy":
        metric_step = CalculateMetricsStep(task=task)
        metrics = metric_step.calculate_metrics(predictions=predictions)
        metric_task_dict[task] = metrics
        output_step = LogOutputStep(task=task, output_file=output_file)
        output_step.run(predictions=predictions)

    table_step = TabulateMetricsStep()
    table_step_result = table_step.run(metrics=metric_task_dict)

    print("\n".join(table_step_result))
    prediction_step.tabulate_efficiency_metrics(efficiency_metrics)


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
    "-l",
    "--limit",
    type=int,
    default=None,
    help="""Limit.""",
)
@click.option(
    "-b",
    "--max_batch_size",
    type=int,
    default=32,
    help="""Maximum batch size.""",
)
@click.option(
    "--gpus",
    type=int,
    help="""Minimum number of GPUs (e.g. 1).""",
)
@click.option(
    "--dataset",
    type=str,
    multiple=True,
    help="""An input dataset in the form of 'dataset-name:/mount/location' to attach to your experiment.
    You can specify this option more than once to attach multiple datasets.""",
)
def submit(
    cmd: Tuple[str, ...],
    task: str,
    split: str = "validation",
    limit: int = None,
    max_batch_size: int = 32,
    gpus: int = 1,
    dataset: Optional[Tuple[str, ...]] = None,
):
    gantry_run(
        arg=cmd,
        task=task,
        split=split,
        limit=limit,
        max_batch_size=max_batch_size,
        cluster=["efficiency-benchmark/elanding-rtx-8000"], # TODO
        beaker_image="haop/efficiency-benchmark",  # TODO
        workspace="efficiency-benchmark/efficiency-benchmark",
        cpus= None,
        gpus=gpus,
        allow_dirty=True,
        dataset=dataset
    )


if __name__ == "__main__":
    main()
