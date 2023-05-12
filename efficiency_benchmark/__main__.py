from typing import Optional, Tuple

import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup

from gantry import run as gantry_run
from efficiency_benchmark.steps import (CalculateMetricsStep, PredictStep,
                                        TabulateMetricsStep, LogOutputStep)

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
    scenario: str = "random_batch",
    output_file: Optional[str] = None,
    limit: Optional[int] = None,
):
    prediction_step = PredictStep(
        cmd=cmd,
        task=task,
        scenario=scenario,
        split=split,
        limit=limit,
    )
    predictions, efficiency_metrics = prediction_step.run()

    metric_task_dict = {}
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
    gpus: int = 1,
    dataset: Optional[Tuple[str, ...]] = None,
):
    gantry_run(
        arg=cmd,
        task=task,
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
