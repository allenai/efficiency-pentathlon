from typing import Tuple

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
    "-b",
    "--batch_size",
    type=str,
    help="""Batch size.""",
)
def run(
    cmd: Tuple[str, ...],
    task: str,
    split: str = "validation",
    batch_size: int = 32,
):
    # _parser = argparse.ArgumentParser()
    # _parser.add_argument('--task', type=str, nargs="+")
    # _parser.add_argument('--split', type=str)
    # _parser.add_argument('--batch_size', type=int, default=32)
    # _parser.add_argument('--num_shots', type=int)
    # _parser.add_argument('--fewshot_seed', type=int)
    # _parser.add_argument('--limit', type=int)
    # _parser.add_argument(
    #     '-d', '-w',
    #     type=str,
    #     default=None,
    #     metavar="workspace",
    #     dest="workspace",
    #     help="the Tango workspace with the cache")
    # _parser.add_argument('cmd', nargs='*')
    # args = _parser.parse_args()

    metric_task_dict = {}
    prediction_step = PredictStep(
        cmd=cmd,
        task=task,
        split=split,
        batch_size=batch_size,
    )
    predictions = prediction_step.run(batch_size=batch_size)
    metric_step = CalculateMetricsStep(task=task)
    metrics = metric_step.calculate_metrics(predictions=predictions)
    metric_task_dict[task] = metrics

    table_step = TabulateMetricsStep()
    table_step_result = table_step.run(metrics=metric_task_dict)
    print("\n".join(table_step_result))


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
    type=str,
    help="""Batch size.""",
)
def submit(
    cmd: Tuple[str, ...],
    task: str,
    split: str = "validation",
    batch_size: int = 32,
):
    gantry_run(
        arg=cmd,
        # name="efficiency-benchmark-submission",
        cluster=["efficiency-benchmark/elanding-rtx-8000"], # TODO
        beaker_image="haop/efficiency-benchmark",  # TODO
        workspace="efficiency-benchmark/efficiency-benchmark",
        cpus= None,
        gpus=None,
        allow_dirty=True
    )


if __name__ == "__main__":
    main()
