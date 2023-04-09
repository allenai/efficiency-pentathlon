import argparse

from tango.common.logging import initialize_logging

from efficiency_benchmark.steps import TabulateMetricsStep
from efficiency_benchmark.tasks import TASK_SETS

_parser = argparse.ArgumentParser()
_parser.add_argument('--model', type=str, required=True)
_parser.add_argument('--task', type=str, nargs="+")
_parser.add_argument('--split', type=str)
_parser.add_argument('--batch_size', type=int, default=32)
_parser.add_argument('--num_shots', type=int)
_parser.add_argument('--fewshot_seed', type=int)
_parser.add_argument('--limit', type=int)
_parser.add_argument(
    '-d', '-w',
    type=str,
    default=None,
    metavar="workspace",
    dest="workspace",
    help="the Tango workspace with the cache")


def main(args: argparse.Namespace):
    initialize_logging(log_level="WARNING")

    limit = args.limit if hasattr(args, "limit") else None

    from efficiency_benchmark.steps import CalculateMetricsStep, PredictStep

    tasks = set()
    for task in args.task:
        try:
            tasks |= TASK_SETS[task]
        except KeyError:
            tasks.add(task)

    kwargs = {}
    if args.num_shots is not None:
        kwargs["num_shots"] = args.num_shots
    if args.fewshot_seed is not None:
        kwargs["fewshot_seed"] = args.fewshot_seed

    metric_task_dict = {}
    for task in tasks:
        prediction_step = PredictStep(
            model=args.model,
            task=task,
            split=args.split,
            batch_size=args.batch_size,
            limit=limit
        )
        predictions = prediction_step.run(**kwargs)
        metric_step = CalculateMetricsStep(model=args.model, task=task)
        metrics = metric_step.run(predictions=predictions)
        metric_task_dict[task] = metrics

    table_step = TabulateMetricsStep()
    table_step_result = table_step.run(metrics=metric_task_dict)
    print("\n".join(table_step_result))


if __name__ == "__main__":
    main(_parser.parse_args())
