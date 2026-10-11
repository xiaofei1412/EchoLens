"""Deterministic epoch batches and the predeclared two-epoch decision."""
import random


def epoch_batches(size, epoch, accumulation=16, seed=20261011):
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return [order[i:i + accumulation] for i in range(0, size, accumulation)]


def second_epoch_decision(initial, midpoint, endpoint):
    improvement = (midpoint - endpoint) / midpoint
    return {"continue": endpoint < initial and improvement >= 0.01,
            "initial_dev_loss": initial, "midpoint_dev_loss": midpoint,
            "endpoint_dev_loss": endpoint, "late_relative_improvement": improvement,
            "rule": "epoch1 endpoint below initial and at least 1% below step25"}
