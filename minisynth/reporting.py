"""Helpers for keeping generated JSON reports compact and reviewable."""

from copy import deepcopy


MODEL_METRIC_KEYS = (
    "model_id",
    "model_type",
    "waveform_mode",
    "target_mode",
    "model_size",
    "pooling_mode",
    "head_mode",
    "loss_preset",
    "tensor_path",
    "tensor_sharded",
    "tensor_shard_count",
    "metadata_path",
    "num_samples",
    "num_features",
    "num_targets",
    "target_parameters",
    "train_samples",
    "test_samples",
    "benchmark_samples",
    "epochs",
    "completed_epochs",
    "batch_size",
    "learning_rate",
    "optimizer",
    "weight_decay",
    "scheduler",
    "scheduler_step_size",
    "scheduler_gamma",
    "early_stopping_patience",
    "checkpoint_selection",
    "best_epoch",
    "best_test_loss",
    "best_test_objective_loss",
    "test_size",
    "benchmark_size",
    "device",
    "train_loss",
    "train_objective_loss",
    "test_loss",
    "test_objective_loss",
    "train_parameter_mse",
    "test_parameter_mse",
    "train_mae",
    "test_mae",
    "train_continuous_mae",
    "test_continuous_mae",
    "train_grouped_mae",
    "test_grouped_mae",
    "train_per_parameter_mae",
    "test_per_parameter_mae",
    "train_waveform_accuracy",
    "test_waveform_accuracy",
    "train_waveform_accuracy_by_name",
    "test_waveform_accuracy_by_name",
)


LONG_METRIC_KEYS = {
    "train_indices",
    "test_indices",
    "benchmark_indices",
    "train_losses",
    "test_losses",
    "train_objective_losses",
    "test_objective_losses",
    "learning_rates",
    "train_prediction_distribution",
    "test_prediction_distribution",
    "benchmark_prediction_distribution",
}


def compact_model_metrics(metrics, loss_tail=5):
    if not metrics:
        return {}

    compact = {
        key: deepcopy(metrics[key])
        for key in MODEL_METRIC_KEYS
        if key in metrics
    }

    loss_history = {}
    if "train_losses" in metrics:
        loss_history["train_tail"] = list(metrics["train_losses"][-loss_tail:])
    if "test_losses" in metrics:
        loss_history["test_tail"] = list(metrics["test_losses"][-loss_tail:])
    if "learning_rates" in metrics:
        loss_history["learning_rate_tail"] = list(metrics["learning_rates"][-loss_tail:])
    if loss_history:
        compact["loss_history_tail"] = loss_history

    return compact


def prune_long_metrics(metrics):
    if not isinstance(metrics, dict):
        return metrics

    return {
        key: deepcopy(value)
        for key, value in metrics.items()
        if key not in LONG_METRIC_KEYS
    }


def compact_clip_result(result):
    clip = {
        "index": result.get("index"),
        "seed": result.get("seed"),
        "audio_path": result.get("audio_path"),
    }
    if "comparison" in result:
        clip["weighted_distance"] = result["comparison"].get("weighted_distance")
    if "error" in result:
        clip["error"] = result["error"]
    return clip


def compact_prediction_distribution(distribution):
    if not distribution:
        return {}

    compact = {}
    for name, metrics in distribution.items():
        compact[name] = {
            key: metrics[key]
            for key in (
                "mean_delta",
                "target_std",
                "predicted_std",
                "std_ratio",
            )
            if key in metrics
        }
    return compact
