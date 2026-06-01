"""Slot-aware and slot-invariant oscillator mix diagnostics."""

from minisynth.oscillators import BASE_WAVES


def oscillator_level_by_wave(patch):
    levels = {wave: 0.0 for wave in BASE_WAVES}
    levels[patch["osc1_wave"]] += float(patch["osc1_level"])
    levels[patch["osc2_wave"]] += float(patch["osc2_level"])
    return levels


def oscillator_total_level(patch):
    return float(patch["osc1_level"]) + float(patch["osc2_level"])


def oscillator_balance(patch):
    total = oscillator_total_level(patch)
    if total <= 0.0:
        return 0.5
    return float(patch["osc2_level"]) / total


def canonicalize_oscillator_slots(patch):
    first = {
        "wave": patch["osc1_wave"],
        "level": float(patch["osc1_level"]),
        "detune": 0.0,
        "source": 1,
    }
    second = {
        "wave": patch["osc2_wave"],
        "level": float(patch["osc2_level"]),
        "detune": float(patch["osc2_detune"]),
        "source": 2,
    }
    ordered = sorted(
        (first, second),
        key=lambda item: (BASE_WAVES.index(item["wave"]), -item["level"], item["source"]),
    )
    result = dict(patch)
    result["osc1_wave"] = ordered[0]["wave"]
    result["osc1_level"] = ordered[0]["level"]
    result["osc2_wave"] = ordered[1]["wave"]
    result["osc2_level"] = ordered[1]["level"]
    if ordered[0]["source"] == 1:
        result["osc2_detune"] = float(patch["osc2_detune"])
    else:
        result["osc2_detune"] = -float(patch["osc2_detune"])
    return result


def categorical_wave_error(target_wave, predicted_wave):
    if len(BASE_WAVES) <= 1:
        return 0.0
    target_index = BASE_WAVES.index(target_wave)
    predicted_index = BASE_WAVES.index(predicted_wave)
    return abs(target_index - predicted_index) / (len(BASE_WAVES) - 1)


def normalized_detune_error(target_detune, predicted_detune):
    return abs(float(target_detune) - float(predicted_detune)) / 2400.0


def oscillator_slot_assignment_error(target_patch, predicted_patch, swapped=False):
    if swapped:
        predicted_osc1_wave = predicted_patch["osc2_wave"]
        predicted_osc1_level = predicted_patch["osc2_level"]
        predicted_osc2_wave = predicted_patch["osc1_wave"]
        predicted_osc2_level = predicted_patch["osc1_level"]
        predicted_detune = -float(predicted_patch["osc2_detune"])
    else:
        predicted_osc1_wave = predicted_patch["osc1_wave"]
        predicted_osc1_level = predicted_patch["osc1_level"]
        predicted_osc2_wave = predicted_patch["osc2_wave"]
        predicted_osc2_level = predicted_patch["osc2_level"]
        predicted_detune = predicted_patch["osc2_detune"]

    component_errors = (
        categorical_wave_error(target_patch["osc1_wave"], predicted_osc1_wave),
        abs(float(target_patch["osc1_level"]) - float(predicted_osc1_level)),
        categorical_wave_error(target_patch["osc2_wave"], predicted_osc2_wave),
        abs(float(target_patch["osc2_level"]) - float(predicted_osc2_level)),
        normalized_detune_error(target_patch["osc2_detune"], predicted_detune),
    )
    return float(sum(component_errors) / len(component_errors))


def oscillator_mix_error_report(target_patch, predicted_patch):
    target_levels = oscillator_level_by_wave(target_patch)
    predicted_levels = oscillator_level_by_wave(predicted_patch)
    per_wave_level_errors = {
        wave: {
            "target": float(target_levels[wave]),
            "predicted": float(predicted_levels[wave]),
            "absolute_error": float(abs(target_levels[wave] - predicted_levels[wave])),
        }
        for wave in BASE_WAVES
    }
    direct_assignment_error = oscillator_slot_assignment_error(
        target_patch,
        predicted_patch,
        swapped=False,
    )
    swapped_assignment_error = oscillator_slot_assignment_error(
        target_patch,
        predicted_patch,
        swapped=True,
    )

    target_total = oscillator_total_level(target_patch)
    predicted_total = oscillator_total_level(predicted_patch)
    target_balance = oscillator_balance(target_patch)
    predicted_balance = oscillator_balance(predicted_patch)

    return {
        "total_level": {
            "target": target_total,
            "predicted": predicted_total,
            "absolute_error": float(abs(target_total - predicted_total)),
        },
        "balance": {
            "target": target_balance,
            "predicted": predicted_balance,
            "absolute_error": float(abs(target_balance - predicted_balance)),
        },
        "per_wave_level": per_wave_level_errors,
        "direct_assignment_error": direct_assignment_error,
        "swapped_assignment_error": swapped_assignment_error,
        "best_assignment_error": float(min(direct_assignment_error, swapped_assignment_error)),
        "best_assignment": "swapped"
        if swapped_assignment_error < direct_assignment_error
        else "direct",
        "detune_note": (
            "Swapped assignment negates osc2_detune for diagnostics only; exact render "
            "equivalence still depends on the current synth's base-pitch convention."
        ),
    }


def summarize_oscillator_mix_errors(results):
    reports = [
        result["oscillator_mix_errors"]
        for result in results
        if "oscillator_mix_errors" in result
    ]
    if not reports:
        return {}

    count = len(reports)
    per_wave = {}
    for wave in BASE_WAVES:
        per_wave[wave] = float(
            sum(report["per_wave_level"][wave]["absolute_error"] for report in reports)
            / count
        )

    swapped_better_count = sum(
        1 for report in reports if report["best_assignment"] == "swapped"
    )
    return {
        "count": count,
        "mean_total_level_error": float(
            sum(report["total_level"]["absolute_error"] for report in reports) / count
        ),
        "mean_balance_error": float(
            sum(report["balance"]["absolute_error"] for report in reports) / count
        ),
        "mean_direct_assignment_error": float(
            sum(report["direct_assignment_error"] for report in reports) / count
        ),
        "mean_best_assignment_error": float(
            sum(report["best_assignment_error"] for report in reports) / count
        ),
        "swapped_better_fraction": float(swapped_better_count / count),
        "per_wave_level_mae": per_wave,
    }
