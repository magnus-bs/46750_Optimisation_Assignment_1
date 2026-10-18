"""Compare flexible-load models under price profiles with different temporal structure."""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import InputData, load_question
from .model import FlexibleConsumerModel


MODEL_CASES = (
    ("linear", "Q2_linear"),
    ("quadratic", "Q2_quadratic"),
    ("rolling", "Q2_temporal"),
)
PRICE_PROFILES = ("alternating", "blocks")
METRICS = (
    "generalized_surplus_DKK",
    "procurement_cost_DKK",
    "disutility_DKK",
    "total_absolute_deviation_kWh",
    "daily_energy_kWh",
)


def _make_price_profiles(prices: np.ndarray) -> dict[str, np.ndarray]:
    if len(prices) % 2:
        raise ValueError("The price-order experiment requires an even number of hourly prices")

    ordered = np.sort(prices)
    half = len(ordered) // 2
    low, high = ordered[:half], ordered[half:]
    return {
        "alternating": np.column_stack((low, high)).ravel(),
        "blocks": np.concatenate((low, high)),
    }


def _assert_same_consumer(inputs: list[InputData]) -> None:
    reference = inputs[0]
    for data in inputs[1:]:
        for field in ("pv_available", "reference_load"):
            left, right = getattr(reference, field), getattr(data, field)
            if left is None or right is None or not np.array_equal(left, right):
                raise ValueError(f"Model cases must use the same {field}")
        for field in (
            "import_tariff",
            "export_tariff",
            "pv_marginal_cost",
            "load_min_kWh",
            "load_max_kWh",
        ):
            if getattr(reference, field) != getattr(data, field):
                raise ValueError(f"Model cases must use the same {field}")
        if not np.array_equal(reference.energy_price, data.energy_price):
            raise ValueError("Model cases must use the same base energy-price profile")


def run_temporal_structure_experiment(out: Path | str, show: bool = False) -> None:
    """Solve all penalty models under alternating and block-ordered prices.

    Each price profile is evaluated at every cyclic rotation to balance its alignment
    with the fixed PV and reference-load profiles.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [(name, load_question(case)) for name, case in MODEL_CASES]
    _assert_same_consumer([data for _, data in cases])
    if cases[0][1].reference_load is None:
        raise ValueError("The Q2 model cases must define a reference load")

    base_prices = cases[0][1].energy_price
    profiles = _make_price_profiles(base_prices)
    if not np.array_equal(np.sort(profiles["alternating"]), np.sort(profiles["blocks"])):
        raise AssertionError("The price profiles must contain exactly the same hourly prices")

    raw_rows: list[dict[str, str | int | float]] = []
    n_hours = len(base_prices)
    for profile_name, profile in profiles.items():
        for rotation in range(n_hours):
            rotated_prices = np.roll(profile, rotation)
            for model_name, base_data in cases:
                data = replace(base_data, energy_price=rotated_prices)
                results = FlexibleConsumerModel(
                    data,
                    name=f"temporal_{model_name}_{profile_name}_{rotation}",
                ).build().solve()
                deviation = (
                    results.hourly["load"].to_numpy() - data.reference_load
                )
                assert (results.hourly["load"].to_numpy() - data.reference_load).max() < 1e-6 or model_name == "linear" and False, "load above reference"
                raw_rows.append({
                    "profile": profile_name,
                    "model": model_name,
                    "rotation_hour": rotation,
                    "generalized_surplus_DKK": -results.objective,
                    "procurement_cost_DKK": results.procurement_cost,
                    "disutility_DKK": results.disutility,
                    "total_absolute_deviation_kWh": float(np.abs(deviation).sum()),
                    "daily_energy_kWh": float(results.hourly["load"].sum()),
                })

    raw_path = out / "temporal_structure_rotations.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    summary: dict[tuple[str, str], dict[str, float | str]] = {}
    for profile_name in PRICE_PROFILES:
        for model_name, _ in MODEL_CASES:
            rows = [
                row for row in raw_rows
                if row["profile"] == profile_name and row["model"] == model_name
            ]
            aggregate: dict[str, float | str] = {
                "profile": profile_name,
                "model": model_name,
                "rotations": len(rows),
            }
            for metric in METRICS:
                values = np.array([float(row[metric]) for row in rows])
                aggregate[f"mean_{metric}"] = float(values.mean())
                aggregate[f"sd_{metric}"] = float(values.std())
            summary[(profile_name, model_name)] = aggregate

    summary_path = out / "temporal_structure_summary.csv"
    fieldnames = list(next(iter(summary.values())))
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary.values())

    profiles_path = out / "temporal_structure_prices.csv"
    with profiles_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("hour", "alternating_price_DKK_per_kWh", "block_price_DKK_per_kWh"))
        writer.writerows(
            (hour, profiles["alternating"][hour], profiles["blocks"][hour])
            for hour in range(n_hours)
        )

    figure, axes = plt.subplots(3, 1, figsize=(11, 12))
    hours = np.arange(n_hours)
    axes[0].step(hours, profiles["alternating"], where="mid", label="alternating")
    axes[0].step(hours, profiles["blocks"], where="mid", label="blocks")
    axes[0].set(
        title="Same hourly price values, different temporal order",
        xlabel="Hour",
        ylabel="Price (DKK/kWh)",
        xlim=(0, n_hours - 1),
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    model_names = [name for name, _ in MODEL_CASES]
    x = np.arange(len(model_names))
    width = 0.36
    for axis, metric, title, ylabel in (
        (
            axes[1],
            "generalized_surplus_DKK",
            "Mean generalized daily surplus across rotations",
            "Surplus (DKK/day)",
        ),
        (
            axes[2],
            "total_absolute_deviation_kWh",
            "Mean total absolute deviation across rotations",
            "Deviation (kWh/day)",
        ),
    ):
        for offset, profile_name, color in (
            (-width / 2, "alternating", "tab:blue"),
            (width / 2, "blocks", "tab:orange"),
        ):
            aggregates = [summary[(profile_name, name)] for name in model_names]
            means = [float(row[f"mean_{metric}"]) for row in aggregates]
            deviations = [float(row[f"sd_{metric}"]) for row in aggregates]
            axis.bar(
                x + offset,
                means,
                width,
                yerr=deviations,
                capsize=3,
                label=profile_name,
                color=color,
            )
        axis.set(title=title, ylabel=ylabel, xticks=x, xticklabels=model_names)
        axis.legend()
        axis.grid(True, axis="y", alpha=0.25)

    figure.tight_layout()
    plot_path = out / "temporal_structure_comparison.png"
    figure.savefig(plot_path, dpi=160)
    if show:
        plt.show()
    plt.close(figure)

    print(f"Temporal-structure experiment ({n_hours} rotations per profile/model)")
    print(
        f"{'profile':>12} {'model':>11} {'surplus':>12} {'procurement':>13} "
        f"{'disutility':>11} {'abs deviation':>15}"
    )
    for (profile_name, model_name), row in summary.items():
        print(
            f"{profile_name:>12} {model_name:>11} "
            f"{float(row['mean_generalized_surplus_DKK']):12.3f} "
            f"{float(row['mean_procurement_cost_DKK']):13.3f} "
            f"{float(row['mean_disutility_DKK']):11.3f} "
            f"{float(row['mean_total_absolute_deviation_kWh']):15.3f}"
        )
    print(f"\nSummary: {summary_path}")
    print(f"All rotations: {raw_path}")
    print(f"Price profiles: {profiles_path}")
    print(f"Figure: {plot_path}")
