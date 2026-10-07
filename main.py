"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib

from src.data_loader import load_question, list_questions
from src.model import FlexibleConsumerModel, Results
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule
from src.scenarios import scale_prices, scale_pv, set_tariffs

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LINEAR_DISUTILITY_VALUES = (0.50, 1.40, 1.43, 1.60, 2.09, 2.10, 3.00)
QUADRATIC_DISUTILITY_VALUES =  (0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00)


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    model = FlexibleConsumerModel(data).build()
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    if data.reference_load is not None and (
        data.linear_disutility is not None or data.quadratic_disutility is not None
    ):
        load = results.hourly["load"].to_numpy()
        deviation = load - data.reference_load
        abs_deviation = abs(deviation)
        allowed_deviation = [
            data.reference_load[t] - data.load_min_kWh if deviation[t] < 0
            else data.load_max_kWh - data.reference_load[t]
            for t in range(data.n_hours)
        ]
        binding_hours = sum(
            abs(deviation[t]) > 1e-6
            and abs(abs_deviation[t] - allowed_deviation[t]) <= 1e-6
            for t in range(data.n_hours)
        )
        coefficient = (
            f"cL={data.linear_disutility:.3f} DKK/kWh"
            if data.linear_disutility is not None
            else f"cQ={data.quadratic_disutility:.3f} DKK/kWh2"
        )
        print(
            f"disutility metrics ({coefficient}):\n"
            f"  procurement cost          : {results.procurement_cost:.2f} DKK\n"
            f"  total disutility          : {results.disutility:.2f} DKK\n"
            f"  daily energy consumed     : {load.sum():.2f} kWh\n"
            f"  total absolute deviation  : {abs_deviation.sum():.2f} kWh\n"
            f"  deviation-bound hours     : {binding_hours}\n"
        )
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_question(question)
    scenarios = {
        "base": base,
        "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
        "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
        "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
        "no_pv": scale_pv(base, factor=0.0),
    }
    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        results = FlexibleConsumerModel(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs


def run_linear_disutility_sweep(out: Path) -> None:
    """Run the selected Q2(b) cL cases, print metrics, and plot load schedules."""
    base = load_question("Q2_linear")
    if base.reference_load is None or base.linear_disutility is None:
        raise ValueError("Q2_linear must define a reference profile and linear disutility coefficient")

    hours = list(range(base.n_hours))
    rows: list[dict[str, float | int]] = []
    figure, axis = matplotlib.pyplot.subplots(figsize=(11, 6))
    axis.step(hours, base.reference_load, where="mid", color="black", linestyle="--",
              linewidth=2, label="reference load")

    for coefficient in LINEAR_DISUTILITY_VALUES:
        data = replace(base, linear_disutility=coefficient)
        results = FlexibleConsumerModel(data, name=f"q2_linear_cL_{coefficient:g}").build().solve()
        load = results.hourly["load"].to_numpy()
        deviation = load - data.reference_load
        absolute_deviation = abs(deviation)
        allowed_deviation = [
            data.reference_load[t] - data.load_min_kWh if deviation[t] < 0
            else data.load_max_kWh - data.reference_load[t]
            for t in hours
        ]
        binding_hours = sum(
            abs(deviation[t]) > 1e-6
            and abs(absolute_deviation[t] - allowed_deviation[t]) <= 1e-6
            for t in hours
        )
        rows.append({
            "cL_DKK_per_kWh": coefficient,
            "procurement_cost_DKK": results.procurement_cost,
            "total_disutility_DKK": coefficient * absolute_deviation.sum(),
            "daily_energy_kWh": load.sum(),
            "total_absolute_deviation_kWh": absolute_deviation.sum(),
            "deviation_bound_hours": binding_hours,
        })
        axis.step(hours, load, where="mid", linewidth=1.5,
                  label=f"$c^L$={coefficient:g}")

    axis.set(
        xlabel="Hour",
        ylabel="Load (kWh/h)",
        title="Load schedules for different linear disutility coefficients", # Q2(b) iv)
        xlim=(0, base.n_hours - 1),
    )
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize="small")
    figure.tight_layout()
    plot_path = out / "linear_disutility_sweep.png"
    figure.savefig(plot_path, dpi=160)
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.pyplot.show()
    matplotlib.pyplot.close(figure)

    csv_path = out / "linear_disutility_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nQ2(b) linear disutility sensitivity results")
    print(
        f"{'cL':>6} {'procurement':>14} {'disutility':>12} {'energy':>10} "
        f"{'abs deviation':>15} {'bound hours':>13}"
    )
    for row in rows:
        print(
            f"{row['cL_DKK_per_kWh']:6.2f} "
            f"{row['procurement_cost_DKK']:14.2f} "
            f"{row['total_disutility_DKK']:12.2f} "
            f"{row['daily_energy_kWh']:10.2f} "
            f"{row['total_absolute_deviation_kWh']:15.2f} "
            f"{row['deviation_bound_hours']:13d}"
        )
    print(f"\nSweep table: {csv_path}\nSweep plot:  {plot_path}")


def run_quadratic_disutility_sweep(out: Path) -> None:
    """Run the selected Q2(c) cQ cases, print metrics, and plot load schedules."""
    base = load_question("Q2_quadratic")
    if base.reference_load is None or base.quadratic_disutility is None:
        raise ValueError("Q2_quadratic must define a reference profile and quadratic disutility coefficient")

    hours = list(range(base.n_hours))
    rows: list[dict[str, float | int]] = []
    figure, axis = matplotlib.pyplot.subplots(figsize=(11, 6))
    axis.step(hours, base.reference_load, where="mid", color="black", linestyle="--",
              linewidth=2, label="reference load")

    for coefficient in QUADRATIC_DISUTILITY_VALUES:
        data = replace(base, quadratic_disutility=coefficient)
        results = FlexibleConsumerModel(data, name=f"q2_quadratic_cQ_{coefficient:g}").build().solve()
        load = results.hourly["load"].to_numpy()
        deviation = load - data.reference_load
        absolute_deviation = abs(deviation)
        allowed_deviation = [
            data.reference_load[t] - data.load_min_kWh if deviation[t] < 0
            else data.load_max_kWh - data.reference_load[t]
            for t in hours
        ]
        binding_hours = sum(
            abs(deviation[t]) > 1e-6
            and abs(absolute_deviation[t] - allowed_deviation[t]) <= 1e-6
            for t in hours
        )
        rows.append({
            "cQ_DKK_per_kWh2": coefficient,
            "procurement_cost_DKK": results.procurement_cost,
            "total_disutility_DKK": results.disutility,
            "daily_energy_kWh": load.sum(),
            "total_absolute_deviation_kWh": absolute_deviation.sum(),
            "deviation_bound_hours": binding_hours,
        })
        axis.step(hours, load, where="mid", linewidth=1.5,
                  label=f"$c^Q$={coefficient:g}")

    axis.set(
        xlabel="Hour",
        ylabel="Load (kWh/h)",
        title="Load schedules for different quadratic disutility coefficients",
        xlim=(0, base.n_hours - 1),
    )
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize="small")
    figure.tight_layout()
    plot_path = out / "quadratic_disutility_sweep.png"
    figure.savefig(plot_path, dpi=160)
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.pyplot.show()
    matplotlib.pyplot.close(figure)

    csv_path = out / "quadratic_disutility_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nQ2(c) quadratic disutility sensitivity results")
    print(
        f"{'cQ':>6} {'procurement':>14} {'disutility':>12} {'energy':>10} "
        f"{'abs deviation':>15} {'bound hours':>13}"
    )
    for row in rows:
        print(
            f"{row['cQ_DKK_per_kWh2']:6.2f} "
            f"{row['procurement_cost_DKK']:14.2f} "
            f"{row['total_disutility_DKK']:12.2f} "
            f"{row['daily_energy_kWh']:10.2f} "
            f"{row['total_absolute_deviation_kWh']:15.2f} "
            f"{row['deviation_bound_hours']:13d}"
        )
    print(f"\nSweep table: {csv_path}\nSweep plot:  {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    base = run_base_case(args.question, out, args.show)
    if args.question == "Q2_linear" and base is not None:
        run_linear_disutility_sweep(out)
    if args.question == "Q2_quadratic" and base is not None:
        run_quadratic_disutility_sweep(out)
    if args.scenarios and base is not None:
        run_scenarios(args.question, out)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
