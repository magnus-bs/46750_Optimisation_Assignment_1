"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

from src.data_loader import load_question, list_questions
from src.model import FlexibleConsumerModel, LinearDisutilityModel, QuadraticDisutilityModel, MinEnergyModel, Results
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule, plot_min_energy_comparison
from src.scenarios import scale_prices, scale_pv, set_tariffs, set_disutility, set_load_preferences

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODELS = {
    "Q1_caseA": FlexibleConsumerModel,
    "Q1_caseB": FlexibleConsumerModel,
    "Q2_linear": LinearDisutilityModel,
    "Q2_quadratic": QuadraticDisutilityModel,
    "Q3": MinEnergyModel,
}

def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    model = MODELS[question](data).build()
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_question(question)
    if question == "Q3":
        scenarios={"base": base}
        for E in [20, 25, 30, 35, 40, 60, 80, 100, 120, 140]:
            scenarios[f"Emin_{E}"] = set_load_preferences(base, min_daily_energy_kWh=E)
        for f in [0.0, 0.5, 1.5, 2.0]:
            scenarios[f"spread_{f}"] = scale_prices(base, factor=f, keep_mean=True)
        for cQ in [0.1, 0.3, 3.0, 10.0]:
            scenarios[f"cQ_{cQ}"] = set_disutility(base, quadratic=cQ)
    else:
        scenarios = {
            "base": base,
            "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
            "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
            "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
            "no_pv": scale_pv(base, factor=0.0),
        }
    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        results = MODELS[question](data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh | procurement {results.procurement_cost:7.2f}" 
              f" | disutility {-results.utility:6.2f} | load {results.hourly['load'].sum():6.1f} kWh"
              f" | mu_E {-results.duals.get('min_energy', 0.0):5.2f}"
              )
        
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs

def run_q3_comparison(out: Path) -> None:
    """Question 3.(e): Q3 base case vs the 2.(c) consumer on the same data."""
    data = load_question("Q3")
    res_3 = MinEnergyModel(data).build().solve()
    res_2c = QuadraticDisutilityModel(data).build().solve()   # ignores E_min
    res_3.save(out, tag="with_Emin")
    res_2c.save(out, tag="without_Emin")
    plot_schedule(res_3, data, save_to=out / "schedule_q3_vs_2c.png", compare=res_2c)

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
    if args.question == "Q3":
        run_q3_comparison(out)
    if args.scenarios and base is not None:
        run_scenarios(args.question, out)
    print(f"\nOutputs written to {out}")




if __name__ == "__main__":
    main()
