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
import pandas as pd

from src.data_loader import load_question, list_questions
from src.model import FlexibleConsumerModel, LinearDisutilityModel, QuadraticDisutilityModel, MinEnergyModel, BatteryModel, Results
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule, plot_min_energy_comparison, plot_battery, plot_sweep, plot_load_profiles, plot_deviation_profiles, plot_battery_value, plot_battery_value_grid, plot_marginal_prices, plot_schedule_with_prices
from src.scenarios import scale_prices, scale_pv, set_tariffs, set_disutility, set_load_preferences, shift_prices, set_battery

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODELS = {
    "Q1_caseA": FlexibleConsumerModel,
    "Q1_caseB": FlexibleConsumerModel,
    "Q2_linear": LinearDisutilityModel,
    "Q2_quadratic": QuadraticDisutilityModel,
    "Q3": MinEnergyModel,
    "Q3_battery": BatteryModel,
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
    plot_schedule_with_prices(res_3, data, save_to=out / "schedule_price_q3.png", compare=res_2c)

def summarize(res: Results, data) -> dict:
    """Daily metrics of one run: mu^E (DKK/kWh), costs (DKK), energy and deviation (kWh), hour counts."""
    h, tol = res.hourly, 1e-6
    dev = h["load"] - h["reference_load"]
    return {
        "mu_E": -res.duals.get("min_energy", 0.0),
        "procurement": res.procurement_cost,
        "disutility": -res.utility,
        "total_cost": res.objective,
        "energy": h["load"].sum(),
        "abs_deviation": dev.abs().sum(),
        "hours_above_ref": int((dev > tol).sum()),
        "hours_below_ref": int((dev < -tol).sum()),
        "hours_at_Lmax": int((h["load"] > data.load_max_kWh - tol).sum()),
    }


def run_sweep(base, modify, values, name: str, out: Path, model_cls=MinEnergyModel):
    """Solve ``model_cls`` for each value; ``modify(base, v)`` returns the scenario data.

    Hourly results go to ``out/sensitivity/<name>/``, the summary table to
    ``out/sensitivity/sweep_<name>.csv``. Returns (summary DataFrame, {value: Results}).
    """
    folder = out / "sensitivity"
    rows, runs = [], {}
    for v in values:
        data = modify(base, v)
        res = model_cls(data).build().solve()
        res.save(folder / name, tag=f"{name}_{v}")
        runs[v] = res
        rows.append({name: v, **summarize(res, data)})
    df = pd.DataFrame(rows).set_index(name)
    df.to_csv(folder / f"sweep_{name}.csv")
    return df, runs


def run_q3_sensitivity(out: Path) -> None:
    """Question 3.(f): one-at-a-time sweeps of E_min, the price spread and c^Q around the Q3 base case."""
    base = load_question("Q3")
    sum_ref = base.reference_load.sum()
    sum_2c = QuadraticDisutilityModel(base).build().solve().hourly["load"].sum()

    sweeps = {
        # name: (modify, values, axis label, values shown in the load plot)
        "Emin": (lambda d, v: set_load_preferences(d, min_daily_energy_kWh=v),
                 sorted(set(range(0, 141, 5)) | {31, 32}), "$E^{min}$ [kWh]", [20, 30, 32, 40, 80]),
        "spread": (lambda d, v: scale_prices(d, v, keep_mean=True),
                   [0.0, 0.5, 1.0, 1.5, 2.0], "price spread factor", None),
        "cQ": (lambda d, v: set_disutility(d, quadratic=v),
               [0.1, 0.3, 1.0, 3.0, 10.0], "$c^Q$ [DKK/kWh$^2$]", None),
    }
    for name, (modify, values, label, shown) in sweeps.items():
        df, runs = run_sweep(base, modify, values, name, out)
        print(f"\n--- sweep over {name} ---\n{df.round(2).to_string()}")
        vlines = {"Σ load 2.(c)": sum_2c, "Σ reference": sum_ref} if name == "Emin" else None
        plot_sweep(df, label, save_to=out / "sensitivity" / f"sweep_{name}.png", log_x=(name == "cQ"), vlines=vlines)
        plot_load_profiles(runs, base, label, values=shown, save_to=out / "sensitivity" / f"loads_{name}.png")
        plot_deviation_profiles(runs, label, values=shown, save_to=out / "sensitivity" / f"deviation_{name}.png")

def run_battery_comparison(out: Path) -> None:
    """Question 3.(g).iv: Q3 consumer with and without the battery on the same data."""
    data = load_question("Q3_battery")
    res_bat = BatteryModel(data).build().solve()
    res_nobat = MinEnergyModel(data).build().solve()     # same data, ignores the battery
    res_bat.save(out, tag="with_battery")
    res_nobat.save(out, tag="without_battery")

    print(f"Max overlap charge/discharge : {res_bat.meta['max_overlap_kWh']:.2e} kWh")
    print(f"Battery value                : {res_nobat.objective - res_bat.objective:.2f} DKK/day")
    print(f"  procurement change         : {res_bat.procurement_cost - res_nobat.procurement_cost:+.2f} DKK")
    print(f"  disutility change          : {res_nobat.utility - res_bat.utility:+.2f} DKK")
    print(f"mu_E with / without battery  : {-res_bat.duals['min_energy']:.3f} / "
          f"{-res_nobat.duals['min_energy']:.3f} DKK/kWh")
    plot_battery(res_bat, data, save_to=out / "battery.png")

def run_battery_sensitivity(out: Path) -> None:
    """Question 3.(g).v: value of the battery (DKK/day) for one-at-a-time sweeps of the price
    spread (environment) and of the battery capacity and efficiency (technology).

    Value = total cost without battery - total cost with battery, on the same scenario data.
    """
    base = load_question("Q3_battery")
    folder = out / "sensitivity"
    sweeps = {
        #"spread": (lambda d, v: scale_prices(d, v, keep_mean=True), [0.0, 0.5, 1.0, 1.5, 2.0], "price spread factor"),
        #"level": (lambda d, v: shift_prices(d, v), [-0.4, -0.2, 0.0, 0.2, 0.4], "price level shift [DKK/kWh]"),
        "capacity": (lambda d, v: set_battery(d, capacity_kWh=v),
                [1.0, 2.0, 4.0, 8.0, 16.0], "battery capacity [kWh]"),
        #"efficiency": (lambda d, v: set_battery(d, efficiency=v), [0.80, 0.85, 0.90, 0.95, 1.00], "charging = discharging efficiency [-]"),
        "power": (lambda d, v: set_battery(d, max_power_kW=v), [0.5, 1, 1.5, 2, 2.5], "power limit [kW]"),
        "capacity_power": (lambda d, v: set_battery(d, capacity_kWh=v, max_power_kW=0.375 * v), [1.0, 2.0, 4.0, 8.0, 16.0], "capacity [kWh] (power = 0.375 x capacity)"),
    }
    dfs, labels = {}, {}
    for name, (modify, values, label) in sweeps.items():
        rows = []
        for v in values:
            data = modify(base, v)
            res_bat = BatteryModel(data).build().solve()
            res_no = MinEnergyModel(data).build().solve()      # same data, ignores the battery
            res_bat.save(folder / name, tag=f"{name}_{v}")
            rows.append({
                name: v,
                "value": res_no.objective - res_bat.objective,
                "procurement_change": res_bat.procurement_cost - res_no.procurement_cost,
                "disutility_change": res_no.utility - res_bat.utility,
                "throughput": res_bat.hourly["discharge"].sum(),
                "cycles": res_bat.hourly["discharge"].sum() / data.battery_capacity_kWh,
                "mu_E_with": -res_bat.duals["min_energy"],
                "mu_E_without": -res_no.duals["min_energy"],
                "max_overlap": res_bat.meta["max_overlap_kWh"],
            })
        df = pd.DataFrame(rows).set_index(name)
        df.to_csv(folder / f"battery_sweep_{name}.csv")
        print(f"\n--- battery value, sweep over {name} ---\n{df.round(3).to_string()}")
        dfs[name], labels[name] = df, label
    plot_battery_value_grid(dfs, labels, save_to=folder / f"battery_value_all.png")

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
    if args.question == "Q3_battery":
        run_battery_comparison(out)
    if args.scenarios and base is not None:
        if args.question == "Q3":
            run_q3_sensitivity(out)
        elif args.question == "Q3_battery":
            run_battery_sensitivity(out)
        else:
            run_scenarios(args.question, out)
    print(f"\nOutputs written to {out}")




if __name__ == "__main__":
    main()
