"""Matplotlib figures for the input data and the optimisation results.

Every function returns the ``Figure`` and optionally saves it, so the same code works in a
script (``python main.py``) and in a notebook (``plot_schedule(results, data);``).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import InputData
from .model import Results


def _finish(fig: plt.Figure, save_to: Path | str | None) -> plt.Figure:
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def plot_inputs(data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly prices (with tariffs) and available PV / load preferences, side by side."""
    h = data.hours
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax1.step(h, data.energy_price, where="mid", label="energy price", color="k")
    ax1.step(h, data.energy_price + data.import_tariff, where="mid", ls="--", label="price + import tariff")
    ax1.step(h, data.energy_price - data.export_tariff, where="mid", ls=":", label="price - export tariff")
    ax1.set(xlabel="hour", ylabel="DKK/kWh", title="Electricity prices")
    ax1.legend(fontsize=8)

    ax2.fill_between(h, data.pv_available, step="mid", alpha=0.4, color="orange", label="PV available")
    ax2.axhline(data.load_max_kWh, color="C3", ls="--", label="max load")
    if data.load_min_kWh > 0:
        ax2.axhline(data.load_min_kWh, color="C3", ls=":", label="min load")
    if data.reference_load is not None:
        ax2.step(h, data.reference_load, where="mid", color="C0", label="reference load")
    ax2.set(xlabel="hour", ylabel="kWh/h", title="PV and load preferences")
    ax2.legend(fontsize=8)
    fig.suptitle(f"Input data - {data.question}", fontsize=11)
    return _finish(fig, save_to)


def plot_schedule(results: Results, data: InputData, save_to: Path | str | None = None, 
                  compare: Results | None = None, compare_label: str = "load 2.c") -> plt.Figure:
    """Optimal schedule: load, PV used, import/export, with prices on a second axis."""
    hr = results.hourly
    h = hr.index.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.2))

    width = 0.8
    if "load" in hr:
        ax.bar(h, hr["load"], width, color="C0", alpha=0.7, label="load")
    if "pv" in hr:
        ax.bar(h, -hr["pv"], width, color="orange", alpha=0.7, label="PV used (negative = generation)")
    if "pv_available" in hr and "pv" in hr:
        ax.step(h, -hr["pv_available"], where="mid", color="orange", ls="--", lw=1, label="PV available")
    if "import" in hr and "export" in hr:
        ax.plot(h, hr["import"] - hr["export"], "k.-", label="net import (+) / export (-)")
    if compare is not None:
        ax.step(h, compare.hourly["load"], where="mid", color="green", ls="--", lw=1.5,
                label=f"{compare_label}")
    if "reference_load" in hr:
        ax.step(h, hr["reference_load"], where="mid", color="C0", ls=":", label="reference load")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="hour", ylabel="kWh/h", title=f"Optimal schedule - {results.question} (cost {results.objective:.1f} DKK)")

    ax2 = ax.twinx()
    ax2.step(h, hr["price"], where="mid", color="C3", lw=1.2, label="energy price")
    ax2.set_ylabel("DKK/kWh", color="C3")

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(fig, save_to)


def plot_duals(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly dual variables (all ``dual_*`` columns) against the price signals."""
    hr = results.hourly
    dual_cols = [c for c in hr.columns if c.startswith("dual_")]
    fig, ax = plt.subplots(figsize=(11, 4))
    h = hr.index.to_numpy()
    for c in dual_cols:
        ax.step(h, hr[c], where="mid", label=c.removeprefix("dual_"))
    ax.step(h, data.energy_price + data.import_tariff, where="mid", color="grey", ls="--", lw=1, label="price + import tariff")
    ax.step(h, data.energy_price - data.export_tariff, where="mid", color="grey", ls=":", lw=1, label="price - export tariff")
    ax.set(xlabel="hour", ylabel="DKK/kWh", title=f"Dual variables - {results.question}")
    ax.legend(fontsize=8, ncol=3)
    return _finish(fig, save_to)


def plot_scenario_comparison(
    runs: dict[str, Results], metric: str = "objective", save_to: Path | str | None = None
) -> plt.Figure:
    """Bar chart of one metric across scenarios. ``metric`` is ``"objective"`` or the name of an
    hourly column whose daily sum is compared (e.g. ``"import"``, ``"export"``, ``"load"``)."""
    names = list(runs)
    if metric == "objective":
        values = [r.objective for r in runs.values()]
        ylabel = "daily cost [DKK]"
    else:
        values = [r.hourly[metric].sum() for r in runs.values()]
        ylabel = f"daily {metric} [kWh]"
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(names)), 3.8))
    ax.bar(names, values, color="C0")
    ax.set(ylabel=ylabel, title=f"Scenario comparison - {metric}")
    ax.tick_params(axis="x", rotation=20)
    return _finish(fig, save_to)

def plot_min_energy_comparison(res_3: Results, res_2c: Results, data: InputData, save_to: Path | str | None = None) -> plt.figure:
    h3, h2 = res_3.hourly, res_2c.hourly
    h = h3.index.to_numpy()
    ref = h3["reference_load"]

    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios":[3,2]})
    # --- top: loads
    ax1.bar(h, h3["load"], 0.8, color="C0", alpha=0.7,
            label=f"load, Q3 with $E^{{min}}$ (Σ = {h3['load'].sum():.1f} kWh)")
    ax1.step(h, h2["load"], where="mid", color="k", ls="--", lw=1.5,
             label=f"load, 2.(c) unconstrained (Σ = {h2['load'].sum():.1f} kWh)")
    ax1.step(h, ref, where="mid", color="C1", ls=":", lw=2,
             label=f"reference (Σ = {ref.sum():.1f} kWh)")
    ax1.set(ylabel="kWh/h", title="Optimal load with and without the minimum daily energy requirement")
    ax1.legend(fontsize=8, loc="upper left")

    # --- bottom: deviations and marginal prices
    w = 0.4
    ax2.bar(h - w / 2, h2["load"] - ref, w, color="grey", label="deviation 2.(c)")
    ax2.bar(h + w / 2, h3["load"] - ref, w, color="C0", label="deviation Q3")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set(xlabel="hour", ylabel="$\\ell_t - \\ell_t^{ref}$ [kWh/h]")

    ax3 = ax2.twinx()
    lam = -h3["dual_balance"]                      # check sign: should equal P_imp in importing hours
    mu_E = -res_3.duals["min_energy"]              # Gurobi: Pi <= 0 for a <= constraint in a min problem
    ax3.step(h, lam, where="mid", color="C3", lw=1.2, label="$\\lambda_t$ (Q3)")
    ax3.axhline(mu_E, color="C3", ls="--", lw=1.2, label=f"$\\mu^E$ = {mu_E:.2f}")
    ax3.set_ylabel("DKK/kWh", color="C3")

    l1, b1 = ax2.get_legend_handles_labels()
    l2, b2 = ax3.get_legend_handles_labels()
    ax2.legend(l1 + l2, b1 + b2, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    return _finish(fig, save_to)

def plot_marginal_prices(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Question 3.(e): marginal value of energy lambda_t per hour against the multiplier mu^E of the
    minimum-energy requirement (DKK/kWh), with the effective import/export prices and C^PV for reference.

    Gurobi reports d(objective)/d(rhs), so lambda_t = -Pi of the balance and mu^E = -Pi of min_energy.
    Hours with lambda_t < mu^E are shaded: there the load lies above the reference.
    """
    hr = results.hourly
    h = hr.index.to_numpy()
    lam = -hr["dual_balance"].to_numpy()
    mu_E = -results.duals["min_energy"]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.step(h, data.energy_price + data.import_tariff, where="mid", color="grey", ls="--", lw=1,
            label="effective import price $P_t^{imp}$")
    ax.step(h, data.energy_price - data.export_tariff, where="mid", color="grey", ls=":", lw=1,
            label="effective export price $P_t^{exp}$")
    #ax.axhline(data.pv_marginal_cost, color="orange", ls=":", lw=1.2, label="$C^{PV}$")
    ax.step(h, lam, where="mid", color="C0", lw=2, label="$\\lambda_t$")
    ax.axhline(mu_E, color="C3", ls="--", lw=1.5, label=f"$\\mu^E$ = {mu_E:.2f}")
    #ax.fill_between(h, lam, mu_E, where=lam < mu_E, step="mid", color="C2", alpha=0.2,label="$\\lambda_t < \\mu^E$ (load above reference)")
    ax.set(xlabel="hour", ylabel="DKK/kWh",
           title=f"Marginal value of energy and of the minimum-energy requirement - {results.question}")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(fig, save_to)

def plot_schedule_with_prices(results: Results, data: InputData, save_to: Path | str | None = None,
                              compare: Results | None = None, compare_label: str = "load 2.c") -> plt.Figure:
    """Question 3.(e): optimal schedule (top) and lambda_t with mu^E (bottom) in one figure with a
    shared hour axis. Same content as plot_schedule and plot_marginal_prices.
    Gurobi reports d(objective)/d(rhs), so lambda_t = -Pi of the balance and mu^E = -Pi of min_energy."""
    hr = results.hourly
    h = hr.index.to_numpy()
    fig, (ax, ax3) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2]})

    # --- top: schedule (as plot_schedule)
    width = 0.8
    ax.bar(h, hr["load"], width, color="C0", alpha=0.7, label="load")
    ax.bar(h, -hr["pv"], width, color="orange", alpha=0.7, label="PV used (negative = generation)")
    ax.step(h, -hr["pv_available"], where="mid", color="orange", ls="--", lw=1, label="PV available")
    ax.plot(h, hr["import"] - hr["export"], "k.-", label="net import (+) / export (-)")
    if compare is not None:
        ax.step(h, compare.hourly["load"], where="mid", color="green", ls="--", lw=1.5, label=compare_label)
    ax.step(h, hr["reference_load"], where="mid", color="C0", ls=":", label="reference load")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(ylabel="kWh/h", title=f"Optimal schedule - {results.question} (cost {results.objective:.1f} DKK)")

    ax2 = ax.twinx()
    ax2.step(h, hr["price"], where="mid", color="C3", lw=1.2, label="energy price")
    ax2.set_ylabel("DKK/kWh", color="C3")
    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.06))

    # --- bottom: marginal prices (as plot_marginal_prices)
    lam = -hr["dual_balance"].to_numpy()
    mu_E = -results.duals["min_energy"]
    ax3.step(h, data.energy_price + data.import_tariff, where="mid", color="grey", ls="--", lw=1,
             label="effective import price $P_t^{imp}$")
    ax3.step(h, data.energy_price - data.export_tariff, where="mid", color="grey", ls=":", lw=1,
             label="effective export price $P_t^{exp}$")
    ax3.step(h, lam, where="mid", color="C0", lw=2, label="$\\lambda_t$")
    ax3.axhline(mu_E, color="C3", ls="--", lw=1.5, label=f"$\\mu^E$ = {mu_E:.2f}")
    ax3.set(xlabel="hour", ylabel="DKK/kWh",
            title=f"Marginal value of energy and of the minimum-energy requirement - {results.question}")
    ax3.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(fig, save_to)

def plot_sweep(df, xlabel: str, save_to: Path | str | None = None, log_x: bool = False,
               vlines: dict[str, float] | None = None) -> plt.Figure:
    """Daily metrics against a swept parameter (one row of ``df`` per value).

    Panels: mu^E (DKK/kWh); procurement cost, disutility and total cost (DKK/day);
    number of hours above / below the reference and at L^max. ``vlines`` marks
    reference values of the parameter (e.g. sum of the reference profile).
    """
    x = df.index.to_numpy(dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(x, df["mu_E"], "o-", color="C3")
    axes[0].set(ylabel="$\\mu^E$ [DKK/kWh]")

    for col, c in [("procurement", "C0"), ("disutility", "C1"), ("total_cost", "k")]:
        axes[1].plot(x, df[col], "o-", color=c, label=col.replace("_", " "))
    axes[1].set(ylabel="DKK/day")
    axes[1].legend(fontsize=8)

    for col, c in [("hours_above_ref", "C2"), ("hours_below_ref", "C4"), ("hours_at_Lmax", "C3")]:
        axes[2].plot(x, df[col], "o-", color=c, label=col.replace("_", " "))
    axes[2].set(xlabel=xlabel, ylabel="number of hours")
    axes[2].legend(fontsize=8)

    if log_x:
        axes[2].set_xscale("log")
    for label, v in (vlines or {}).items():
        for ax in axes:
            ax.axvline(v, color="grey", ls=":", lw=1)
        axes[0].annotate(label, (v, 1), xycoords=("data", "axes fraction"), rotation=90,
                         va="top", ha="right", fontsize=8, color="grey")
    return _finish(fig, save_to)


def plot_load_profiles(runs: dict, data: InputData, label: str, values: list | None = None,
                       save_to: Path | str | None = None) -> plt.Figure:
    """Hourly load (kWh/h) for selected values of a swept parameter, against the reference profile."""
    values = list(runs) if values is None else values
    h = data.hours
    fig, ax = plt.subplots(figsize=(11, 4))
    cmap = plt.get_cmap("viridis")
    for i, v in enumerate(values):
        ax.step(h, runs[v].hourly["load"], where="mid", lw=1.5,
                color=cmap(i / max(len(values) - 1, 1)), label=f"{label} = {v:g}")
    ax.step(h, data.reference_load, where="mid", color="k", ls=":", lw=2, label="reference")
    ax.set(xlabel="hour", ylabel="kWh/h", title=f"Optimal load for different values of {label}")
    ax.legend(fontsize=8, ncol=3)
    return _finish(fig, save_to)

def plot_battery(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Battery operation (Question 3.(g)).

    Top: charging (positive) and discharging (negative) per hour (kWh/h), with the effective
    import and export prices (DKK/kWh) on a second axis.
    Bottom: state of charge at the hour boundaries (kWh), from S^0 at the start of the day
    to the end of hour 23, with the capacity and S^0 as reference lines.
    """
    hr = results.hourly
    h = hr.index.to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})

    # --- top: charging / discharging and effective prices
    ax1.bar(h, hr["charge"], 0.8, color="C2", alpha=0.8, label="charging")
    ax1.bar(h, -hr["discharge"], 0.8, color="C4", alpha=0.8, label="discharging (negative)")
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set(ylabel="kWh/h", title=f"Battery operation - {results.question}")

    ax1b = ax1.twinx()
    ax1b.step(h, data.energy_price + data.import_tariff, where="mid", color="C3", lw=1.2,
              label="effective import price $P_t^{imp}$")
    ax1b.step(h, data.energy_price - data.export_tariff, where="mid", color="C3", ls="--", lw=1.2,
              label="effective export price $P_t^{exp}$")
    ax1b.set_ylabel("DKK/kWh", color="C3")

    l1, b1 = ax1.get_legend_handles_labels()
    l2, b2 = ax1b.get_legend_handles_labels()
    ax1.legend(l1 + l2, b1 + b2, fontsize=8, ncol=2, loc="upper left")

    # --- bottom: state of charge at the hour boundaries
    # soc[t] is the state at the END of hour t; bars are centred on t, so that point sits at t + 0.5
    x_soc = np.arange(len(h) + 1) - 0.5
    soc = np.concatenate([[data.battery_initial_soc_kWh], hr["soc"].to_numpy()])
    ax2.plot(x_soc, soc, "k.-", label="state of charge")
    ax2.axhline(data.battery_capacity_kWh, color="C3", ls="--", lw=1, label="capacity")
    ax2.axhline(data.battery_initial_soc_kWh, color="grey", ls=":", lw=1, label="initial = final SoC")
    ax2.set(xlabel="hour", ylabel="kWh", ylim=(-0.2, 1.15 * data.battery_capacity_kWh))
    ax2.legend(fontsize=8, ncol=3, loc="upper left")
    return _finish(fig, save_to)

def plot_deviation_profiles(runs: dict, label: str, values: list | None = None,
                            save_to: Path | str | None = None) -> plt.Figure:
    """Deviation from the reference (kWh/h, top) and energy price (DKK/kWh, bottom) for
    selected values of a swept parameter. Uses the 'price' column of each run, so it also
    shows the scenario prices when the prices themselves are varied."""
    values = list(runs) if values is None else values
    first = runs[values[0]].hourly
    h = first.index.to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    cmap = plt.get_cmap("viridis")
    for i, v in enumerate(values):
        hr = runs[v].hourly
        c = cmap(i / max(len(values) - 1, 1))
        ax1.step(h, hr["load"] - hr["reference_load"], where="mid", color=c, lw=1.5, label=f"{label} = {v:g}")
        ax2.step(h, hr["price"], where="mid", color=c, lw=1.2)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set(ylabel="$l_t^L - l^{ref}$ [kWh/h]",
            title=f"Deviation from the reference for different values of {label}")
    ax1.legend(fontsize=8, ncol=3)
    ax2.set(xlabel="hour", ylabel="energy price $p_t$ [DKK/kWh]")
    return _finish(fig, save_to)

def plot_battery_value(df, xlabel: str, save_to: Path | str | None = None) -> plt.Figure:
    """Value of the battery (DKK/day) against a swept parameter, with the discharged energy
    (kWh/day) on a second axis."""
    x = df.index.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, df["value"], "o-", color="C0", label="battery value")
    ax.set(xlabel=xlabel, ylabel="value of the battery [DKK/day]")
    ax2 = ax.twinx()
    ax2.plot(x, df["throughput"], "s--", color="C1", label="discharged energy")
    ax2.set_ylabel("discharged energy [kWh/day]", color="C1")
    l1, b1 = ax.get_legend_handles_labels()
    l2, b2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, b1 + b2, fontsize=8, loc="upper left")
    return _finish(fig, save_to)

def plot_battery_value_grid(dfs: dict, labels: dict, save_to: Path | str | None = None) -> plt.Figure:
    """Value of the battery (DKK/day) for several one-at-a-time sweeps, one panel per parameter,
    with a shared y-axis. ``dfs`` maps sweep name -> DataFrame (index = parameter value, column
    'value'), ``labels`` maps sweep name -> x-axis label. The base case is marked in each panel."""
    names = list(dfs)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        df = dfs[name]
        ax.plot(df.index.to_numpy(dtype=float), df["value"], "o-", color="C0")
        ax.set(xlabel=labels[name])
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("value of the battery [DKK/day]")
    return _finish(fig, save_to)