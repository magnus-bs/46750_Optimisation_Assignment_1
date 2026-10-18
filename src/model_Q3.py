"""Question 3: minimum daily energy requirement (MinEnergyModel) and small battery (BatteryModel).

Self-contained models, independent of FlexibleConsumerModel in src/model.py, but still build from that; only the Results
container and the helpers _dual / _status_name are shared, so the plotting functions work for both.

    model = MinEnergyModel(data).build()   # data = load_question("Q3")
    results = model.solve()

Objective = procurement cost + quadratic disutility (DKK). results.utility is 0 and
results.disutility holds the disutility, following the convention of src/model.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import InputData
from .model import Results, _dual, _status_name

class MinEnergyModel:
    """Question 3: the quadratic-disutility consumer of 2.(c) with a minimum daily energy
    requirement (eqs. 12a-12i). The daily requirement (12i) couples all hours."""

    def __init__(self, data: InputData, name: str = "min_energy", verbose: bool = False):
        if data.reference_load is None or data.quadratic_disutility is None:
            raise ValueError("MinEnergyModel needs reference_load and quadratic_disutility (use a Q3 case).")
        if data.min_daily_energy_kWh is None:
            raise ValueError("MinEnergyModel needs min_daily_energy_kWh (use a Q3 case).")
        if data.min_daily_energy_kWh > data.n_hours * data.load_max_kWh:
            raise ValueError(
                f"Infeasible: E_min = {data.min_daily_energy_kWh} kWh exceeds the maximum daily "
                f"consumption {data.n_hours * data.load_max_kWh} kWh (24 * L_max)."
            )
        self.data = data
        self.T = range(data.n_hours)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.var: dict[str, gp.tupledict] = {}   # decision variables by name
        self.con: dict[str, gp.tupledict | gp.Constr] = {}   # constraints by name (duals read from here)

    # ------------------------------------------------------------------ build
    def build(self) -> "MinEnergyModel":
        """Declare variables and constraints, then set objective = procurement + disutility."""
        self._add_variables()
        self._add_constraints()
        self._procurement = self._procurement_expr()   # DKK
        self._disutility = self._disutility_expr()     # DKK
        self.m.setObjective(self._procurement + self._disutility, GRB.MINIMIZE)
        self.m.update()
        return self

    def _add_variables(self) -> None:
        """Hourly load, PV production, import and export (kWh/h). Free; bounds are explicit constraints."""
        m, T = self.m, self.T
        self.var["load"] = m.addVars(T, lb=-GRB.INFINITY, name="load")      # l_t^L
        self.var["pv"] = m.addVars(T, lb=-GRB.INFINITY, name="pv")          # g_t^PV
        self.var["import"] = m.addVars(T, lb=-GRB.INFINITY, name="import")  # x_t^imp
        self.var["export"] = m.addVars(T, lb=-GRB.INFINITY, name="export")  # x_t^exp

    def _balance_lhs(self, t: int) -> gp.LinExpr:
        """Net energy demand in hour t (kWh/h); the power balance sets it to 0.
        Subclasses with storage add the charging/discharging terms."""
        L, PV = self.var["load"], self.var["pv"]
        IMP, EXP = self.var["import"], self.var["export"]
        return L[t] + EXP[t] - PV[t] - IMP[t]

    def _add_constraints(self) -> None:
        """Power balance (12b), load and PV bounds (12c-12f), non-negativity (12g-12h), E_min (12i)."""
        d, m, T = self.data, self.m, self.T
        L, PV = self.var["load"], self.var["pv"]
        IMP, EXP = self.var["import"], self.var["export"]
        self.con["balance"] = m.addConstrs((self._balance_lhs(t) == 0 for t in T), name="balance")
        self.con["load_min"] = m.addConstrs((d.load_min_kWh - L[t] <= 0 for t in T), name="load_min")
        self.con["load_max"] = m.addConstrs((L[t] - d.load_max_kWh <= 0 for t in T), name="load_max")
        self.con["pv_min"] = m.addConstrs((-PV[t] <= 0 for t in T), name="pv_min")
        self.con["pv_max"] = m.addConstrs((PV[t] - d.pv_available[t] <= 0 for t in T), name="pv_max")
        self.con["import_nonneg"] = m.addConstrs((-IMP[t] <= 0 for t in T), name="import_nonneg")
        self.con["export_nonneg"] = m.addConstrs((-EXP[t] <= 0 for t in T), name="export_nonneg")
        self.con["min_energy"] = m.addConstr(
            d.min_daily_energy_kWh - gp.quicksum(L[t] for t in T) <= 0, name="min_energy")

    def _procurement_expr(self) -> gp.LinExpr:
        """Daily procurement cost: PV production cost + import cost - export revenue (DKK)."""
        d, T = self.data, self.T
        PV, IMP, EXP = self.var["pv"], self.var["import"], self.var["export"]
        p_imp = d.energy_price + d.import_tariff   # effective import price, DKK/kWh
        p_exp = d.energy_price - d.export_tariff   # effective export price, DKK/kWh
        return gp.quicksum(
            d.pv_marginal_cost * PV[t] + float(p_imp[t]) * IMP[t] - float(p_exp[t]) * EXP[t] for t in T)

    def _disutility_expr(self) -> gp.QuadExpr:
        """Daily disutility sum_t c^Q (l_t - l_t^ref)^2 (DKK)."""
        d, L = self.data, self.var["load"]
        return gp.quicksum(
            d.quadratic_disutility * (L[t] - float(d.reference_load[t])) * (L[t] - float(d.reference_load[t]))
            for t in self.T)

    # ------------------------------------------------------------------ solve
    def solve(self) -> Results:
        """Optimise and return primal values, objective and dual values."""
        self.m.update()
        self.m.optimize()
        status = _status_name(self.m.Status)
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimisation ended with status {status}.")
        return self._extract_results(status)

    def _extract_results(self, status: str) -> Results:
        """Hourly primal values, deviation from the reference, hourly and scalar duals, cost split."""
        d, T = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = d.energy_price
        hourly["pv_available"] = d.pv_available
        hourly["reference_load"] = d.reference_load
        for name, v in self.var.items():
            hourly[name] = [v[t].X for t in T]
        hourly["deviation"] = hourly["load"] - hourly["reference_load"]
        hourly["abs_deviation"] = hourly["deviation"].abs()

        duals: dict[str, float] = {}
        for name, c in self.con.items():
            if isinstance(c, gp.tupledict):
                hourly[f"dual_{name}"] = [_dual(c[t]) for t in T]
            else:
                duals[name] = _dual(c)

        procurement = self._procurement.getValue()
        disutility = self._disutility.getValue()
        assert abs(procurement + disutility - self.m.ObjVal) < 1e-6 * max(1.0, abs(self.m.ObjVal)), \
            "procurement + disutility should equal the objective value"

        return Results(
            question=d.question, status=status, objective=self.m.ObjVal,
            procurement_cost=procurement, utility=0.0, disutility=disutility,
            hourly=hourly, duals=duals, meta={},
        )