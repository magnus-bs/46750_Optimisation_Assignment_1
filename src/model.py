"""Optimisation model of a single flexible consumer, implemented with gurobipy.

The class below separates the three steps you will repeat for every question:

    model = FlexibleConsumerModel(data)   # 1. hand over the input data
    model.build()                         # 2. declare variables, objective, constraints
    results = model.solve()               # 3. optimise and collect primal AND dual values

``build()`` is the only method you need to complete for Question 1; the other questions
are variations of it (a different objective, an extra constraint). Copy this file or
subclass ``FlexibleConsumerModel`` and override ``build()`` to keep one model per question.

Two conventions make the dual variables easy to read out afterwards:

* Every constraint family is stored in ``self.con`` under a descriptive name, e.g.
  ``self.con["balance"] = self.m.addConstrs(...)``. ``solve()`` then returns the dual value
  (shadow price, Gurobi attribute ``Pi``) of every constraint in ``self.con`` automatically.
* Bounds that you want a dual for must be written as explicit constraints (``addConstr``),
  not as variable bounds (``lb=``/``ub=``). Gurobi reports the sensitivity of a variable
  bound in the reduced cost (``RC``), not in ``Pi``.
* Duals of quadratic constraints (``m.addQConstr``) are read from ``QCPi`` and require ``QCPDual = 1`` (set below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import InputData


@dataclass
class Results:
    """Primal and dual solution of one model run."""

    question: str
    status: str
    objective: float
    procurement_cost: float             
    utility: float   
    disutility: float
    hourly: pd.DataFrame                   # one row per hour: variables, prices, hourly duals
    duals: dict[str, float] = field(default_factory=dict)   # duals of non-hourly constraints
    meta: dict = field(default_factory=dict)                 # anything else worth keeping (scenario name, ...)

    def save(self, folder: Path | str, tag: str = "") -> None:
        """Write ``hourly`` to CSV and the scalar values to a small text file."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        self.hourly.to_csv(folder / f"{stem}_hourly.csv", index_label="hour")
        with open(folder / f"{stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"status    : {self.status}\nobjective : {self.objective:.4f} DKK\n")
            for k, v in self.duals.items():
                f.write(f"dual[{k}] : {v:.4f}\n")

    def __str__(self) -> str:
        cols = [c for c in self.hourly.columns if not c.startswith("dual_")]
        return (
            f"status: {self.status} | objective: {self.objective:.2f} DKK\n"
            f"procurement cost: {self.procurement_cost:.2f} DKK\n"
            f"utility: {self.utility:.2f} DKK | disutility: {self.disutility:.2f} DKK\n"
            f"daily totals (kWh): " + ", ".join(f"{c}={self.hourly[c].sum():.1f}" for c in cols if c in ("import", "export", "load", "pv"))
            + (f"\nduals: {self.duals}" if self.duals else "")
        )


def _dual(c) -> float:
    """Dual value of a linear (``Pi``) or quadratic (``QCPi``, requires QCPDual=1) constraint."""
    return c.QCPi if isinstance(c, gp.QConstr) else c.Pi


class FlexibleConsumerModel:
    """Consumption problem of one consumer over a 24-hour horizon (Question 1); extend it for Questions 2 and 3."""

    def __init__(self, data: InputData, name: str = "flexible_consumer", verbose: bool = False):
        self.data = data
        self.T = range(data.n_hours)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.m.Params.QCPDual = 1          # only relevant if you add a quadratic constraint (none is needed in Assignment 1)
        self.var: dict[str, gp.tupledict | gp.Var] = {}   # decision variables by name
        self.con: dict[str, gp.tupledict | gp.Constr] = {}  # constraints by name (duals read from here)

    # ------------------------------------------------------------------ 2. build
    def build(self) -> "FlexibleConsumerModel":
        """Declare decision variables, objective and constraints.

        TODO (Question 1): complete this method with the variables, objective and constraints
        of the problem you formulated in Question 1. Keep the naming pattern below so
        that ``solve()`` can return the primal and dual values automatically.
        """
        d, m, T = self.data, self.m, self.T

        # --- Decision variables --------------------------------------------------------
        # TODO: identify and declare the decision variables of your formulation.
        # Store every variable family in self.var["<name>"]: solve() then returns its hourly
        # values automatically as a column of results.hourly.
        # Pattern for hourly variables (one per hour):
        #   self.var["<name>"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="<name>")
        # Pattern for a single (daily) variable:
        #   self.var["<name>"] = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="<name>")
        # Notes:
        # * gurobipy indexes the names automatically: name="<name>" in addVars(T, ...) creates
        #   <name>[0], <name>[1], ..., <name>[23] - no need to build per-hour names yourself.
        # * vtype: the same format takes GRB.BINARY or GRB.INTEGER if you ever need them (the
        #   problem then becomes a MILP and dual values are no longer defined; solve() skips them).
        # * lb defaults to 0 in gurobipy: a free variable needs an explicit lb=-GRB.INFINITY, and
        #   a bound you want a dual for must be an explicit constraint, not lb=/ub= (see the README).
        # * naming the families "import", "export", "load", "pv" makes the standard plots of
        #   src/plotting.py work out of the box.

        self.var["load"]   = m.addVars(T,  lb=-GRB.INFINITY, name="load")     # l_t^L
        self.var["pv"]     = m.addVars(T,  lb=-GRB.INFINITY, name="pv")        # g_t^PV
        self.var["import"] = m.addVars(T, lb=-GRB.INFINITY, name="import")    # x_t^imp
        self.var["export"] = m.addVars(T, lb=-GRB.INFINITY, name="export")    # x_t^exp

        l   = self.var["load"]
        g   = self.var["pv"]
        xi  = self.var["import"]
        xe  = self.var["export"]

        P_imp = {t: d.energy_price[t] + d.import_tariff for t in T}
        P_exp = {t: d.energy_price[t] - d.export_tariff for t in T}


        # --- Objective ---------------------------------------------------------------
        # Question 1: utility enters as a benefit term, so the objective is cost minus utility.
        # Question 2(b): linear disutility of deviation from a reference profile.
        # Question 2(c): quadratic disutility of deviation from a reference profile.
        objective = gp.quicksum(
            d.pv_marginal_cost * g[t] + P_imp[t] * xi[t] - P_exp[t] * xe[t] for t in T
        )
        if d.consumption_utility is not None:
            objective -= gp.quicksum(d.consumption_utility * l[t] for t in T)
        if d.linear_disutility is not None and d.reference_load is not None: # Question 2.b iii) + iv)
            self.var["dev_plus"] = m.addVars(T, lb=0, name="dev_plus")
            self.var["dev_minus"] = m.addVars(T, lb=0, name="dev_minus")
            self.con["deviation"] = m.addConstrs(
                (l[t] - d.reference_load[t] == self.var["dev_plus"][t] - self.var["dev_minus"][t] for t in T),
                name="deviation",
            )
            objective += gp.quicksum(
                d.linear_disutility * (self.var["dev_plus"][t] + self.var["dev_minus"][t])
                for t in T
            )
        if d.quadratic_disutility is not None and d.reference_load is not None: # Question 2.c iii) + iv)
            objective += gp.quicksum(
                d.quadratic_disutility * (l[t] - d.reference_load[t]) ** 2
                for t in T
            )
        m.setObjective(objective, GRB.MINIMIZE)

        # --- Constraints -------------------------------------------------------------
        # TODO: add the constraints of your formulation.
        # Pattern for hourly constraints (one per hour, duals returned as a 24-vector; names are
        # indexed automatically, like for the variables):
        #   self.con["<name>"] = m.addConstrs(
        #       (<lhs expression> - <rhs expression> <= 0 for t in T), name="<name>")
        # Pattern for a single constraint (dual returned as a scalar):
        #   self.con["<name>"] = m.addConstr(<lhs expression> - <rhs expression> <= 0, name="<name>")
        self.con["balance"] = m.addConstrs(
            (l[t] + xe[t] - g[t] - xi[t] == 0 for t in T), name="balance")

        # (1b) L^min - l_t^L <= 0  -> dual mu_t^
        self.con["load_min"] = m.addConstrs(
            (d.load_min_kWh - l[t] <= 0 for t in T), name="load_min")

        # (1c) l_t^L - L^max <= 0  -> dual mu_bar_t^L  <-- this is the one you report in (c)
        self.con["load_max"] = m.addConstrs(
            (l[t] - d.load_max_kWh <= 0 for t in T), name="load_max")

        # (1d) -g_t^PV <= 0
        self.con["pv_min"] = m.addConstrs(
            (-g[t] <= 0 for t in T), name="pv_min")

        # (1e) g_t^PV - PV_t^max <= 0
        
        self.con["pv_max"] = m.addConstrs(
            (g[t] - d.pv_available[t] <= 0 for t in T), name="pv_max")

        # (1f), (1g): non-negativity of import/export as explicit constraints (only needed
        # if you want their duals too — otherwise the default lb=0 above already enforces them)
        self.con["import_nonneg"] = m.addConstrs((-xi[t] <= 0 for t in T), name="import_nonneg")
        self.con["export_nonneg"] = m.addConstrs((-xe[t] <= 0 for t in T), name="export_nonneg")


        m.update()
        return self

    # ------------------------------------------------------------------ 3. solve
    def solve(self) -> Results:
        """Optimise and return primal values, objective and dual values."""
        m = self.m
        m.update()
        if m.NumConstrs == 0 and m.NumQConstrs == 0:
            raise NotImplementedError(
                "The model has no constraints: complete FlexibleConsumerModel.build() in src/model.py first."
            )
        m.optimize()
        status = _status_name(m.Status)
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimisation ended with status {status}. Check the model (m.computeIIS() helps for infeasibility).")
        return self._extract_results(status)

    # --------------------------------------------------------------- extraction
    def _extract_results(self, status: str) -> Results:
        d, T = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = d.energy_price
        hourly["pv_available"] = d.pv_available
        if d.reference_load is not None:
            hourly["reference_load"] = d.reference_load

        # Primal values: every hourly variable family in self.var becomes a column
        for name, v in self.var.items():
            if isinstance(v, gp.tupledict):
                hourly[name] = [v[t].X for t in T]
        scalars = {name: v.X for name, v in self.var.items() if isinstance(v, gp.Var)}

        # Dual values: every constraint family in self.con becomes a 'dual_<name>' column or scalar
        duals: dict[str, float] = {}
        for name, c in self.con.items():
            try:
                if isinstance(c, gp.tupledict):
                    hourly[f"dual_{name}"] = [_dual(c[t]) for t in T]
                else:
                    duals[name] = _dual(c)
            except (AttributeError, gp.GurobiError):
                # No duals available (e.g. model with integer variables)
                pass
          # --- procurement cost and utility, split out of the objective ---------
        g, xi, xe, l = self.var["pv"], self.var["import"], self.var["export"], self.var["load"]
        P_imp = {t: d.energy_price[t] + d.import_tariff for t in T}
        P_exp = {t: d.energy_price[t] - d.export_tariff for t in T}
        procurement_cost = sum(d.pv_marginal_cost * g[t].X + P_imp[t] * xi[t].X - P_exp[t] * xe[t].X for t in T)
        utility = sum(d.consumption_utility * l[t].X for t in T) if d.consumption_utility is not None else 0.0
        disutility = 0.0
        if d.linear_disutility is not None and d.reference_load is not None:
            disutility = sum(d.linear_disutility * abs(l[t].X - d.reference_load[t]) for t in T)
        if d.quadratic_disutility is not None and d.reference_load is not None:
            disutility = sum(
                d.quadratic_disutility * (l[t].X - d.reference_load[t]) ** 2
                for t in T
            )
        assert abs((procurement_cost - utility + disutility) - self.m.ObjVal) < 1e-6, \
            "procurement_cost - utility + disutility should equal the objective value"

        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            procurement_cost=procurement_cost,   
            utility=utility, 
            disutility=disutility,
            hourly=hourly,
            duals=duals,
            meta={"scalar_variables": scalars},
        )


_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")
