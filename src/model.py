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
    procurement_cost: float                 # DKK PV_cost + import_cost - export_revenue
    utility: float                         #
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
            f"procurement cost: {self.procurement_cost:.2f} DKK | utility: {self.utility:.2f} DKK\n"
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
        #d, m, T = self.data, self.m, self.T

        self._add_variables()
        self._add_constraints()
        self._procurement = self._procurement_expr()   # DKK
        self._preference = self._preference_expr()     # DKK
        self.m.setObjective(self._procurement + self._preference, GRB.MINIMIZE)
        self.m.update()
        return self
    
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

        # Setting lb to -infinity since we will be defining their lower bounds as constraints
    def _add_variables(self) -> None:
        """Hourly load, PV production, import and export (kWh/h). Free here; bounds are explicit constraints."""
        m, T = self.m, self.T
        self.var["load"]   = m.addVars(T, lb=-GRB.INFINITY, name="load")    # l_t^L
        self.var["pv"]     = m.addVars(T, lb=-GRB.INFINITY, name="pv")      # g_t^PV
        self.var["import"] = m.addVars(T, lb=-GRB.INFINITY, name="import")  # x_t^imp
        self.var["export"] = m.addVars(T, lb=-GRB.INFINITY, name="export")  # x_t^exp

        # --- Objective ---------------------------------------------------------------
        # TODO: express the objective function and its direction (GRB.MINIMIZE or GRB.MAXIMIZE):
        #   m.setObjective(gp.quicksum(<expression in t> for t in T), <direction>)
        # The input-data attributes (with units) are documented in src/data_loader.py (InputData).


        # --- Constraints -------------------------------------------------------------
        # TODO: add the constraints of your formulation.
        # Pattern for hourly constraints (one per hour, duals returned as a 24-vector; names are
        # indexed automatically, like for the variables):
        #   self.con["<name>"] = m.addConstrs(
        #       (<lhs expression> - <rhs expression> <= 0 for t in T), name="<name>")
        # Pattern for a single constraint (dual returned as a scalar):
        #   self.con["<name>"] = m.addConstr(<lhs expression> - <rhs expression> <= 0, name="<name>")
    def _add_constraints(self) -> None:
        """Power balance, load bounds, PV bounds and non-negativity of import/export."""
        d, m, T = self.data, self.m, self.T
        L, PV = self.var["load"], self.var["pv"]
        IMP, EXP = self.var["import"], self.var["export"]

        self.con["balance"] = m.addConstrs(
            (L[t] + EXP[t] - PV[t] - IMP[t] == 0 for t in T), name="balance")
        self.con["load_min"] = m.addConstrs(
            (d.load_min_kWh - L[t] <= 0 for t in T), name="load_min")
        self.con["load_max"] = m.addConstrs(
            (L[t] - d.load_max_kWh <= 0 for t in T), name="load_max")
        self.con["pv_min"] = m.addConstrs(
            (-PV[t] <= 0 for t in T), name="pv_min")
        self.con["pv_max"] = m.addConstrs(
            (PV[t] - d.pv_available[t] <= 0 for t in T), name="pv_max")
        self.con["imp_nonneg"] = m.addConstrs(
            (-IMP[t] <= 0 for t in T), name="imp_nonneg")
        self.con["exp_nonneg"] = m.addConstrs(
            (-EXP[t] <= 0 for t in T), name="exp_nonneg")

    def _procurement_expr(self) -> gp.LinExpr:
        """Daily procurement cost: PV production cost + import cost - export revenue (DKK)."""
        d, T = self.data, self.T
        PV, IMP, EXP = self.var["pv"], self.var["import"], self.var["export"]
        p_imp = d.energy_price + d.import_tariff   # effective import price, DKK/kWh
        p_exp = d.energy_price - d.export_tariff   # effective export price, DKK/kWh
        return gp.quicksum(
            d.pv_marginal_cost * PV[t] + p_imp[t] * IMP[t] - p_exp[t] * EXP[t] for t in T
        )

    def _preference_expr(self) -> gp.LinExpr:
        """Question 1: preference cost = -utility = -u^L * sum_t l_t (DKK)."""
        L = self.var["load"]
        return -gp.quicksum(self.data.consumption_utility * L[t] for t in self.T)
    
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

        procurement_cost = self._procurement.getValue()
        utility = -self._preference.getValue()   # Q2-3: utility = -disutility
        assert abs((procurement_cost - utility) - self.m.ObjVal) < 1e-6

        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            procurement_cost=procurement_cost,
            utility=utility,
            hourly=hourly,
            duals=duals,
            meta={"scalar_variables": scalars},
        )

# =========================== Add the model from 2b ================================================
class LinearDisutilityModel(FlexibleConsumerModel):

    def __init__(self, data: InputData, name: str = "linear_disutility", verbose: bool = False):
        if data.reference_load is None or data.linear_disutility is None:
            raise ValueError("LinearDisutilityModel needs reference_load and linear_disutility "
                             "(use the Q2_linear case).")
        super().__init__(data, name, verbose)

    def _add_variables(self) -> None:
        """Base variables plus the deviation above (delta_plus) and below (delta_minus) the reference (kWh/h)."""
        super()._add_variables()
        self.var["delta_plus"] = self.m.addVars(self.T, lb=-GRB.INFINITY, name="delta_plus")
        self.var["delta_minus"] = self.m.addVars(self.T, lb=-GRB.INFINITY, name="delta_minus")

    def _add_constraints(self) -> None:
        """Base constraints plus the deviation definition (10h) and non-negativity (10i)."""
        super()._add_constraints()
        d, m, T = self.data, self.m, self.T
        L = self.var["load"]
        Dp, Dm = self.var["delta_plus"], self.var["delta_minus"]
        self.con["deviation"] = m.addConstrs(
            (L[t] - float(d.reference_load[t]) - Dp[t] + Dm[t] == 0 for t in T), name="deviation")  # (10h)
        self.con["delta_plus_nonneg"] = m.addConstrs(
            (-Dp[t] <= 0 for t in T), name="delta_plus_nonneg")                                  # (10i)
        self.con["delta_minus_nonneg"] = m.addConstrs(
            (-Dm[t] <= 0 for t in T), name="delta_minus_nonneg") 

    def _preference_expr(self) -> gp.LinExpr:
        """Daily disutility sum_t c^L (delta_t^+ + delta_t^-) (DKK)."""
        Dp, Dm = self.var["delta_plus"], self.var["delta_minus"]
        return gp.quicksum(self.data.linear_disutility * (Dp[t] + Dm[t]) for t in self.T)

    def _extract_results(self, status: str) -> Results:
        """Base extraction plus the absolute hourly deviation |l_t - l_t^ref| (kWh/h)."""
        res = super()._extract_results(status)
        res.hourly["abs_deviation"] = res.hourly["delta_plus"] + res.hourly["delta_minus"]
        return res
    
        
_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")
