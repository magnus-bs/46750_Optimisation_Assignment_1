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
        # --- Objective ---------------------------------------------------------------
        # TODO: express the objective function and its direction (GRB.MINIMIZE or GRB.MAXIMIZE):
        #   m.setObjective(gp.quicksum(<expression in t> for t in T), <direction>)
        # The input-data attributes (with units) are documented in src/data_loader.py (InputData).
        # We define our objective function as the procurement + preference (preference is negative). And call it not as a method but 
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


    # We define the balance constraint as a seperate method, so that we may alter it when we build our battery model in 3g.
    def _balance_lhs(self, t:int) -> gp.LinExpr:
        """Net energy demand in hour t (kWh/h); the power balance sets it to 0.
        Subclasses with storage add the charging/discharging terms."""
        L, PV = self.var["load"], self.var["pv"]
        IMP, EXP = self.var["import"], self.var["export"]
        return L[t] + EXP[t] - PV[t] - IMP[t]

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
            (self._balance_lhs(t) == 0 for t in T), name="balance") # Note that the method we defined before is used to build the constraint.
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

# ================================================= Add the model from 2c ===============================================
class QuadraticDisutilityModel(FlexibleConsumerModel):

    def __init__(self, data: InputData, name: str = "quadratic_disutility", verbose: bool = False):
        if data.reference_load is None or data.quadratic_disutility is None:
            raise ValueError("QuadraticDisutilityModel needs reference_load and quadratic_disutility "
                             "(use the Q2_quadratic or Q3 case).")
        super().__init__(data, name, verbose)

    def _preference_expr(self) -> gp.QuadExpr:
        """Daily disutility sum_t c^Q (l_t - l_t^ref)^2 (DKK)."""
        d, L = self.data, self.var["load"]
        cQ = d.quadratic_disutility
        return gp.quicksum(
            cQ * (L[t] - float(d.reference_load[t])) * (L[t] - float(d.reference_load[t])) for t in self.T
        )

    def _extract_results(self, status: str) -> Results:
        """Base extraction plus the signed and absolute hourly deviation from the reference (kWh/h)."""
        res = super()._extract_results(status)
        res.hourly["deviation"] = res.hourly["load"] - res.hourly["reference_load"]
        res.hourly["abs_deviation"] = res.hourly["deviation"].abs()
        return res

# ======================================== add the model from q3 ==================================
class MinEnergyModel(QuadraticDisutilityModel):

    def __init__(self, data: InputData, name: str = "min_energy", verbose: bool = False):
        if data.min_daily_energy_kWh is None:
            raise ValueError("MinEnergyModel needs min_daily_energy_kWh (use the Q3 case).")
        if data.min_daily_energy_kWh > data.n_hours * data.load_max_kWh:
            raise ValueError(
                f"Infeasible: E_min = {data.min_daily_energy_kWh} kWh exceeds the maximum daily "
                f"consumption {data.n_hours * data.load_max_kWh} kWh (24 * L_max)."
            )
        super().__init__(data, name, verbose)

    def _add_constraints(self) -> None:
        """Constraints of 2.(c) plus the minimum daily energy requirement (13i)."""
        super()._add_constraints()
        L = self.var["load"]
        self.con["min_energy"] = self.m.addConstr(
            self.data.min_daily_energy_kWh - gp.quicksum(L[t] for t in self.T) <= 0, name="min_energy")  # (13i)

# ======================================= Add the battery model ======================================
class BatteryModel(MinEnergyModel):
    """Question 3.(g): the Q3 consumer with a small battery (eqs. 16a-16k).

    New variables (one per hour): charge b_t^ch and discharge b_t^dis (kWh/h), and the
    state of charge s_t at the end of hour t (kWh). The battery starts at S^0 and must
    return to S^0 at the end of the day (16d). Battery energy does not count towards E^min:
    only the load l_t does.

    Indexing: the report uses s_1..s_24 with s_0 = S^0; the code uses soc[0..23], so
    soc[t] in the code is s_{t+1} in the report, and S^0 enters as a parameter.
    """

    def __init__(self, data: InputData, name: str = "battery", verbose: bool = False):
        if data.battery_capacity_kWh is None:
            raise ValueError("BatteryModel needs battery data (use the Q3_battery case).")
        super().__init__(data, name, verbose)

    def _add_variables(self) -> None:
        """Variables of Q3 plus charging, discharging (kWh/h) and state of charge (kWh)."""
        super()._add_variables()
        m, T = self.m, self.T
        self.var["charge"] = m.addVars(T, lb=-GRB.INFINITY, name="charge")        # b_t^ch
        self.var["discharge"] = m.addVars(T, lb=-GRB.INFINITY, name="discharge")  # b_t^dis
        self.var["soc"] = m.addVars(T, lb=-GRB.INFINITY, name="soc")              # s_t

    def _balance_lhs(self, t: int) -> gp.LinExpr:
        """Power balance (16b): charging adds to demand, discharging adds to supply."""
        return super()._balance_lhs(t) + self.var["charge"][t] - self.var["discharge"][t]

    def _add_constraints(self) -> None:
        """Constraints of Q3 plus the battery constraints (16c)-(16g)."""
        super()._add_constraints()
        d, m, T = self.data, self.m, self.T
        Bc, Bd, S = self.var["charge"], self.var["discharge"], self.var["soc"]
        eta_c, eta_d = d.battery_charging_efficiency, d.battery_discharging_efficiency
        S0, Smax = d.battery_initial_soc_kWh, d.battery_capacity_kWh

        def soc_prev(t):
            """State of charge at the start of hour t (S^0 for the first hour)."""
            return S0 if t == 0 else S[t - 1]

        self.con["soc_dynamics"] = m.addConstrs(                                    # (16c)
            (S[t] - soc_prev(t) - eta_c * Bc[t] + Bd[t] / eta_d == 0 for t in T), name="soc_dynamics")
        self.con["soc_end"] = m.addConstr(S[T[-1]] - S0 == 0, name="soc_end")       # (16d)
        self.con["soc_min"] = m.addConstrs((-S[t] <= 0 for t in T), name="soc_min")             # (16e)
        self.con["soc_max"] = m.addConstrs((S[t] - Smax <= 0 for t in T), name="soc_max")       # (16e)
        self.con["charge_min"] = m.addConstrs((-Bc[t] <= 0 for t in T), name="charge_min")      # (16f)
        self.con["charge_max"] = m.addConstrs(
            (Bc[t] - d.battery_max_charge_kW <= 0 for t in T), name="charge_max")               # (16f)
        self.con["discharge_min"] = m.addConstrs((-Bd[t] <= 0 for t in T), name="discharge_min")  # (16g)
        self.con["discharge_max"] = m.addConstrs(
            (Bd[t] - d.battery_max_discharge_kW <= 0 for t in T), name="discharge_max")         # (16g)

    def _extract_results(self, status: str) -> Results:
        """Q3 extraction plus the overlap check of 3.(g).ii: min(charge, discharge) per hour."""
        res = super()._extract_results(status)
        h = res.hourly
        h["overlap"] = h[["charge", "discharge"]].min(axis=1)
        res.meta["max_overlap_kWh"] = float(h["overlap"].max())
        return res

_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")
