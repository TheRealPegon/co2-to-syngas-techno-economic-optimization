"""
solver_utils.py
---------------
Shared solve backend for the flowsheet models (Membrane / PSA / AmineWash).

All three flowsheets build the same IPOPT option dictionary and variable
scaling suffix, then hand the model off to ``run_solver``.  Keeping the actual
solver invocation in one place means the single-solve path and the
multistart ("pseudo-global") path stay consistent across flowsheets.

Multistart
----------
True global solvers (MAiNGO / BARON) aren't available here, so we approximate
a global search by re-initialising the variables and re-solving with IPOPT
repeatedly, keeping the best objective.  We roll our own loop rather than use
``pyomo.contrib.multistart`` because, on these stiff flowsheets, a perturbed
restart routinely drives IPOPT to a *bad/error* status — and Pyomo's wrapper
solves the sub-problems with ``load_solutions=True``, which raises and aborts
the whole run (discarding the good incumbent).  Our loop instead:

  * solves the unperturbed incumbent first (robust, never raises),
  * perturbs *from that incumbent* and re-solves N times,
  * silently skips any restart that fails to reach an acceptable solution,
  * keeps the best objective and always falls back to the incumbent.

The default ``rand_guess_and_bound`` strategy perturbs each variable between
its current (incumbent) value and the farther bound — far more likely to
re-converge than a random point anywhere in the box (``rand``).
"""

import random

import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError

_OK_CONDITIONS = {
    pyo.TerminationCondition.optimal,
    pyo.TerminationCondition.locallyOptimal,
    pyo.TerminationCondition.feasible,
}

# Defaults for the multistart loop; override per call via multistart_kwargs.
_MULTISTART_DEFAULTS = {
    "strategy": "rand_guess_and_bound",  # "rand" | "rand_guess_and_bound"
    "iterations": 10,
    "seed": None,                        # set an int for reproducible restarts
}


def run_solver(
    model,
    solver_name: str,
    options: dict,
    show_log: bool = False,
    multistart: bool = False,
    multistart_kwargs: dict = None,
):
    """Solve ``model`` and load the solution back into it.

    Parameters
    ----------
    model : pyomo ConcreteModel
        The flowsheet to solve (its ``scaling_factor`` suffix is assumed set).
    solver_name : str
        Local NLP solver, e.g. "ipopt".  Used for the single solve and every
        multistart restart.
    options : dict
        Solver options (IPOPT keys) applied to each solve.
    show_log : bool
        Stream solver output to stdout.
    multistart : bool
        If True, run the multistart loop instead of a single solve.
    multistart_kwargs : dict, optional
        Overrides for the multistart loop (strategy, iterations, seed).

    Returns
    -------
    Pyomo SolverResults of the (best) solve, or None if the solver crashed.
    """
    if multistart:
        return _run_multistart(model, solver_name, options, show_log, multistart_kwargs)

    results = _single_solve(model, solver_name, options, show_log)
    if results is not None:
        tc = results.solver.termination_condition
        if tc not in _OK_CONDITIONS:
            print(f"\n[WARNING] Solver terminated with: {tc}. "
                  "Results may be infeasible or inaccurate.")
    return results


def _single_solve(model, solver_name, options, show_log):
    """One local solve.  Robust: returns None on crash and never raises on a
    bad/error solver status (the solution simply isn't loaded)."""
    solver = pyo.SolverFactory(solver_name)
    for k, v in options.items():
        solver.options[k] = v
    try:
        results = solver.solve(model, tee=show_log, load_solutions=False)
    except ApplicationError as e:
        print(f"\n[WARNING] Solver exited abnormally: {e}")
        return None
    # Load whatever we got so partial/feasible results are accessible.
    # Guarded: a "bad status: error" result raises here, which we swallow.
    try:
        model.solutions.load_from(results)
    except Exception:
        pass
    return results


def _result_ok(results) -> bool:
    return (results is not None
            and results.solver.termination_condition in _OK_CONDITIONS)


def _free_vars(model):
    """All unfixed variable data objects, descending into sub-blocks."""
    return [v for v in model.component_data_objects(pyo.Var) if not v.fixed]


def _snapshot(free_vars):
    """Capture current values as a list of (var, value) pairs.  Var references
    are stable because we solve in place (no cloning)."""
    return [(v, v.value) for v in free_vars]


def _restore(snapshot):
    for v, val in snapshot:
        if val is not None:
            v.set_value(val, skip_validation=True)


def _perturb(free_vars, strategy, rng):
    """Re-initialise bounded variables for the next restart.  Unbounded
    variables are left at their incumbent value (can't sample without bounds)."""
    for v in free_vars:
        lb, ub = v.lb, v.ub
        if lb is None or ub is None or ub <= lb:
            continue
        cur = v.value
        if strategy == "rand" or cur is None:
            v.set_value(rng.uniform(lb, ub), skip_validation=True)
        else:  # rand_guess_and_bound: between current value and the farther bound
            cur = min(max(cur, lb), ub)
            if (ub - cur) >= (cur - lb):
                v.set_value(rng.uniform(cur, ub), skip_validation=True)
            else:
                v.set_value(rng.uniform(lb, cur), skip_validation=True)


def _obj_value(model):
    obj = next(model.component_data_objects(pyo.Objective, active=True), None)
    if obj is None:
        return None, None
    sense = 1 if obj.sense == pyo.minimize else -1
    try:
        return float(pyo.value(obj.expr)), sense
    except Exception:
        return None, sense


def _run_multistart(model, solver_name, options, show_log, multistart_kwargs):
    cfg = dict(_MULTISTART_DEFAULTS)
    if multistart_kwargs:
        cfg.update(multistart_kwargs)
    n_iter = int(cfg.get("iterations", 10))
    strategy = cfg.get("strategy", "rand_guess_and_bound")
    rng = random.Random(cfg.get("seed"))

    obj = next(model.component_data_objects(pyo.Objective, active=True), None)
    if obj is None:
        print("  [multistart] no active objective — falling back to single solve")
        return _single_solve(model, solver_name, options, show_log)

    print(f"  [multistart] {n_iter} restarts via '{strategy}' (sub-solver: {solver_name})")

    free_vars = _free_vars(model)

    # 1) Establish the incumbent with a normal solve; perturb from this point.
    best_result = _single_solve(model, solver_name, options, show_log)
    base_point = _snapshot(free_vars)
    best_vals = None
    best_obj = None
    _, sense = _obj_value(model)
    sense = sense or 1
    if _result_ok(best_result):
        best_obj, _ = _obj_value(model)
        best_vals = _snapshot(free_vars)

    # 2) Perturbed restarts.
    n_improved = 0
    for _ in range(n_iter):
        _restore(base_point)                 # always perturb from the incumbent
        _perturb(free_vars, strategy, rng)
        res = _single_solve(model, solver_name, options, show_log=False)
        if not _result_ok(res):
            continue
        val, _ = _obj_value(model)
        if val is None:
            continue
        if best_obj is None or val * sense < best_obj * sense:
            best_obj = val
            best_vals = _snapshot(free_vars)
            best_result = res
            n_improved += 1

    # 3) Leave the model holding the best point found.
    if best_vals is not None:
        _restore(best_vals)
        print(f"  [multistart] best objective = {best_obj:.6g} "
              f"({n_improved} improvement(s) over incumbent)")
    else:
        print("  [multistart] no restart reached an acceptable solution")

    if not _result_ok(best_result):
        tc = best_result.solver.termination_condition if best_result else "crashed"
        print(f"\n[WARNING] Multistart best solve terminated with: {tc}. "
              "Results may be infeasible or inaccurate.")
    return best_result
