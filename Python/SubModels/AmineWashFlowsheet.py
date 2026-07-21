"""
AmineWashFlowsheet
==================
Combines the CO2 electrolyzer surrogate model and an MEA amine wash
unit into a single Pyomo optimisation model.

Process topology
----------------

    [CO2 fresh feed]
           |
           v
    +------------------+  <-- CO2 recycle (amine wash captured stream)
    |  CO2 Electrolyzer |
    +--------+---------+
             |  CO2 + CO + H2  (cathode gas, ~atmospheric pressure)
             v
    +---------------------------+
    |  MEA Amine Wash           |
    |  - absorbs 90 % of CO2   |---> CO2-rich stream (captured)
    |  - CO + H2 pass through   |        ↓
    +---------------------------+   (1-purge) recycled to electrolyzer
             |
             v  lean gas: CO + H2 + trace CO2
             |
          Syngas product

Recycle mole balance (closed loop)
-----------------------------------
    nf_CO2_elec_in = n_CO2_fresh + (1 - purge_fraction) × amine.nf_CO2_captured
    nf_CO_elec_in  ≈ 0   (no CO in amine recycle — CO passes to syngas)

NOTE on CO2 concentration
--------------------------
The amine wash TEC correlation (Kim & Léonard 2025) was validated for
5–50 mol% CO2. The electrolyzer outlet is typically 40–66 % CO2.
The y_CO2_in upper bound is enforced at 0.50, which forces the optimizer
to operate the electrolyzer at ≥50 % CO2 conversion so that the outlet
CO2 fraction stays within the validated range.

Design / optimisation variables
---------------------------------
    Electrolyzer : E_appl, v, n (number of cells via area)
    System       : n_CO2_fresh  (fresh CO2 feed rate [mol/s])
    Optional     : purge_fraction (when float_purge=True)

Objective
---------
    minimise  total_lifecycle_cost = total_capex + plant_life × total_opex
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyomo.environ as pyo

from configuration import Economics, Constants, H2Parameters
from SubModels.CO2_ElectrolyzerSurrogateModel import CO2Electrolyzer
from SubModels.AmineWash import AmineWash
from SubModels.H2_Electrolyzer_Model import H2Model
from SubModels.solver_utils import run_solver


class AmineWashFlowsheet(pyo.ConcreteModel):
    """
    See module docstring for full description.

    Parameters
    ----------
    purge_fraction : float
        Fraction of the captured CO2 recycle stream that is vented rather
        than recycled [0, 1].  Default 0.0 (fully closed recycle).
        Ignored when float_purge=True (used only as initial guess).
    float_purge : bool
        If True, purge_fraction becomes a Var (decision variable).
    capture_rate : float
        MEA capture rate passed to AmineWash (default 0.90).
    steam_cost : float
        LP steam cost [€/GJ] passed to AmineWash.
    cooling_cost : float
        Cooling water cost [€/GJ] passed to AmineWash.
    """

    def __init__(
        self,
        purge_fraction: float = 0.0,
        float_purge: bool = False,
        capture_rate: float = 0.90,
        steam_cost: float = None,   # None → AmineWash default
        cooling_cost: float = None, # None → AmineWash default
    ):
        super().__init__()

        self._float_purge = float_purge

        _econ      = Economics()
        op_hours   = _econ.operating_hours_per_year
        plant_life = _econ.plant_life_years

        # ==================================================================
        # 1.  CO2 Electrolyzer
        # ==================================================================
        self.elec = CO2Electrolyzer()
        # No H2 recycle in this flowsheet — H2 goes straight to syngas.
        self.elec.nf_H2_elec_in.fix(0.0)

        # ==================================================================
        # 2.  MEA Amine Wash
        # ==================================================================
        _amine_kwargs = {"capture_rate": capture_rate}
        if steam_cost   is not None: _amine_kwargs["steam_cost"]   = steam_cost
        if cooling_cost is not None: _amine_kwargs["cooling_cost"] = cooling_cost
        self.amine = AmineWash(**_amine_kwargs)

        # ==================================================================
        # 3.  Total electrolyzer outlet flow
        # ==================================================================
        self.n_total_elec_out = pyo.Var(
            initialize=150.0, bounds=(1.0, 1e5),
            doc="Total molar flow leaving electrolyzer cathode [mol/s]",
        )
        self.eq_n_total = pyo.Constraint(
            expr=self.n_total_elec_out
            == self.elec.nf_CO2_elec_out
            + self.elec.nf_CO_elec_out
            + self.elec.nf_H2_elec_out,
        )

        # ==================================================================
        # 4.  Connect electrolyzer outlet → amine wash inlet
        #
        #   The supplemental H2 (bought or from water electrolyzer) is
        #   mixed into the gas stream BEFORE the amine wash.  This serves
        #   two purposes:
        #     1. Dilutes the CO2 fraction so y_CO2_in stays ≤ 0.50
        #        (required by the TEC correlation validity range).
        #     2. H2 passes through MEA unchanged → the amine lean gas
        #        is already the syngas product with no extra mixing step.
        #
        #   nf_H2_premix is set to zero here and linked to the H2 source
        #   inside set_syngas_target().
        # ==================================================================
        self.nf_H2_premix = pyo.Var(
            initialize=0.0, bounds=(0.0, 1e5),
            doc="Supplemental H2 mixed in before amine wash [mol/s]",
        )

        self.eq_amine_CO2_in = pyo.Constraint(
            expr=self.amine.nf_CO2_in == self.elec.nf_CO2_elec_out,
        )
        self.eq_amine_inert_in = pyo.Constraint(
            expr=self.amine.nf_inert_in
            == self.elec.nf_CO_elec_out + self.elec.nf_H2_elec_out + self.nf_H2_premix,
        )

        # ==================================================================
        # 5.  Purge fraction
        # ==================================================================
        if float_purge:
            self.purge_fraction = pyo.Var(
                initialize=purge_fraction, bounds=(0.0, 1.0),
                doc="Fraction of CO2 recycle purged to flare [-]",
            )
        else:
            self.purge_fraction = pyo.Param(
                initialize=purge_fraction,
                doc="Fraction of CO2 recycle purged to flare [-]",
            )

        self.n_CO2_fresh = pyo.Var(
            initialize=50.0, bounds=(0.1, 1e5),
            doc="Fresh CO2 feed entering the system [mol/s]",
        )

        _recycle = 1.0 - self.purge_fraction

        # ==================================================================
        # 6.  Recycle mole balances
        #
        #   CO2 inlet = fresh feed + (1-purge) × captured CO2
        #   CO  inlet ≈ 0  (amine only recycles pure CO2; CO goes to syngas)
        #       Set to a nominal 1e-4 mol/s to respect the electrolyzer
        #       model's lower bound on nf_CO_elec_in (1e-6 mol/s).
        # ==================================================================
        self.eq_recycle_CO2 = pyo.Constraint(
            expr=self.elec.nf_CO2_elec_in
            == self.n_CO2_fresh + _recycle * self.amine.nf_CO2_captured,
        )
        # No CO recycled: fix CO inlet to effectively zero
        self.elec.nf_CO_elec_in.fix(1e-4)

        # ==================================================================
        # 7.  Economics aggregation
        # ==================================================================
        self.total_capex = pyo.Var(initialize=2e6, bounds=(0.0, None))
        self.total_opex  = pyo.Var(initialize=5e5, bounds=(0.0, None))

        self.eq_total_capex = pyo.Constraint(
            expr=self.total_capex == self.elec.CAPEX + self.amine.CAPEX,
        )
        self.eq_total_opex = pyo.Constraint(
            expr=self.total_opex
            == self.elec.OPEX * op_hours
            + self.amine.OPEX * op_hours,
        )

        self.total_lifecycle_cost = pyo.Var(initialize=1e7, bounds=(0.0, None))
        # Discounted lifecycle cost: CAPEX + PVF × OPEX
        # PVF = (1-(1+r)^-n)/r at r=8.5%, n=20 → PVF=9.46  (vs undiscounted 20)
        self.eq_lifecycle_cost = pyo.Constraint(
            expr=self.total_lifecycle_cost
            == self.total_capex + _econ.pvf * self.total_opex,
        )

        # ==================================================================
        # 8.  Objective: minimise total lifecycle cost
        # ==================================================================
        self.obj = pyo.Objective(
            expr=self.total_lifecycle_cost, sense=pyo.minimize,
        )

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def set_syngas_target(
        self,
        syngas_target: float = 40.0,
        syngas_ratio: float = 2.0,
        H2_type: str = "Electrolyzer",
        price_CO2_feed: float = Economics.price_CO2_feed,
        price_H2_buy: float = 0.96 * 3.0 * 2.016 / 1000,
    ):
        """Add syngas production target and H2 sourcing constraints.

        The syngas product is the lean gas from the amine wash:
            CO  — passes through the absorber unchanged
            H2  — passes through the absorber unchanged
            CO2 — small uncaptured fraction (10 % of electrolyzer CO2 out)

        H2/CO ratio is met by one of three strategies (H2_type):
            "Electrolyzer"   — use only by-product H2 from the CO2 electrolyzer.
            "Buy"            — supplement with externally purchased H2.
            "H2Electrolyzer" — supplement with a dedicated water electrolyzer.

        Parameters
        ----------
        syngas_target : float
            Desired syngas mass flow rate [t/day].
        syngas_ratio : float
            Molar H2/CO ratio in the syngas product.
        H2_type : str
            H2 sourcing strategy (see above).
        price_CO2_feed : float
            Cost of fresh CO2 feed [€/mol].
        price_H2_buy : float
            Purchase price of H2 [€/mol] (only used when H2_type="Buy").
        """
        if H2_type not in ("Electrolyzer", "Buy", "H2Electrolyzer"):
            raise ValueError("H2_type must be 'Electrolyzer', 'Buy', or 'H2Electrolyzer'")
        if hasattr(self, "_syngas_target_set"):
            raise RuntimeError("set_syngas_target() has already been called.")

        self._syngas_target_set = True
        self._H2_type = H2_type

        _econ            = Economics()
        op_hours         = _econ.operating_hours_per_year
        seconds_per_year = op_hours * 3600.0

        # --- Initial-guess back-calculation ----------------------------------
        _mf_kgs   = syngas_target * 1000.0 / (3600.0 * 24.0)  # kg/s
        _MW_sg    = (syngas_ratio * Constants.MW_H2 + Constants.MW_CO) / (syngas_ratio + 1)
        _nf_CO_ig = _mf_kgs / _MW_sg / (syngas_ratio + 1)
        _nf_H2_ig = syngas_ratio * _nf_CO_ig
        self._nf_CO_init_target = _nf_CO_ig

        _clip = lambda v, lo, hi: max(lo, min(hi, v))
        self.n_CO2_fresh.set_value(_clip(_nf_CO_ig * 1.5, 0.1, 1e4))
        self.n_total_elec_out.set_value(_clip((_nf_CO_ig + _nf_H2_ig) * 2.0, 1.0, 1e5))

        # --- Parameters ------------------------------------------------------
        self.syngas_ratio_param = pyo.Param(
            initialize=syngas_ratio,
            doc="H2/CO molar ratio in syngas [-]",
        )
        _mf_target_kgs = syngas_target * 1000.0 / (3600.0 * 24.0)
        self.mf_syngas_target = pyo.Param(
            initialize=_mf_target_kgs, mutable=True,
            doc="Syngas mass flow target [kg/s]",
        )

        # --- Syngas product variables ----------------------------------------
        self.nf_CO_syngas  = pyo.Var(initialize=_nf_CO_ig,             bounds=(1e-6, 1e4))
        self.nf_H2_syngas  = pyo.Var(initialize=_nf_H2_ig,             bounds=(1e-6, 1e4))
        self.nf_CO2_syngas = pyo.Var(initialize=1e-3,                   bounds=(1e-9, 1e4))
        self.nf_syngas     = pyo.Var(initialize=_nf_CO_ig + _nf_H2_ig, bounds=(1e-6, 1e4))
        self.mf_syngas     = pyo.Var(initialize=_mf_target_kgs,         bounds=(1e-9, 1e3))

        # --- Syngas composition constraints ----------------------------------
        # The amine lean gas IS the syngas: CO + H2_total + trace CO2 all
        # pass through the absorber unchanged.
        #
        # CO: all electrolyzer CO passes through
        self.eq_CO_syngas = pyo.Constraint(
            expr=self.nf_CO_syngas == self.elec.nf_CO_elec_out,
        )
        # Trace CO2: 10 % of electrolyzer CO2 not captured
        self.eq_CO2_syngas = pyo.Constraint(
            expr=self.nf_CO2_syngas == self.amine.nf_CO2_out,
        )

        # H2 sourcing — supplemental H2 is mixed BEFORE the amine wash so it
        # dilutes the CO2 fraction.  It passes through MEA and exits with the
        # lean gas as syngas H2.  nf_H2_premix is linked to the H2 source here.
        if H2_type == "Electrolyzer":
            # No supplemental H2 — only electrolyzer by-product H2
            self.eq_H2_premix = pyo.Constraint(
                expr=self.nf_H2_premix == 0.0,
            )
            self.eq_H2_syngas = pyo.Constraint(
                expr=self.nf_H2_syngas == self.elec.nf_H2_elec_out,
            )
        elif H2_type == "Buy":
            self.nf_H2_buy = pyo.Var(
                initialize=max(_nf_H2_ig - 1.0, 0.1), bounds=(0.0, 1e4),
                doc="Purchased H2 pre-mixed before amine wash [mol/s]",
            )
            # Link premix variable to the purchased H2 flow
            self.eq_H2_premix = pyo.Constraint(
                expr=self.nf_H2_premix == self.nf_H2_buy,
            )
            self.eq_H2_syngas = pyo.Constraint(
                expr=self.nf_H2_syngas
                == self.elec.nf_H2_elec_out + self.nf_H2_buy,
            )
        else:  # "H2Electrolyzer"
            _nf_H2_sup   = max(_nf_H2_ig - 1.0, 0.1)
            _kwh_per_kg  = H2Parameters.actual_kwh_per_kg_h2
            _power_guess = max(_nf_H2_sup * Constants.MW_H2 * _kwh_per_kg * 3600.0 / 1000.0, 10.0)
            self.h2_elec = H2Model()
            self.h2_elec.power_hp_mw.setlb(1.0)
            self.h2_elec.power_hp_mw.set_value(max(_power_guess, 1.0))
            for _c in (
                "total_h2_kwh_in_constraint",
                "total_h2_kg_constraint",
                "capex_for_1kg_h2_constraint",
                "annualized_capex_per_kg_h2_constraint",
                "total_annualized_cost_constraint",
                "cost_per_kg_h2_constraint",
            ):
                getattr(self.h2_elec, _c).deactivate()
            # Link premix variable to the water electrolyzer output
            self.eq_H2_premix = pyo.Constraint(
                expr=self.nf_H2_premix == self.h2_elec.H2_out_mol_per_sec,
            )
            self.eq_H2_syngas = pyo.Constraint(
                expr=self.nf_H2_syngas
                == self.elec.nf_H2_elec_out + self.h2_elec.H2_out_mol_per_sec,
            )

        # H2/CO ratio
        self.eq_syngas_ratio = pyo.Constraint(
            expr=self.nf_H2_syngas == syngas_ratio * self.nf_CO_syngas,
        )
        # Total molar flow
        self.eq_nf_syngas = pyo.Constraint(
            expr=self.nf_syngas
            == self.nf_CO_syngas + self.nf_H2_syngas + self.nf_CO2_syngas,
        )
        # Mass flow [kg/s] — CO + H2 only; CO2 is an impurity, not counted
        self.eq_mf_syngas = pyo.Constraint(
            expr=self.mf_syngas
            == (
                self.nf_CO_syngas * Constants.MW_CO
                + self.nf_H2_syngas * Constants.MW_H2
            ),
        )
        # Mass flow target (equality)
        self.eq_mf_syngas_target = pyo.Constraint(
            expr=self.mf_syngas == self.mf_syngas_target,
        )

        # --- Feedstock costs -------------------------------------------------
        self.price_CO2_feed_param = pyo.Param(
            initialize=price_CO2_feed, doc="CO2 fresh feed price [€/mol]",
        )
        self.cost_CO2_feed = pyo.Var(initialize=1e4, bounds=(0.0, None))
        self.eq_cost_CO2_feed = pyo.Constraint(
            expr=self.cost_CO2_feed
            == self.price_CO2_feed_param * self.n_CO2_fresh * seconds_per_year,
        )

        _feedstock_opex = self.cost_CO2_feed

        if H2_type == "Buy":
            self.price_H2_buy_param = pyo.Param(
                initialize=price_H2_buy, doc="Purchased H2 price [€/mol]",
            )
            self.cost_H2_buy = pyo.Var(initialize=1e4, bounds=(0.0, None))
            self.eq_cost_H2_buy = pyo.Constraint(
                expr=self.cost_H2_buy
                == self.price_H2_buy_param * self.nf_H2_buy * seconds_per_year,
            )
            _feedstock_opex = _feedstock_opex + self.cost_H2_buy

        # --- Update total_opex to include feedstock + optional H2 elec ------
        _base_opex = self.elec.OPEX * op_hours + self.amine.OPEX * op_hours
        if H2_type == "H2Electrolyzer":
            _base_opex = _base_opex + self.h2_elec.OPEX + self.h2_elec.MAINTENANCE

        self.eq_total_opex.deactivate()
        self.eq_total_opex_syngas = pyo.Constraint(
            expr=self.total_opex == _base_opex + _feedstock_opex,
        )

        # --- Update total_capex to include H2 electrolyzer if used ----------
        if H2_type == "H2Electrolyzer":
            self.eq_total_capex.deactivate()
            self.eq_total_capex_syngas = pyo.Constraint(
                expr=self.total_capex
                == self.elec.CAPEX + self.amine.CAPEX + self.h2_elec.CAPEX,
            )

    # -----------------------------------------------------------------------

    def initialize_electrolyzer(
        self,
        E_appl: float = -1.31,
        v: float = 0.08,
        CO2share: float = 0.999,
        nf_CO_out_target: float = None,
    ):
        """Seed all variables to a self-consistent starting point.

        Evaluates the surrogate in pure Python (no solver) and propagates
        consistent values into the electrolyzer and amine wash variables.
        Must be called before solve().

        Parameters
        ----------
        E_appl : float
            Applied voltage [V].
        v : float
            Gas velocity in electrolyzer channel [m/s].
        CO2share : float
            Inlet CO2 mole fraction [-].  Default 0.90 (higher than membrane
            case since no CO is recycled back to the electrolyzer here).
        nf_CO_out_target : float or None
            Target CO outlet [mol/s]. Auto-detected from set_syngas_target()
            if previously called, otherwise defaults to 15.0.
        """
        from math import exp as _exp

        if nf_CO_out_target is None:
            nf_CO_out_target = getattr(self, "_nf_CO_init_target", 15.0)

        _P = 1e5; _R = 8.314; _T = 298.0
        c_total = _P / (_R * _T)

        # --- Clip CO2share so CO_ineq constraint holds ---
        try:
            _denom  = E_appl - v ** (-0.46066585)
            _base2  = E_appl * -0.6278232
            _term2  = _base2 ** 21.189863 if _base2 >= 0 else -((-_base2) ** 21.189863)
            _inner  = 1.4358697 / _denom - _term2 + E_appl * -1.1215197
            _c_co_u = max(_inner, 1e-4) ** 10.42661
        except Exception:
            _c_co_u = 1.5 * c_total
        _co2share_min = c_total / max(_c_co_u + c_total, 1e-9)
        if CO2share < _co2share_min + 0.02:
            CO2share = min(_co2share_min + 0.03, 0.99)

        c_in_CO2 = c_total * CO2share
        c_in_CO  = c_total * (1.0 - CO2share)

        # --- Surrogate evaluation ---
        c_out_CO2 = (
            (E_appl * (((v ** 0.25 + E_appl) * -3.7203176) - 1.3941393)) ** 3
            + 40.244705
        ) * CO2share
        c_out_CO2 = max(min(c_out_CO2, c_in_CO2 * 0.999), 1e-3)

        try:
            _denom  = E_appl - v ** (-0.46066585)
            _base2  = E_appl * -0.6278232
            _term2  = _base2 ** 21.189863 if _base2 >= 0 else -((-_base2) ** 21.189863)
            _inner  = 1.4358697 / _denom - _term2 + E_appl * -1.1215197
            c_out_CO = CO2share * (max(_inner, 1e-4) ** 10.42661)
        except Exception:
            c_out_CO = c_in_CO * 1.5
        c_out_CO = max(c_out_CO, c_in_CO + 1e-3)

        _base_h2  = (E_appl ** 2 * 0.26091927) + 0.35308436
        c_out_H2  = (max(_base_h2, 1e-6) ** 14.994374) / v
        c_out_H2  = max(c_out_H2, 1e-3)
        c_mean_CO2 = (c_in_CO2 + c_out_CO2) / 2.0 * 0.85

        vol_flow   = nf_CO_out_target / max(c_out_CO, 1e-6)
        nf_CO2_out = c_out_CO2 * vol_flow
        nf_CO_out  = c_out_CO  * vol_flow
        nf_H2_out  = c_out_H2  * vol_flow
        nf_total   = nf_CO2_out + nf_CO_out + nf_H2_out
        nf_CO2_in  = c_in_CO2 * vol_flow
        nf_CO_in   = c_in_CO  * vol_flow

        # Seed electrolyzer
        self.elec.E_appl.set_value(E_appl)
        self.elec.v.set_value(v)
        # CO2share ≈ 1.0 since no CO is recycled back — seed close to upper bound
        # to avoid ill-conditioning from (1 - CO2share) → 0 during iteration
        _co2share_init = nf_CO2_in / max(nf_CO2_in + 1e-4, 1e-9)
        _co2share_init = min(max(_co2share_init, 0.80), 0.9999)
        self.elec.CO2share.set_value(_co2share_init)
        self.elec.c_elec_in_CO2.set_value(c_in_CO2)
        self.elec.c_elec_in_CO.set_value(c_in_CO)
        self.elec.c_elec_out_CO2.set_value(c_out_CO2)
        self.elec.c_elec_out_CO.set_value(c_out_CO)
        self.elec.c_elec_out_H2.set_value(c_out_H2)
        self.elec.c_elec_mean_CO2.set_value(c_mean_CO2)
        self.elec.vol_gas_flow.set_value(vol_flow)
        self.elec.nf_CO2_elec_in.set_value(max(nf_CO2_in, 0.1))
        # nf_CO_elec_in is fixed to 1e-4 — do not override
        self.elec.nf_CO2_elec_out.set_value(nf_CO2_out)
        self.elec.nf_CO_elec_out.set_value(nf_CO_out)
        self.elec.nf_H2_elec_out.set_value(nf_H2_out)

        # Seed n_total_elec_out
        self.n_total_elec_out.set_value(nf_total)

        # Seed fresh CO2 feed (= CO2 in - recycle; approximate as 10 % at start)
        _cap_rate   = pyo.value(self.amine.capture_rate)
        _nf_CO2_cap = nf_CO2_out * _cap_rate
        _purge_val  = (pyo.value(self.purge_fraction)
                       if isinstance(self.purge_fraction, pyo.Param)
                       else pyo.value(self.purge_fraction))
        _recycle_CO2 = (1.0 - _purge_val) * _nf_CO2_cap
        _fresh       = max(nf_CO2_in - _recycle_CO2, 1.0)
        self.n_CO2_fresh.set_value(_fresh)

        # Seed nf_H2_premix: target H2/CO = 2 as default guess
        # nf_H2_premix ≈ 2*CO - H2_elec (supplemental H2 needed)
        _h2_premix_guess = max(2.0 * nf_CO_out - nf_H2_out, 0.0)
        self.nf_H2_premix.set_value(_h2_premix_guess)

        # Seed amine wash variables
        self.amine.nf_CO2_in.set_value(nf_CO2_out)
        self.amine.nf_inert_in.set_value(nf_CO_out + nf_H2_out + _h2_premix_guess)
        self.amine.nf_CO2_captured.set_value(nf_CO2_out * _cap_rate)
        self.amine.nf_CO2_out.set_value(nf_CO2_out * (1 - _cap_rate))
        _nf_total_amine = nf_total + _h2_premix_guess
        _y = nf_CO2_out / max(_nf_total_amine, 1e-9)
        _y_clipped = min(max(_y, 0.05), 0.50)
        self.amine.y_CO2_in.set_value(_y_clipped)
        _F = _nf_total_amine * 22.414e-3 * 3.6
        self.amine.F_thousands.set_value(max(min(_F, 1613.81), 4.03))
        _S = nf_CO2_out * _cap_rate * 44.01e-3 * 3.6
        self.amine.scale_tph.set_value(max(_S, 0.1))

    # -----------------------------------------------------------------------

    def solve(
        self,
        solver_name: str = "ipopt",
        show_log: bool = False,
        solver_options: dict = None,
        multistart: bool = False,
        multistart_kwargs: dict = None,
    ):
        """Solve the combined NLP with IPOPT.

        Set ``multistart=True`` to instead run ``pyomo.contrib.multistart``
        (repeated IPOPT restarts, best objective kept) for a pseudo-global
        solution; ``multistart_kwargs`` overrides iterations/strategy.
        """
        if not hasattr(self, "scaling_factor"):
            self.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        sf = self.scaling_factor

        # Cost variables
        sf[self.total_lifecycle_cost] = 1e-6
        sf[self.total_capex]          = 1e-5
        sf[self.total_opex]           = 1e-4
        if hasattr(self, "cost_CO2_feed"):
            sf[self.cost_CO2_feed]    = 1e-4
        if hasattr(self, "cost_H2_buy"):
            sf[self.cost_H2_buy]      = 1e-4

        # Electrolyzer
        sf[self.elec.CAPEX]      = 1e-5
        sf[self.elec.OPEX]       = 1e-1
        sf[self.elec.Power]      = 1e-4
        sf[self.elec.I]          = 1e-5
        sf[self.elec.n]          = 1e-1
        sf[self.elec.total_area] = 1e-4

        # Amine wash
        sf[self.amine.CAPEX]         = 1e-5
        sf[self.amine.OPEX]          = 1e-1
        sf[self.amine.TEC]           = 1e-1
        sf[self.amine.Q_reboiler_kW] = 1e-3
        sf[self.amine.W_elec_kW]     = 1e-2
        sf[self.amine.Q_cooling_kW]  = 1e-3

        if hasattr(self, "h2_elec"):
            sf[self.h2_elec.CAPEX]                      = 1e-5
            sf[self.h2_elec.OPEX]                       = 1e-4
            sf[self.h2_elec.MAINTENANCE]                = 1e-4
            sf[self.h2_elec.power_hp_mw]               = 1e-1
            sf[self.h2_elec.H2_out_kg_per_sec]         = 1e1
            sf[self.h2_elec.H2_out_mol_per_sec]        = 1e-1

        _defaults = {
            "nlp_scaling_method": "gradient-based",
            "max_iter": 10000,
            "tol": 1e-5,
            "acceptable_tol": 1.0,
            "acceptable_dual_inf_tol": 1e6,
            "acceptable_constr_viol_tol": 1.0,
            "acceptable_compl_inf_tol": 1e6,
            "acceptable_iter": 5,
            "mu_strategy": "adaptive",
            "mu_init": 1e-1,
            "bound_push": 1e-4,
            "bound_frac": 1e-4,
            "print_level": 5 if show_log else 0,
        }
        if solver_options:
            _defaults.update(solver_options)

        return run_solver(
            self, solver_name, _defaults, show_log=show_log,
            multistart=multistart, multistart_kwargs=multistart_kwargs,
        )

    # -----------------------------------------------------------------------

    def report(self):
        """Print a structured summary of the amine wash flowsheet solution."""
        from prettytable import PrettyTable

        _econ      = Economics()
        op_hours   = _econ.operating_hours_per_year
        plant_life = _econ.plant_life_years
        _cv        = pyo.value

        sep = "=" * 68
        print("\n" + sep)
        print(f"{'AMINE WASH FLOWSHEET SUMMARY':^68}")
        print(f"{'[Electrolyzer -> Amine Wash -> Syngas]':^68}")
        print(sep)

        # --- Electrolyzer ---------------------------------------------------
        print("\n[CO2 ELECTROLYZER]")
        print(f"  E_appl        : {_cv(self.elec.E_appl):.4f}  V")
        print(f"  CD            : {_cv(self.elec.CD):.2f}  A/m²"
              f"  ({_cv(self.elec.CD)*0.1:.2f} mA/cm²)")
        print(f"  v             : {_cv(self.elec.v):.4f}  m/s")
        print(f"  n (cells)     : {_cv(self.elec.n)*1e6:.3e}")
        print(f"  Power         : {_cv(self.elec.Power)/1e6:.3f}  MW")
        print(f"  CO2share_in   : {_cv(self.elec.CO2share):.4f}")
        print(f"  n_CO2_fresh   : {_cv(self.n_CO2_fresh):.3f}  mol/s")
        print(f"  nf_CO2_in     : {_cv(self.elec.nf_CO2_elec_in):.3f}  mol/s")
        print(f"  nf_CO2_out    : {_cv(self.elec.nf_CO2_elec_out):.3f}  mol/s")
        print(f"  nf_CO_out     : {_cv(self.elec.nf_CO_elec_out):.3f}  mol/s")
        print(f"  nf_H2_out     : {_cv(self.elec.nf_H2_elec_out):.3f}  mol/s")
        print(f"  Total cathode : {_cv(self.n_total_elec_out):.3f}  mol/s")
        print(f"  lamda_CO2     : {_cv(self.elec.lamda_CO2):.4f}  -")
        print(f"  Vcell         : {_cv(self.elec.Vcell):.4f}  V")
        print(f"  total_area    : {_cv(self.elec.total_area):,.1f}  m²")
        print(f"  CAPEX         : €{_cv(self.elec.CAPEX):,.0f}")
        print(f"  OPEX          : €{_cv(self.elec.OPEX)*op_hours:,.0f}/yr")

        # --- Amine Wash -----------------------------------------------------
        _y   = _cv(self.amine.y_CO2_in)
        _cap = _cv(self.amine.nf_CO2_captured)
        _pf  = _cv(self.purge_fraction)
        _pf_label = " (optimised)" if self._float_purge else " (fixed)"

        print("\n[MEA AMINE WASH]")
        print(f"  y_CO2_in      : {_y*100:.2f}  mol%"
              f"  {'(extrapolated > 50 mol%)' if _y > 0.5001 else ''}")
        print(f"  F feed        : {_cv(self.amine.F_thousands):.2f}  10³ Nm³/h")
        print(f"  Capture scale : {_cv(self.amine.scale_tph):.2f}  tCO2/h"
              f"  ({_cv(self.amine.scale_tph)*op_hours/1e3:.1f} kt/y)")
        print(f"  nf_CO2_in     : {_cv(self.amine.nf_CO2_in):.3f}  mol/s")
        print(f"  nf_CO2_cap    : {_cap:.3f}  mol/s")
        print(f"  nf_CO2_out    : {_cv(self.amine.nf_CO2_out):.3f}  mol/s")
        print(f"  purge_frac    : {_pf:.4f}{_pf_label}")
        print(f"  CO2 recycled  : {(1-_pf)*_cap:.3f}  mol/s")
        print(f"  TEC           : {_cv(self.amine.TEC):.2f}  M€")
        print(f"  Q_reboiler    : {_cv(self.amine.Q_reboiler_spec):.3f}  GJ/tCO2"
              f"  ({_cv(self.amine.Q_reboiler_kW)/1e3:.2f} MW)")
        print(f"  W_elec        : {_cv(self.amine.W_elec_spec):.3f}  kWh/tCO2"
              f"  ({_cv(self.amine.W_elec_kW)/1e3:.2f} MW)")
        print(f"  Q_cooling     : {_cv(self.amine.Q_cooling_spec):.3f}  GJ/tCO2"
              f"  ({_cv(self.amine.Q_cooling_kW)/1e3:.2f} MW)")
        print(f"  CAPEX         : €{_cv(self.amine.CAPEX):,.0f}")
        print(f"  OPEX          : €{_cv(self.amine.OPEX)*op_hours:,.0f}/yr")

        # --- Syngas ---------------------------------------------------------
        if hasattr(self, "_syngas_target_set"):
            _mf_tpd = _cv(self.mf_syngas) * 3600.0 * 24.0 / 1000.0
            print("\n[SYNGAS PRODUCT]")
            print(f"  H2 type       : {self._H2_type}")
            print(f"  H2/CO ratio   : {_cv(self.syngas_ratio_param):.2f}")
            print(f"  nf_CO         : {_cv(self.nf_CO_syngas):.3f}  mol/s")
            print(f"  nf_H2         : {_cv(self.nf_H2_syngas):.3f}  mol/s")
            print(f"  nf_CO2 (trace): {_cv(self.nf_CO2_syngas):.4f}  mol/s")
            _co2_tpd = _cv(self.nf_CO2_syngas) * Constants.MW_CO2 * 3600 * 24 / 1000
            print(f"  Mass flow     : {_mf_tpd:.3f}  t/day  (CO+H2 only)"
                  f"  (target: {_cv(self.mf_syngas_target)*3600*24/1000:.3f} t/day)")
            print(f"  CO2 impurity  : {_co2_tpd:.3f}  t/day  (not counted in target)")
            print(f"  Annual CO2 cost: €{_cv(self.cost_CO2_feed):,.0f}/yr")
            if self._H2_type == "Buy":
                print(f"  nf_H2_buy     : {_cv(self.nf_H2_buy):.3f}  mol/s")
                print(f"  Annual H2 cost: €{_cv(self.cost_H2_buy):,.0f}/yr")
            elif self._H2_type == "H2Electrolyzer":
                print("\n[H2 WATER ELECTROLYZER]")
                print(f"  Power         : {_cv(self.h2_elec.power_hp_mw):.2f}  MW")
                print(f"  H2 produced   : {_cv(self.h2_elec.H2_out_mol_per_sec):.3f}  mol/s")
                print(f"  CAPEX         : €{_cv(self.h2_elec.CAPEX):,.0f}")
                print(f"  OPEX          : €{_cv(self.h2_elec.OPEX):,.0f}/yr")
                print(f"  Maintenance   : €{_cv(self.h2_elec.MAINTENANCE):,.0f}/yr")

        # --- Cost tables ----------------------------------------------------
        capex_rows = [
            ("CO2 Electrolyzer", _cv(self.elec.CAPEX)),
            ("MEA Amine Wash",   _cv(self.amine.CAPEX)),
        ]
        opex_rows = [
            ("CO2 Electrolyzer electricity+maint.", _cv(self.elec.OPEX) * op_hours),
            ("MEA Amine Wash",   _cv(self.amine.OPEX) * op_hours),
        ]
        if hasattr(self, "_syngas_target_set"):
            opex_rows.append(("CO2 feed cost", _cv(self.cost_CO2_feed)))
            if self._H2_type == "Buy":
                capex_rows.append(("H2 Electrolyzer (none)", 0.0))
                opex_rows.append(("H2 purchase cost", _cv(self.cost_H2_buy)))
            elif self._H2_type == "H2Electrolyzer":
                capex_rows.append(("H2 Electrolyzer", _cv(self.h2_elec.CAPEX)))
                opex_rows.append(("H2 Electrolyzer",
                                  (_cv(self.h2_elec.OPEX) or 0.0)
                                  + (_cv(self.h2_elec.MAINTENANCE) or 0.0)))

        def _cost_table(title, rows, unit):
            total = sum(v for _, v in rows)
            t = PrettyTable()
            t.field_names = ["Component", f"Cost ({unit})", "Share (%)"]
            t.align["Component"] = "l"
            t.align[f"Cost ({unit})"] = "r"
            t.align["Share (%)"] = "r"
            for name, val in rows:
                share = val / total * 100.0 if total > 0 else 0.0
                t.add_row([name, f"{val:,.0f}", f"{share:.1f}"])
            t.add_divider()
            t.add_row(["TOTAL", f"{total:,.0f}", "100.0"])
            print(f"\n{title}")
            print(t)

        _cost_table("CAPEX BREAKDOWN", capex_rows, "€")
        _cost_table("OPEX BREAKDOWN  (€/yr)", opex_rows, "€/yr")

        # --- Totals ---------------------------------------------------------
        print("\n" + sep)
        print(f"{'COMBINED FLOWSHEET TOTALS':^68}")
        print(sep)
        _tc  = _cv(self.total_capex)
        _to  = _cv(self.total_opex)
        _tlc = _cv(self.total_lifecycle_cost)
        print(f"  Total CAPEX                    : €{_tc:,.0f}")
        print(f"  Total Annual OPEX              : €{_to:,.0f}/yr")
        print(f"  NPV Cost ({plant_life} yr, {_econ.interest_rate*100:.1f}%)          : €{_tlc:,.0f}  [CAPEX + PVF×OPEX, PVF={_econ.pvf:.2f}]")
        if hasattr(self, "_syngas_target_set"):
            _mf_tpd = _cv(self.mf_syngas) * 3600.0 * 24.0 / 1000.0
            if _mf_tpd > 0:
                _lc_per_t = _tlc / (_mf_tpd * 365.0 * plant_life)
                print(f"  Lifecycle cost / syngas        : €{_lc_per_t:,.2f}/t")
        print(sep + "\n")
