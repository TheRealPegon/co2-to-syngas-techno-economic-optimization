"""
PSAFlowsheet
============
Combines the CO2 electrolyzer surrogate model and a PSA separation unit
into a single Pyomo optimisation model.

Process topology
----------------

    [CO2 fresh feed]
           |
           v
    +------------------+  <-- CO2/CO recycle (PSA low-pressure desorption outlet)
    |  CO2 Electrolyzer |   H2 in recycle is dropped (PSA does not model H2)
    +--------+---------+
             |  CO2 + CO + H2  (cathode gas, atmospheric pressure)
             |
             |   H2 --------> syngas product (passes through PSA unadsorbed)
             |
             v  CO2 + CO
    +-----------------------------+
    |  Pressure Swing Adsorption  |----> psaL_out (CO2-rich, low pressure)
    |  (PSA)                      |         |
    +-------------+---------------+         v
                  |                    recycle to electrolyzer
                  v                    (purge fraction vented)
           psaH_out: CO-rich (high purity CO product)

Recycle mole balance (closed loop)
------------------------------------
    nf_CO2_elec_in = n_CO2_fresh + (1 - purge_fraction) * psa.nf_CO2_psaL_out
    nf_CO_elec_in  =               (1 - purge_fraction) * psa.nf_CO_psaL_out

    H2 from the electrolyzer is NOT recycled — it passes through the PSA
    unadsorbed and is credited directly to the syngas product.

Design / optimisation variables
---------------------------------
    Electrolyzer : E_appl, v, n (number of cells / total_area)
    PSA          : p_H, p_L, mf_ads (adsorbent circulation rate)
    System       : n_CO2_fresh (fresh CO2 feed rate [mol/s])
    Optional     : purge_fraction (when float_purge=True)

Objective
---------
    minimise  total_lifecycle_cost = total_capex + plant_life * total_opex
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyomo.environ as pyo
from prettytable import PrettyTable

from configuration import Economics, Constants, H2Parameters
from SubModels.CO2_ElectrolyzerSurrogateModel import CO2Electrolyzer
from SubModels.H2_Electrolyzer_Model import H2Model
from SubModels.PSAModel import PSA
from SubModels.solver_utils import run_solver


class PSAFlowsheet(pyo.ConcreteModel):
    """
    See module docstring for full description.

    Parameters
    ----------
    purge_fraction : float
        Fraction of the PSA low-pressure desorption outlet (recycle stream)
        that is vented to flare rather than recycled [0, 1].
        Default 0.1  Used as initial guess when
        ``float_purge=True``.
    float_purge : bool
        If True, ``purge_fraction`` becomes a Pyomo Var (decision variable).
        Default False.
    plant_life : int
        Plant lifetime in years used for lifecycle cost calculation.
        Default 20.
    """

    def __init__(
        self,
        purge_fraction: float = 0.1,
        float_purge: bool = False,
        plant_life: int = 20,
    ):
        super().__init__()

        _econ = Economics()
        op_hours = _econ.operating_hours_per_year  # h/yr

        # ==================================================================
        # 1.  CO2 Electrolyzer
        # ==================================================================
        self.elec = CO2Electrolyzer()
        # No H2 recycle in this flowsheet — H2 goes straight to syngas.
        self.elec.nf_H2_elec_in.fix(0.0)

        # ==================================================================
        # 2.  PSA separation unit
        # ==================================================================
        self.psa = PSA(allow_vacuum=True)

        # ==================================================================
        # 3.  Connect electrolyzer outlet to PSA feed
        #
        #     H2 from the electrolyzer is NOT fed to the PSA — it passes
        #     directly to the syngas accounting.  Only CO2 and CO enter the PSA.
        # ==================================================================
        self.eq_psa_co2_feed = pyo.Constraint(
            expr=self.psa.nf_CO2_psaH_in == self.elec.nf_CO2_elec_out,
            doc="PSA CO2 feed = electrolyzer CO2 outlet [mol/s]",
        )
        self.eq_psa_co_feed = pyo.Constraint(
            expr=self.psa.nf_CO_psaH_in == self.elec.nf_CO_elec_out,
            doc="PSA CO feed = electrolyzer CO outlet [mol/s]",
        )

        # ==================================================================
        # 4.  Recycle mole balance  (with optional purge)
        #
        #     The PSA low-pressure desorption outlet (psaL_out) is CO2-rich
        #     and is recycled back to the electrolyzer feed.
        #     CO in psaL_out is also recycled (small amount).
        #     H2 is absent from the PSA model — it is never part of the recycle.
        #
        #     purge_fraction=0  → fully closed recycle.
        #     purge_fraction>0  → (1-purge_fraction) recycled; rest vented.
        # ==================================================================
        if float_purge:
            self.purge_fraction = pyo.Var(
                initialize=purge_fraction,
                bounds=(0.0, 1.0),
                doc="Fraction of recycle stream purged to flare [-] (decision variable)",
            )
        else:
            self.purge_fraction = pyo.Param(
                initialize=purge_fraction,
                doc="Fraction of recycle stream purged to flare [-]",
            )

        self.n_CO2_fresh = pyo.Var(
            initialize=50.0,
            bounds=(0.1, 1e5),
            doc="Fresh CO2 feed entering the system [mol/s]",
        )

        _recycle = 1.0 - self.purge_fraction

        self.eq_recycle_CO2 = pyo.Constraint(
            expr=self.elec.nf_CO2_elec_in == self.n_CO2_fresh + _recycle * self.psa.nf_CO2_psaL_out,
            doc="CO2 recycle balance: fresh feed + PSA desorption recycle",
        )
        self.eq_recycle_CO = pyo.Constraint(
            expr=self.elec.nf_CO_elec_in == _recycle * self.psa.nf_CO_psaL_out,
            doc="CO recycle balance: PSA desorption recycle only",
        )

        # ==================================================================
        # 5.  Economics aggregation  (all OPEX converted to €/yr)
        #
        #     PSA op_cost_psa covers compressor + vacuum pump electricity [€/h].
        #     Maintenance is added here as fraction of installed CAPEX per year.
        # ==================================================================
        self.total_capex = pyo.Var(
            initialize=2e6,
            bounds=(0.0, None),
            doc="Total plant CAPEX [€]",
        )
        self.total_opex = pyo.Var(
            initialize=5e5,
            bounds=(0.0, None),
            doc="Total plant annual OPEX [€/yr]",
        )

        # PSA maintenance: rotating equipment (comp + pump) + static (vessel)
        _maint_psa = (
            _econ.maintenance_factor_rotating * (self.psa.fixed_cost_comp + self.psa.fixed_cost_pump)
            + _econ.maintenance_factor_static * self.psa.fixed_cost_pv
        )

        self.eq_total_capex = pyo.Constraint(
            expr=self.total_capex == self.elec.CAPEX + self.psa.CAPEX,
            doc="Total CAPEX = electrolyzer + PSA (vessel + adsorbent + comp + pump) [€]",
        )
        self.eq_total_opex = pyo.Constraint(
            expr=self.total_opex
            == self.elec.OPEX * op_hours  # electrolyzer electricity + maintenance [€/yr]
            + self.psa.op_cost_psa * op_hours  # PSA electricity [€/yr]
            + _maint_psa,  # PSA maintenance [€/yr]
            doc="Total annual OPEX [€/yr]",
        )

        self.total_lifecycle_cost = pyo.Var(
            initialize=1e7,
            bounds=(0.0, None),
            doc="Discounted lifecycle cost = CAPEX + PVF × OPEX [€]",
        )
        # Discounted lifecycle cost: CAPEX + PVF × OPEX
        # PVF = (1-(1+r)^-n)/r at r=8.5%, n=20 → PVF=9.46  (vs undiscounted 20)
        self.eq_lifecycle_cost = pyo.Constraint(
            expr=self.total_lifecycle_cost == self.total_capex + _econ.pvf * self.total_opex,
        )

        # ==================================================================
        # 6.  Objective: minimise total lifecycle cost
        # ==================================================================
        self.obj = pyo.Objective(
            expr=self.total_lifecycle_cost,
            sense=pyo.minimize,
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

        The PSA high-pressure product (CO-rich) forms the syngas CO stream.
        H2 from the CO2 electrolyzer passes through the PSA unadsorbed and
        is credited to syngas.  The H2/CO ratio target is met by one of three
        sourcing strategies controlled by ``H2_type``:

        - ``"Electrolyzer"``   — use only the H2 already present in the
          electrolyzer outlet (produced as a by-product of CO2 reduction).
          No additional H2 equipment or cost is added.
        - ``"Buy"``            — supplement electrolyzer H2 with externally
          purchased H2.  A continuous variable ``nf_H2_buy`` [mol/s] is added
          and its cost appended to ``total_opex``.
        - ``"H2Electrolyzer"`` — supplement electrolyzer H2 with H2 produced
          on-site by a dedicated PEM/alkaline water electrolyzer (``H2Model``).
          Its CAPEX is added to ``total_capex`` and its annual electricity +
          maintenance costs are added to ``total_opex``.

        Calling this method also appends feedstock costs (CO2 fresh feed and,
        if applicable, purchased or produced H2) to ``total_opex`` so the
        lifecycle objective already includes them.

        Parameters
        ----------
        syngas_target : float
            Desired syngas mass flow rate [t/day].
        syngas_ratio : float
            Molar H2/CO ratio in the product syngas (e.g. 2 for FT synthesis).
        H2_type : str
            "Electrolyzer"   — use only electrolyzer by-product H2.
            "Buy"            — supplement with purchased H2.
            "H2Electrolyzer" — supplement with a dedicated water electrolyzer.
        price_CO2_feed : float
            Cost of the fresh CO2 feed stream [€/mol].
            Default: 0.05 €/kg × 44.01 g/mol.
        price_H2_buy : float
            Purchase price of H2 [€/mol]  (only used when H2_type="Buy").
            Default: 3 USD/kg × 0.96 USD→EUR × 2.016 g/mol.
        """
        if H2_type not in ("Electrolyzer", "Buy", "H2Electrolyzer"):
            raise ValueError("H2_type must be 'Electrolyzer', 'Buy', or 'H2Electrolyzer'")
        if hasattr(self, "_syngas_target_set"):
            raise RuntimeError("set_syngas_target() has already been called.")

        self._syngas_target_set = True
        self._H2_type = H2_type

        _econ = Economics()
        op_hours = _econ.operating_hours_per_year
        seconds_per_year = op_hours * 3600.0  # s/yr

        # ------------------------------------------------------------------
        # Back-calculate initial guesses from the syngas target
        # ------------------------------------------------------------------
        _mf_kgs = syngas_target * 1000.0 / (3600.0 * 24.0)  # kg/s
        _MW_sg = (syngas_ratio * Constants.MW_H2 + Constants.MW_CO) / (syngas_ratio + 1)
        _nf_CO_ig = _mf_kgs / _MW_sg / (syngas_ratio + 1)  # mol/s  CO
        _nf_H2_ig = syngas_ratio * _nf_CO_ig  # mol/s  H2 total

        _clip = lambda v, lo, hi: max(lo, min(hi, v))
        self.n_CO2_fresh.set_value(_clip(_nf_CO_ig * 1.5, 0.1, 1e4))
        # Pass CO target to initialize() so the electrolyzer is sized correctly
        self._nf_CO_init_target = _nf_CO_ig

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.syngas_ratio_param = pyo.Param(
            initialize=syngas_ratio,
            doc="H2/CO molar ratio in product syngas [-]",
        )
        _mf_target_kgs = syngas_target * 1000.0 / (3600.0 * 24.0)
        self.mf_syngas_target = pyo.Param(
            initialize=_mf_target_kgs,
            mutable=True,
            doc="Syngas mass flow target [kg/s]",
        )

        # ------------------------------------------------------------------
        # Syngas product variables  [mol/s]
        # ------------------------------------------------------------------
        self.nf_CO_syngas = pyo.Var(
            initialize=_nf_CO_ig,
            bounds=(1e-6, 1e4),
            doc="Syngas CO flow [mol/s]",
        )
        self.nf_H2_syngas = pyo.Var(
            initialize=_nf_H2_ig,
            bounds=(1e-6, 1e4),
            doc="Syngas H2 flow [mol/s]",
        )
        self.nf_CO2_syngas = pyo.Var(
            initialize=1e-3,
            bounds=(1e-9, 1e4),
            doc="Syngas CO2 flow (trace from PSA product) [mol/s]",
        )
        self.nf_syngas = pyo.Var(
            initialize=_nf_CO_ig + _nf_H2_ig,
            bounds=(1e-6, 1e4),
            doc="Total syngas molar flow [mol/s]",
        )
        self.mf_syngas = pyo.Var(
            initialize=_mf_target_kgs,
            bounds=(1e-9, 1e3),
            doc="Syngas mass flow [kg/s]",
        )

        # ------------------------------------------------------------------
        # Syngas composition constraints
        #
        #   CO  comes from the PSA high-pressure product (psaH_out)
        #   CO2 trace comes from the same stream (impurity at 0.4 % by design)
        #   H2  comes from the electrolyzer cathode outlet directly
        # ------------------------------------------------------------------
        self.eq_CO_syngas = pyo.Constraint(
            expr=self.nf_CO_syngas == self.psa.nf_CO_psaH_out,
            doc="Syngas CO = PSA high-pressure CO product [mol/s]",
        )
        self.eq_CO2_syngas = pyo.Constraint(
            expr=self.nf_CO2_syngas == self.psa.nf_CO2_psaH_out,
            doc="Syngas CO2 = trace CO2 in PSA product [mol/s]",
        )

        # H2 sourcing
        if H2_type == "Electrolyzer":
            self.eq_H2_syngas = pyo.Constraint(
                expr=self.nf_H2_syngas == self.elec.nf_H2_elec_out,
                doc="Syngas H2 = electrolyzer by-product H2 only",
            )
        elif H2_type == "Buy":
            self.nf_H2_buy = pyo.Var(
                initialize=max(_nf_H2_ig - 1.0, 0.1),
                bounds=(0.0, 1e4),
                doc="Purchased H2 [mol/s]",
            )
            self.eq_H2_syngas = pyo.Constraint(
                expr=self.nf_H2_syngas == self.elec.nf_H2_elec_out + self.nf_H2_buy,
            )
        else:  # "H2Electrolyzer"
            _nf_H2_supplement = max(_nf_H2_ig - 1.0, 0.1)
            _kwh_per_kg = H2Parameters.actual_kwh_per_kg_h2
            _mw_h2 = Constants.MW_H2
            _power_guess_mw = max(_nf_H2_supplement * _mw_h2 * _kwh_per_kg * 3600.0 / 1000.0, 10.0)
            self.h2_elec = H2Model()
            self.h2_elec.power_hp_mw.setlb(1.0)
            self.h2_elec.power_hp_mw.set_value(max(_power_guess_mw, 1.0))

            for _c in (
                "total_h2_kwh_in_constraint",
                "total_h2_kg_constraint",
                "capex_for_1kg_h2_constraint",
                "annualized_capex_per_kg_h2_constraint",
                "total_annualized_cost_constraint",
                "cost_per_kg_h2_constraint",
            ):
                getattr(self.h2_elec, _c).deactivate()

            self.eq_H2_syngas = pyo.Constraint(
                expr=self.nf_H2_syngas == self.elec.nf_H2_elec_out + self.h2_elec.H2_out_mol_per_sec,
            )

        # H2/CO ratio
        self.eq_syngas_ratio = pyo.Constraint(
            expr=self.nf_H2_syngas == syngas_ratio * self.nf_CO_syngas,
        )

        # Total mole flow
        self.eq_nf_syngas = pyo.Constraint(
            expr=self.nf_syngas == self.nf_CO_syngas + self.nf_H2_syngas + self.nf_CO2_syngas,
        )

        # Mass flow [kg/s] — CO + H2 only; CO2 is an impurity, not counted
        self.eq_mf_syngas = pyo.Constraint(
            expr=self.mf_syngas == (self.nf_CO_syngas * Constants.MW_CO + self.nf_H2_syngas * Constants.MW_H2),
        )

        # Mass-flow target (equality — pin the production rate)
        self.eq_mf_syngas_target = pyo.Constraint(
            expr=self.mf_syngas == self.mf_syngas_target,
        )

        # ------------------------------------------------------------------
        # Feedstock costs  [€/yr]
        # ------------------------------------------------------------------
        self.price_CO2_feed_param = pyo.Param(
            initialize=price_CO2_feed,
            doc="CO2 fresh feed price [€/mol]",
        )
        self.cost_CO2_feed = pyo.Var(
            initialize=1e4,
            bounds=(0.0, None),
            doc="Annual CO2 feed cost [€/yr]",
        )
        self.eq_cost_CO2_feed = pyo.Constraint(
            expr=self.cost_CO2_feed == self.price_CO2_feed_param * self.n_CO2_fresh * seconds_per_year,
        )

        _feedstock_opex = self.cost_CO2_feed

        if H2_type == "Buy":
            self.price_H2_buy_param = pyo.Param(
                initialize=price_H2_buy,
                doc="Purchased H2 price [€/mol]",
            )
            self.cost_H2_buy = pyo.Var(
                initialize=1e4,
                bounds=(0.0, None),
                doc="Annual purchased H2 cost [€/yr]",
            )
            self.eq_cost_H2_buy = pyo.Constraint(
                expr=self.cost_H2_buy == self.price_H2_buy_param * self.nf_H2_buy * seconds_per_year,
            )
            _feedstock_opex = _feedstock_opex + self.cost_H2_buy

        # ------------------------------------------------------------------
        # Update total_opex to include feedstock costs (and H2 electrolyzer)
        # ------------------------------------------------------------------
        _maint_psa = (
            _econ.maintenance_factor_rotating * (self.psa.fixed_cost_comp + self.psa.fixed_cost_pump)
            + _econ.maintenance_factor_static * self.psa.fixed_cost_pv
        )
        _base_opex = self.elec.OPEX * op_hours + self.psa.op_cost_psa * op_hours + _maint_psa
        if H2_type == "H2Electrolyzer":
            _base_opex = _base_opex + self.h2_elec.OPEX + self.h2_elec.MAINTENANCE

        self.eq_total_opex.deactivate()
        self.eq_total_opex_syngas = pyo.Constraint(
            expr=self.total_opex == _base_opex + _feedstock_opex,
        )

        # ------------------------------------------------------------------
        # Update total_capex to include H2 electrolyzer CAPEX if used
        # ------------------------------------------------------------------
        if H2_type == "H2Electrolyzer":
            self.eq_total_capex.deactivate()
            self.eq_total_capex_syngas = pyo.Constraint(
                expr=self.total_capex == self.elec.CAPEX + self.psa.CAPEX + self.h2_elec.CAPEX,
            )

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def initialize(
        self,
        E_appl: float = -1.31,
        v: float = 0.08,
        CO2share: float = 0.80,
        p_H_kpa: float = 300.0,
        p_L_kpa: float = 10.0,
        co2_recovery: float = 0.95,
        nf_CO_out_target: float = None,
    ):
        """Seed all model variables to a self-consistent starting point.

        Evaluates the CO2 electrolyzer surrogate and the PSA Sips-isotherm
        mass balances in pure Python (no solver), then sets every Pyomo Var
        so that IPOPT's first Jacobian evaluation is numerically well-behaved.

        **Must be called before solve().**

        Parameters
        ----------
        E_appl : float
            Applied electrolyzer voltage [V], typically -1.2 to -1.5.
        v : float
            Gas flow velocity in the electrolyzer channel [m/s].
        CO2share : float
            Inlet CO2 mole fraction at cathode, CO2/(CO2+CO) [-].
        p_H_kpa : float
            PSA adsorption pressure [kPa].  Default 300 kPa (3 bar).
        p_L_kpa : float
            PSA desorption (vacuum) pressure [kPa].  Default 10 kPa.
        co2_recovery : float
            Target fraction of feed CO2 captured by the PSA (sent to
            desorption/recycle stream).  Used to back-calculate mf_ads.
            Default 0.95 (95 % CO2 recovery).
        nf_CO_out_target : float or None
            Target CO outlet flow [mol/s] used to scale all molar flows.
            Auto-detected from set_syngas_target() when previously called,
            otherwise falls back to 15.0 mol/s.
        """
        from math import exp as _exp, asinh as _asinh

        if nf_CO_out_target is None:
            nf_CO_out_target = getattr(self, "_nf_CO_init_target", 15.0)

        _P = 1e5  # Pa
        _R = 8.314  # J/(mol·K)
        _T = 298.0  # K
        _henry = 0.85

        c_total = _P / (_R * _T)  # ≈ 40.9 mol/m³

        # ---------------------------------------------------------------
        # 1.  Electrolyzer surrogate — evaluate in pure Python
        # ---------------------------------------------------------------
        # Guard CO2share against the CO_ineq constraint (c_out_CO >= c_in_CO)
        try:
            _denom_co = E_appl - v ** (-0.46066585)
            _base2_co = E_appl * -0.6278232
            _term2_co = _base2_co**21.189863 if _base2_co >= 0 else -((-_base2_co) ** 21.189863)
            _inner_co = 1.4358697 / _denom_co - _term2_co + E_appl * -1.1215197
            _c_co_unit = max(_inner_co, 1e-4) ** 10.42661
        except Exception:
            _c_co_unit = 1.5 * c_total
        _co2share_min = c_total / max(_c_co_unit + c_total, 1e-9)
        if CO2share < _co2share_min + 0.02:
            CO2share = min(_co2share_min + 0.03, 0.99)

        c_in_CO2 = c_total * CO2share
        c_in_CO = c_total * (1.0 - CO2share)

        c_out_CO2 = ((E_appl * (((v**0.25 + E_appl) * -3.7203176) - 1.3941393)) ** 3 + 40.244705) * CO2share
        c_out_CO2 = max(min(c_out_CO2, c_in_CO2 * 0.999), 1e-3)

        try:
            _denom = E_appl - v ** (-0.46066585)
            _base2 = E_appl * -0.6278232
            _term2 = _base2**21.189863 if _base2 >= 0 else -((-_base2) ** 21.189863)
            _inner = 1.4358697 / _denom - _term2 + E_appl * -1.1215197
            c_out_CO = CO2share * (max(_inner, 1e-4) ** 10.42661)
        except Exception:
            c_out_CO = c_in_CO * 1.5
        c_out_CO = max(c_out_CO, c_in_CO + 1e-3)

        _base_h2 = (E_appl**2 * 0.26091927) + 0.35308436
        c_out_H2 = (max(_base_h2, 1e-6) ** 14.994374) / v
        c_out_H2 = max(c_out_H2, 1e-3)

        c_mean_CO2 = (c_in_CO2 + c_out_CO2) / 2.0 * _henry

        vol_flow = nf_CO_out_target / max(c_out_CO, 1e-6)
        nf_CO2_out = c_out_CO2 * vol_flow
        nf_CO_out = c_out_CO * vol_flow
        nf_H2_out = c_out_H2 * vol_flow
        nf_CO2_in = c_in_CO2 * vol_flow
        nf_CO_in = c_in_CO * vol_flow

        # ---------------------------------------------------------------
        # 2.  Electrolyzer Pyomo variables
        # ---------------------------------------------------------------
        _BV_iCO = pyo.value(self.elec.BV_iCO)
        _BV_iH2 = pyo.value(self.elec.BV_iH2)
        _BV_CO = pyo.value(self.elec.BV_CO)
        _BV_H2 = pyo.value(self.elec.BV_H2)
        _BV_ECO = pyo.value(self.elec.BV_ECO)
        _BV_CO2ref = pyo.value(self.elec.BV_CO2ref)
        _E_anode = pyo.value(self.elec.E_anode)
        _Ch_H = pyo.value(self.elec.Channel_H)
        _Ch_W = pyo.value(self.elec.Channel_W)
        _sigma_el = pyo.value(self.elec.sigma_el)
        _Lm = pyo.value(self.elec.Lm)
        _sigma_m = pyo.value(self.elec.sigma_m)
        _r_area = pyo.value(self.elec.reacting_area)

        _F_const = 96485.0
        _CD = max(
            _BV_iCO * c_mean_CO2 / _BV_CO2ref * _exp(-(_BV_CO * _F_const / (_R * _T)) * (E_appl + _BV_ECO))
            + _BV_iH2 * _exp(-(_BV_H2 * _F_const / (_R * _T)) * E_appl),
            1.0,
        )
        _eta_actA = (_R * _T) / (0.5 * 96485.0) * _asinh(_CD / (2.0 * 1e-7))
        _eta_ohm = _CD * (_Ch_H / _sigma_el + _Lm / _sigma_m)
        _Vcell = _E_anode + _eta_actA + _BV_ECO - E_appl + _eta_ohm
        _n_cells = vol_flow / max(v * _Ch_H * _Ch_W, 1e-15)
        _tot_area = max(_n_cells * _r_area, 3000.0)
        _I = _CD * _tot_area
        _Power = _I * _Vcell  # W

        from configuration import Economics as _Econ

        _price_pw = _Econ.price_power
        _maint_f = pyo.value(self.elec.maint_factor)
        _b_cost = pyo.value(self.elec.base_cost_elec)  # €/m²
        _f_inst = pyo.value(self.elec.f_installation_elec)  # 5.04
        _repl_f_co2 = 1.0 + (1.0 + _Econ.interest_rate) ** (-10)
        _CAPEX_elec = _f_inst * _b_cost * _tot_area * _repl_f_co2  # installed € incl. replacement
        _OPEX_elec = _Power * _price_pw / 1000.0 + _maint_f * _CAPEX_elec / _Econ.operating_hours_per_year

        self.elec.E_appl.set_value(E_appl)
        self.elec.v.set_value(v)
        self.elec.CO2share.set_value(CO2share)
        self.elec.c_elec_in_CO2.set_value(c_in_CO2)
        self.elec.c_elec_in_CO.set_value(c_in_CO)
        self.elec.c_elec_out_CO2.set_value(c_out_CO2)
        self.elec.c_elec_out_CO.set_value(c_out_CO)
        self.elec.c_elec_out_H2.set_value(c_out_H2)
        self.elec.c_elec_mean_CO2.set_value(c_mean_CO2)
        self.elec.vol_gas_flow.set_value(vol_flow)
        self.elec.nf_CO2_elec_out.set_value(nf_CO2_out)
        self.elec.nf_CO_elec_out.set_value(nf_CO_out)
        self.elec.nf_H2_elec_out.set_value(nf_H2_out)
        self.elec.CD.set_value(max(_CD, 1.0))
        self.elec.eta_actA.set_value(_eta_actA)
        self.elec.eta_ohm.set_value(_eta_ohm)
        self.elec.Vcell.set_value(max(_Vcell, 2.5))
        self.elec.n.set_value(max(_n_cells / 1e6, 3.0))
        self.elec.total_area.set_value(max(_tot_area, 3000.0))
        self.elec.I.set_value(max(_I, 1.0))
        self.elec.Power.set_value(max(_Power, 1.0))
        self.elec.OPEX.set_value(max(_OPEX_elec, 1.0))
        self.elec.CAPEX.set_value(max(_CAPEX_elec, 1.0))

        _FE_val = (
            _BV_iCO
            * c_out_CO2
            * _henry
            / _BV_CO2ref
            * _exp(-(_BV_CO * 96485.0 / (_R * _T)) * (E_appl + _BV_ECO))
            / max(_CD, 1e-9)
            * 100.0
        )
        self.elec.FE.set_value(max(min(_FE_val, 100.0), 1e-6))
        _n_CO2_th = max(_I / (2.0 * 96485.0), 1e-6)
        self.elec.n_CO2_theoretical.set_value(_n_CO2_th)
        self.elec.lamda_CO2.set_value(max(min(nf_CO2_in / max(_n_CO2_th, 1e-9), 6.0), 1.0))

        # ---------------------------------------------------------------
        # 3.  PSA — Sips isotherms and mass balances in pure Python
        # ---------------------------------------------------------------
        from configuration import Constants as _Const, PSAParameters as _PSAPar

        _q_CO2_s = _Const.q_CO2_s
        _Bs_CO2 = _Const.Bs_CO2
        _ns_CO2 = _Const.ns_CO2
        _q_CO_s = _Const.q_CO_s
        _Bs_CO = _Const.Bs_CO
        _ns_CO = _Const.ns_CO
        _sat = _Const.saturation_adsorbent  # 0.95
        _rho = _Const.bulk_density  # kg/m³
        _voidage = _PSAPar.voidage  # 0.4
        _t_ads = pyo.value(self.psa.t_ads) or 500.0  # adsorption step time [s] (initial value)
        _t_des = pyo.value(self.psa.t_des) or 250.0  # desorption step time [s] (initial value)
        _t_overhead = 20.0  # valve switching overhead [s]

        # PSA feed mole fractions
        _nf_psa_in = nf_CO2_out + nf_CO_out
        _y_CO2_H = nf_CO2_out / max(_nf_psa_in, 1e-9)
        _y_CO_H = nf_CO_out / max(_nf_psa_in, 1e-9)

        # Assume desorption stream is CO2-rich
        _y_CO2_L = 0.90
        _y_CO_L = 0.10

        def _sips(q_s, Bs, ns, y, p):
            """Sips isotherm loading [mol/kg]."""
            other_Bs = _Bs_CO2 if Bs is _Bs_CO else _Bs_CO
            other_ns = _ns_CO2 if ns is _ns_CO else _ns_CO
            other_y = _y_CO_H if Bs is _Bs_CO2 else _y_CO2_H
            arg_self = (Bs * y * p) ** (1.0 / ns)
            arg_other = (other_Bs * other_y * p) ** (1.0 / other_ns)
            return q_s * arg_self / (1.0 + arg_self + arg_other)

        _q_CO2_H = _sips(_q_CO2_s, _Bs_CO2, _ns_CO2, _y_CO2_H, p_H_kpa)
        _q_CO_H = _sips(_q_CO_s, _Bs_CO, _ns_CO, _y_CO_H, p_H_kpa)
        _q_CO2_L = _sips(_q_CO2_s, _Bs_CO2, _ns_CO2, _y_CO2_L, p_L_kpa)
        _q_CO_L = _sips(_q_CO_s, _Bs_CO, _ns_CO, _y_CO_L, p_L_kpa)

        # Void-term contributions [mol/kg-ads]
        _void_CO2_H = _voidage * _y_CO2_H * p_H_kpa * 1000.0 / (_rho * _R * _T)
        _void_CO_H = _voidage * _y_CO_H * p_H_kpa * 1000.0 / (_rho * _R * _T)
        _void_CO2_L = _voidage * _y_CO2_L * p_L_kpa * 1000.0 / (_rho * _R * _T)
        _void_CO_L = _voidage * _y_CO_L * p_L_kpa * 1000.0 / (_rho * _R * _T)

        # mf_ads from CO2 mass balance:
        # nf_CO2_psaL_out = co2_recovery * nf_CO2_feed = mf_ads * [(q_CO2_H-q_CO2_L)*sat + void_diff]
        _co2_desorb_target = co2_recovery * nf_CO2_out
        _co2_spec = (_q_CO2_H - _q_CO2_L) * _sat + (_void_CO2_H - _void_CO2_L)
        _co2_spec = max(_co2_spec, 1e-6)
        _mf_ads = _co2_desorb_target / _co2_spec
        _mf_ads = max(_mf_ads, pyo.value(self.psa.mf_ads.lb or 10.0))

        # HR / LR values
        _nf_CO2_HR = _mf_ads * (_q_CO2_H * _sat + _void_CO2_H)
        _nf_CO_HR = _mf_ads * (_q_CO_H * _sat + _void_CO_H)
        _nf_CO2_LR = _mf_ads * (_q_CO2_L * _sat + _void_CO2_L)
        _nf_CO_LR = _mf_ads * (_q_CO_L * _sat + _void_CO_L)

        # Output flows
        # PSA product composition is fixed by the purity constraint (y_CO2_product = 0.02).
        # Using the surrogate mass balance directly gives inconsistent values when CO selectivity
        # is low (large CO2:CO ratio), so we pin the product CO2 to the purity-consistent value.
        _nf_CO_psaH_out = max(nf_CO_out - (_nf_CO_HR - _nf_CO_LR), 1e-3)
        _y_CO2_prod = _PSAPar.y_CO2_product  # 0.02
        _y_CO_prod = _PSAPar.y_CO_product  # 0.98
        _nf_CO2_psaH_out = max((_y_CO2_prod / _y_CO_prod) * _nf_CO_psaH_out, 1e-4)
        _nf_CO2_psaL_out = max(_nf_CO2_HR - _nf_CO2_LR, 1e-3)
        _nf_CO_psaL_out = max(_nf_CO_HR - _nf_CO_LR, 1e-3)

        _y_CO2_L_out = _nf_CO2_psaL_out / max(_nf_CO2_psaL_out + _nf_CO_psaL_out, 1e-9)
        _y_CO_L_out = _nf_CO_psaL_out / max(_nf_CO2_psaL_out + _nf_CO_psaL_out, 1e-9)

        # Adsorbent mass total: m_ads = mf_ads * t_cyc  (VMB link)
        _m_ads = _mf_ads * (_t_ads + _t_des + _t_overhead)

        # Compressor power (isothermal work, feed pressurisation atm→p_H)
        _p_atm = 100.0  # kPa (PSA model uses p_L as lower reference, but feed is from elec at ~1 bar)
        _power_comp_out = max(
            _nf_psa_in * _R * _T * max(0.0, __import__("math").log(p_H_kpa / _p_atm)) / 1000.0,
            0.1,
        )
        _eff_comp = _PSAPar.efficiency_comp
        _power_comp_in = _power_comp_out / _eff_comp

        # Vacuum pump power (desorption side)
        _nf_des = _nf_CO2_psaL_out + _nf_CO_psaL_out
        _power_pump_out = max(
            _nf_des * _R * _T * __import__("math").log(p_H_kpa / p_L_kpa) / 1000.0,
            0.1,
        )
        _eff_pump = _PSAPar.efficiency_pump
        _power_pump_in = _power_pump_out / _eff_pump

        _op_cost_psa = (_power_comp_in + _power_pump_in) * _Econ.price_power

        # Pressure vessel sizing
        _V_pv_min = _m_ads / _rho
        _Diameter = max(pyo.value(self.psa.Diameter.lb or 0.3048), 0.5)
        _Height = max(_V_pv_min / (__import__("math").pi * (_Diameter / 2.0) ** 2), 1.5)
        _Height = min(_Height, pyo.value(self.psa.Height.ub or 30.48))
        _V_pv = __import__("math").pi * (_Diameter**2) * _Height / 4.0

        from configuration import Economics as _Econ2

        _cepci_ratio = _Econ2.CEPCI_2024 / _Econ2.CEPCI_1974

        _fp = 2e-13 * p_H_kpa**3 - 1e-8 * p_H_kpa**2 + 0.0006 * p_H_kpa + 0.9081
        # Guthrie (1974) correlation — coefficient is 10^3 (kUSD), NOT 10^6.
        # The original bachelor thesis (Tjahjana, 2025) erroneously used 10^6,
        # inflating vessel CAPEX by a factor of 1000.
        _base_pv = 1e3 * (_Height / 1.2192) ** 0.81 * (_Diameter / 0.9144) ** 1.05
        _inst_pv = _PSAPar.factor_installation * _PSAPar.factor_material * _fp * _base_pv
        _fixed_pv = _Econ2.convert_usd_to_euro * _PSAPar.indirect_cost_factor_pv * _inst_pv * 2 * _cepci_ratio

        _price_ads = _m_ads * _PSAPar.price_zeoperkg

        # Compressor CAPEX
        _base_comp = 23000.0 * (_power_comp_out / 73.55) ** 0.77
        _Fa_comp = 776.5 * max(_base_comp, 1e-6) ** 0.493
        _inst_comp = _PSAPar.factor_installation_comp * (_base_comp + _Fa_comp)
        _fixed_comp = _Econ2.convert_usd_to_euro * _PSAPar.indirect_cost_factor_comp * _inst_comp * _cepci_ratio

        # Pump CAPEX — Turton et al. (2012) liquid ring vacuum pump correlation
        # log10(Cp_USD_2001) = K1 + K2*log10(W_kW) + K3*(log10(W_kW))^2
        _cepci_ratio_turton = _Econ2.CEPCI_2024 / _Econ2.CEPCI_2001
        _log10_pump_kw = __import__("math").log10(max(_power_pump_in, 1.0))
        _base_pump = 10.0 ** (
            _PSAPar.turton_K1_pump
            + _PSAPar.turton_K2_pump * _log10_pump_kw
            + _PSAPar.turton_K3_pump * _log10_pump_kw**2
        )
        _inst_pump = _PSAPar.turton_FBM_pump * _base_pump
        _fixed_pump = _Econ2.convert_usd_to_euro * _PSAPar.indirect_cost_factor_pump * _inst_pump * _cepci_ratio_turton

        _CAPEX_psa = _fixed_pv + _price_ads + _fixed_comp + _fixed_pump

        # ---------------------------------------------------------------
        # 4.  Set PSA Pyomo variables
        # ---------------------------------------------------------------
        self.psa.p_H.set_value(p_H_kpa)
        self.psa.p_L.set_value(p_L_kpa)
        self.psa.nf_CO2_psaH_in.set_value(nf_CO2_out)
        self.psa.nf_CO_psaH_in.set_value(nf_CO_out)
        self.psa.y_CO2_psaH_in.set_value(max(min(_y_CO2_H, 1.0 - 1e-6), 1e-6))
        self.psa.y_CO_psaH_in.set_value(max(min(_y_CO_H, 1.0 - 1e-6), 1e-6))
        self.psa.mf_ads.set_value(_mf_ads)
        self.psa.m_ads.set_value(max(_m_ads, 1.0))
        self.psa.q_CO2_H.set_value(max(_q_CO2_H, 1e-4))
        self.psa.q_CO_H.set_value(max(_q_CO_H, 1e-6))
        self.psa.q_CO2_L.set_value(max(_q_CO2_L, 1e-4))
        self.psa.q_CO_L.set_value(max(_q_CO_L, 1e-6))
        self.psa.nf_CO2_HR.set_value(max(_nf_CO2_HR, 1e-3))
        self.psa.nf_CO_HR.set_value(max(_nf_CO_HR, 1e-3))
        self.psa.nf_CO2_LR.set_value(max(_nf_CO2_LR, 1e-3))
        self.psa.nf_CO_LR.set_value(max(_nf_CO_LR, 1e-3))
        self.psa.nf_CO2_psaH_out.set_value(max(_nf_CO2_psaH_out, 1e-3))
        self.psa.nf_CO_psaH_out.set_value(max(_nf_CO_psaH_out, 1e-3))
        self.psa.nf_CO2_product.set_value(max(_nf_CO2_psaH_out, 1e-3))
        self.psa.nf_CO_product.set_value(max(_nf_CO_psaH_out, 1e-3))
        self.psa.nf_CO2_psaL_out.set_value(max(_nf_CO2_psaL_out, 1e-3))
        self.psa.nf_CO_psaL_out.set_value(max(_nf_CO_psaL_out, 1e-3))
        self.psa.y_CO2_psaL_out.set_value(max(min(_y_CO2_L_out, 1.0 - 1e-6), 1e-6))
        self.psa.y_CO_psaL_out.set_value(max(min(_y_CO_L_out, 1.0 - 1e-6), 1e-6))
        self.psa.V_pv_min.set_value(max(_V_pv_min, 0.01))
        self.psa.Diameter.set_value(_Diameter)
        self.psa.Height.set_value(_Height)
        self.psa.V_pv.set_value(max(_V_pv, 0.01))
        self.psa.factor_pressure.set_value(max(_fp, 1.0))
        self.psa.power_comp_out.set_value(max(_power_comp_out, 0.001))
        self.psa.power_comp_in.set_value(max(_power_comp_in, 0.001))
        self.psa.power_pump_out.set_value(max(_power_pump_out, 0.001))
        self.psa.power_pump_in.set_value(max(_power_pump_in, 0.001))
        self.psa.op_cost_psa.set_value(max(_op_cost_psa, 0.001))
        self.psa.price_adsorbent.set_value(max(_price_ads, 1.0))
        self.psa.base_cost_pv.set_value(max(_base_pv, 1.0))
        self.psa.installed_cost_pv.set_value(max(_inst_pv, 1.0))
        self.psa.fixed_cost_pv.set_value(max(_fixed_pv, 1.0))
        self.psa.base_cost_comp.set_value(max(_base_comp, 1.0))
        self.psa.F_a_cost_comp.set_value(max(_Fa_comp, 1.0))
        self.psa.installed_cost_comp.set_value(max(_inst_comp, 1.0))
        self.psa.fixed_cost_comp.set_value(max(_fixed_comp, 1.0))
        self.psa.base_cost_pump.set_value(max(_base_pump, 1.0))
        self.psa.installed_cost_pump.set_value(max(_inst_pump, 1.0))
        self.psa.fixed_cost_pump.set_value(max(_fixed_pump, 1.0))
        self.psa.CAPEX.set_value(max(_CAPEX_psa, 1.0))

        # New PSA model variables
        self.psa.saturation_ads_CO2.set_value(0.75)
        self.psa.saturation_ads_CO.set_value(0.85)
        self.psa.saturation_des_CO2.set_value(0.75)
        self.psa.saturation_des_CO.set_value(0.85)
        self.psa.t_ads.set_value(_t_ads)
        self.psa.t_des.set_value(_t_des)
        self.psa.N_trains.set_value(1.0)
        self.psa.factor_pressure_L.set_value(max(_fp, 1.0))
        self.psa.base_cost_pv_L.set_value(max(_base_pv, 1.0))
        self.psa.installed_cost_pv_L.set_value(max(_inst_pv, 1.0))
        self.psa.fixed_cost_pv_L.set_value(0.0)

        # ---------------------------------------------------------------
        # 5.  Recycle mole balance — close the loop
        #
        #     eq_recycle_CO2: nf_CO2_elec_in = n_CO2_fresh + (1-purge)*psaL_CO2
        #     eq_recycle_CO:  nf_CO_elec_in  = (1-purge)*psaL_CO
        # ---------------------------------------------------------------
        _pf = pyo.value(self.purge_fraction)
        _recycle = 1.0 - _pf
        _nf_CO2_recycle = _recycle * _nf_CO2_psaL_out
        _nf_CO_recycle = _recycle * _nf_CO_psaL_out

        _nf_CO2_elec_in = nf_CO2_in  # from surrogate
        _nf_fresh = max(_nf_CO2_elec_in - _nf_CO2_recycle, 0.1)
        self.n_CO2_fresh.set_value(_nf_fresh)
        self.elec.nf_CO2_elec_in.set_value(max(_nf_fresh + _nf_CO2_recycle, nf_CO2_in))
        self.elec.nf_CO_elec_in.set_value(max(_nf_CO_recycle, 1e-6))
        # Update CO2share to reflect actual recycle composition
        _nf_CO2_in_val = pyo.value(self.elec.nf_CO2_elec_in)
        _co2share_in = _nf_CO2_in_val / max(_nf_CO2_in_val + max(_nf_CO_recycle, 1e-9), 1e-9)
        self.elec.CO2share.set_value(max(min(_co2share_in, 0.9999), 0.01))
        self.elec.c_elec_in_CO.set_value(max(_nf_CO_recycle / max(vol_flow, 1e-9), 1e-9))

        # ---------------------------------------------------------------
        # 6.  Economics aggregation
        # ---------------------------------------------------------------
        _op_hours = _Econ.operating_hours_per_year
        _maint_psa = (
            _Econ.maintenance_factor_rotating * (_fixed_comp + _fixed_pump)
            + _Econ.maintenance_factor_static * _fixed_pv
        )
        _total_capex = _CAPEX_elec + _CAPEX_psa
        _total_opex = _OPEX_elec * _op_hours + _op_cost_psa * _op_hours + _maint_psa
        self.total_capex.set_value(max(_total_capex, 1.0))
        self.total_opex.set_value(max(_total_opex, 1.0))
        self.total_lifecycle_cost.set_value(max(_total_capex + 20.0 * _total_opex, 1.0))

        # ---------------------------------------------------------------
        # 7.  Syngas product variables (if set_syngas_target was called)
        # ---------------------------------------------------------------
        if hasattr(self, "_syngas_target_set"):
            _nf_CO_sg = max(_nf_CO_psaH_out, 1e-6)
            _nf_CO2_sg = max(_nf_CO2_psaH_out, 1e-9)
            _ratio = pyo.value(self.syngas_ratio_param)
            _nf_H2_sg = _ratio * _nf_CO_sg

            self.nf_CO_syngas.set_value(_nf_CO_sg)
            self.nf_CO2_syngas.set_value(_nf_CO2_sg)
            self.nf_H2_syngas.set_value(_nf_H2_sg)
            self.nf_syngas.set_value(_nf_CO_sg + _nf_H2_sg + _nf_CO2_sg)
            from configuration import Constants as _C

            _mf_sg = _nf_CO_sg * _C.MW_CO + _nf_H2_sg * _C.MW_H2 + _nf_CO2_sg * _C.MW_CO2
            self.mf_syngas.set_value(max(_mf_sg, 1e-9))

            if self._H2_type == "Buy":
                _h2_supplement = max(_nf_H2_sg - nf_H2_out, 0.0)
                self.nf_H2_buy.set_value(max(_h2_supplement, 1e-6))
            elif self._H2_type == "H2Electrolyzer":
                _h2_supplement = max(_nf_H2_sg - nf_H2_out, 0.0)
                _kwh_per_kg = pyo.value(self.h2_elec.actual_kwh_per_kg_h2)
                _mw_h2_const = pyo.value(self.h2_elec.MW_H2)
                _power_mw = max(_h2_supplement * _mw_h2_const * _kwh_per_kg * 3600.0 / 1000.0, 1.0)
                self.h2_elec.power_hp_mw.set_value(_power_mw)

    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------

    def solve(
        self,
        solver_name: str = "ipopt",
        show_log: bool = False,
        solver_options: dict = None,  # type: ignore[assignment]
        multistart: bool = False,
        multistart_kwargs: dict = None,
    ):
        """Solve the combined NLP.

        Parameters
        ----------
        solver_name : str
            Solver executable name (default "ipopt").
        show_log : bool
            Stream solver output to stdout.
        solver_options : dict, optional
            Override / extend any IPOPT option.
        multistart : bool
            If True, solve via ``pyomo.contrib.multistart`` (repeated IPOPT
            restarts, best objective kept) for a pseudo-global solution.
        multistart_kwargs : dict, optional
            Overrides for the multistart wrapper (e.g. iterations, strategy).
        """
        # --- Variable scaling (exported to IPOPT) ---
        if not hasattr(self, "scaling_factor"):
            self.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        sf = self.scaling_factor

        # Cost variables
        sf[self.total_lifecycle_cost] = 1e-7  # ~1e8 €
        sf[self.total_capex] = 1e-6  # ~1e7 €
        sf[self.total_opex] = 1e-5  # ~1e6 €/yr
        if hasattr(self, "cost_CO2_feed"):
            sf[self.cost_CO2_feed] = 1e-5
        if hasattr(self, "cost_H2_buy"):
            sf[self.cost_H2_buy] = 1e-5

        # Electrolyzer
        sf[self.elec.CAPEX] = 1e-6
        sf[self.elec.OPEX] = 1e-1  # €/h ~O(10)
        sf[self.elec.Power] = 1e-5  # W  ~O(1e5-1e7)
        sf[self.elec.I] = 1e-5  # A  ~O(1e5)
        sf[self.elec.n] = 1e-1  # millions of cells
        sf[self.elec.total_area] = 1e-4  # m² ~O(1e4)

        # PSA cost variables — scaling updated for corrected 10^3 Guthrie coefficient
        # base_cost_pv ~O(1e3-1e4 USD), installed ~O(1e4-1e5 USD), fixed ~O(1e4-1e5 EUR)
        sf[self.psa.CAPEX] = 1e-5
        sf[self.psa.base_cost_pv] = 1e-3
        sf[self.psa.installed_cost_pv] = 1e-4
        sf[self.psa.fixed_cost_pv] = 1e-4
        sf[self.psa.price_adsorbent] = 1e-4
        sf[self.psa.base_cost_comp] = 1e-4
        sf[self.psa.F_a_cost_comp] = 1e-4
        sf[self.psa.installed_cost_comp] = 1e-4
        sf[self.psa.fixed_cost_comp] = 1e-4
        sf[self.psa.base_cost_pump] = 1e-3
        sf[self.psa.installed_cost_pump] = 1e-3
        sf[self.psa.fixed_cost_pump] = 1e-3

        # PSA operating variables
        sf[self.psa.power_comp_out] = 1e-2  # kW ~O(100-10000)
        sf[self.psa.power_comp_in] = 1e-2
        sf[self.psa.power_pump_out] = 1e-1  # kW ~O(10-1000)
        sf[self.psa.power_pump_in] = 1e-1
        sf[self.psa.op_cost_psa] = 1e-1  # €/h ~O(100-1000)
        sf[self.psa.mf_ads] = 1e-2  # kg/s ~O(10-1000)
        sf[self.psa.m_ads] = 1e-3  # kg ~O(100-10000)
        sf[self.psa.p_H] = 1e-2  # kPa ~O(100-10000)
        sf[self.psa.p_L] = 1e-1  # kPa ~O(1-100)

        if hasattr(self, "h2_elec"):
            sf[self.h2_elec.CAPEX] = 1e-6
            sf[self.h2_elec.OPEX] = 1e-5
            sf[self.h2_elec.MAINTENANCE] = 1e-5
            sf[self.h2_elec.power_hp_mw] = 1e-1
            sf[self.h2_elec.x_hp] = 1e0
            sf[self.h2_elec.capex_electrolyzer_per_kw] = 1e-3
            sf[self.h2_elec.H2_out_kg_per_sec] = 1e1
            sf[self.h2_elec.H2_out_mol_per_sec] = 1e-1

        _defaults = {
            "nlp_scaling_method": "gradient-based",
            "max_iter": 5000,
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
    # Reporting
    # -----------------------------------------------------------------------

    def report(self):
        """Print a structured summary of the combined PSA flowsheet solution."""

        def _cv(x) -> float:
            v = pyo.value(x)
            return float(v) if v is not None else 0.0

        _econ = Economics()
        op_hours = _econ.operating_hours_per_year  # h/yr
        plant_life = _econ.plant_life_years

        sep = "=" * 68
        sep2 = "-" * 68

        print("\n" + sep)
        print(f"  PSA FLOWSHEET SUMMARY  [Electrolyzer -> PSA]")
        print(sep)

        # ==================================================================
        # ELECTROLYZER
        # ==================================================================
        print("\n[ELECTROLYZER]")
        print(f"  E_appl          : {_cv(self.elec.E_appl):.4f}  V")
        print(f"  CD              : {_cv(self.elec.CD):.2f}  A/m²" f"  ({_cv(self.elec.CD) * 0.1:.2f} mA/cm²)")
        print(f"  v               : {_cv(self.elec.v):.4f}  m/s")
        print(f"  n (cells)       : {_cv(self.elec.n) * 1e6:.3e}")
        print(f"  Vcell           : {_cv(self.elec.Vcell):.4f}  V")
        print(f"  Power           : {_cv(self.elec.Power) / 1e6:.3f}  MW")
        print(f"  CO2share_in     : {_cv(self.elec.CO2share):.4f}  -")
        print(f"  lamda_CO2       : {_cv(self.elec.lamda_CO2):.4f}  -")
        print(f"  total_area      : {_cv(self.elec.total_area):,.1f}  m²")
        print(sep2)
        print(f"  n_CO2_fresh     : {_cv(self.n_CO2_fresh):.3f}  mol/s  (fresh feed)")
        print(f"  nf_CO2_in       : {_cv(self.elec.nf_CO2_elec_in):.3f}  mol/s  (fresh + recycle)")
        print(f"  nf_CO_in        : {_cv(self.elec.nf_CO_elec_in):.3f}  mol/s  (recycle only)")
        print(f"  nf_CO2_out      : {_cv(self.elec.nf_CO2_elec_out):.3f}  mol/s")
        print(f"  nf_CO_out       : {_cv(self.elec.nf_CO_elec_out):.3f}  mol/s")
        print(f"  nf_H2_out       : {_cv(self.elec.nf_H2_elec_out):.3f}  mol/s  (-> syngas directly)")
        _nf_total_out = _cv(self.elec.nf_CO2_elec_out) + _cv(self.elec.nf_CO_elec_out) + _cv(self.elec.nf_H2_elec_out)
        print(f"  Total cathode   : {_nf_total_out:.3f}  mol/s")
        print(sep2)
        print(f"  CAPEX           : €{_cv(self.elec.CAPEX):,.0f}")
        print(f"  OPEX            : €{_cv(self.elec.OPEX) * op_hours:,.0f}/yr")

        # ==================================================================
        # PSA
        # ==================================================================
        _pf_label = " (optimised)" if isinstance(self.purge_fraction, pyo.ScalarVar) else " (fixed)"
        _nf_psa_in = _cv(self.psa.nf_CO2_psaH_in) + _cv(self.psa.nf_CO_psaH_in)
        _nf_psa_out = _cv(self.psa.nf_CO2_psaH_out) + _cv(self.psa.nf_CO_psaH_out)
        _nf_des_out = _cv(self.psa.nf_CO2_psaL_out) + _cv(self.psa.nf_CO_psaL_out)
        _co_purity = _cv(self.psa.nf_CO_psaH_out) / (_nf_psa_out + 1e-12) * 100.0
        _co2_recovery = (
            (_cv(self.psa.nf_CO2_psaH_in) - _cv(self.psa.nf_CO2_psaH_out))
            / (_cv(self.psa.nf_CO2_psaH_in) + 1e-12)
            * 100.0
        )

        print("\n[PSA UNIT]")
        print(f"  Operating pressures:")
        print(f"    p_H (adsorption)   : {_cv(self.psa.p_H):.2f}  kPa")
        print(f"    p_L (desorption)   : {_cv(self.psa.p_L):.2f}  kPa")
        print(f"    Pressure ratio p_H/p_L : {_cv(self.psa.p_H) / (_cv(self.psa.p_L) + 1e-12):.2f}  -")
        print(sep2)
        print(f"  Adsorbent:")
        print(f"    mf_ads             : {_cv(self.psa.mf_ads):.3f}  kg/s")
        print(f"    m_ads (total)      : {_cv(self.psa.m_ads):.1f}  kg")
        print(f"    t_ads (half-cycle) : {_cv(self.psa.t_ads):.2f}  s")
        print(f"    q_CO2_H            : {_cv(self.psa.q_CO2_H):.4f}  mol/kg  (high-P loading)")
        print(f"    q_CO2_L            : {_cv(self.psa.q_CO2_L):.4f}  mol/kg  (low-P loading)")
        print(f"    q_CO_H             : {_cv(self.psa.q_CO_H):.4f}  mol/kg  (high-P loading)")
        print(f"    q_CO_L             : {_cv(self.psa.q_CO_L):.4f}  mol/kg  (low-P loading)")
        print(sep2)
        print(f"  Feed (psaH_in):")
        print(
            f"    nf_CO2             : {_cv(self.psa.nf_CO2_psaH_in):.3f}  mol/s"
            f"  (y_CO2 = {_cv(self.psa.y_CO2_psaH_in):.4f})"
        )
        print(
            f"    nf_CO              : {_cv(self.psa.nf_CO_psaH_in):.3f}  mol/s"
            f"  (y_CO  = {_cv(self.psa.y_CO_psaH_in):.4f})"
        )
        print(f"    Total              : {_nf_psa_in:.3f}  mol/s")
        print(f"  Product (psaH_out — CO-rich):")
        print(f"    nf_CO              : {_cv(self.psa.nf_CO_psaH_out):.3f}  mol/s")
        print(f"    nf_CO2             : {_cv(self.psa.nf_CO2_psaH_out):.3f}  mol/s")
        print(f"    Total              : {_nf_psa_out:.3f}  mol/s")
        print(f"    CO purity          : {_co_purity:.2f}  %")
        print(f"    CO2 captured       : {_co2_recovery:.2f}  %  of feed CO2")
        print(f"  Desorption (psaL_out — CO2-rich -> recycle):")
        print(f"    nf_CO2             : {_cv(self.psa.nf_CO2_psaL_out):.3f}  mol/s")
        print(f"    nf_CO              : {_cv(self.psa.nf_CO_psaL_out):.3f}  mol/s")
        print(f"    Total              : {_nf_des_out:.3f}  mol/s")
        print(f"    y_CO2              : {_cv(self.psa.y_CO2_psaL_out):.4f}  -")
        print(sep2)
        print(f"  Pressure vessel:")
        print(f"    Diameter           : {_cv(self.psa.Diameter):.3f}  m")
        print(f"    Height             : {_cv(self.psa.Height):.3f}  m")
        print(f"    Volume (pv)        : {_cv(self.psa.V_pv):.3f}  m³")
        print(f"    F_pressure         : {_cv(self.psa.factor_pressure):.4f}  -")
        print(f"  Compressor (feed pressurisation):")
        print(f"    Power_out          : {_cv(self.psa.power_comp_out):.2f}  kW")
        print(f"    Power_in           : {_cv(self.psa.power_comp_in):.2f}  kW  (shaft, incl. η)")
        print(f"  Vacuum pump (desorption):")
        print(f"    Power_out          : {_cv(self.psa.power_pump_out):.2f}  kW")
        print(f"    Power_in           : {_cv(self.psa.power_pump_in):.2f}  kW  (shaft, incl. η)")
        print(sep2)
        print(f"  CAPEX (total PSA)   : €{_cv(self.psa.CAPEX):,.0f}")
        print(f"    Pressure vessel   : €{_cv(self.psa.fixed_cost_pv):,.0f}")
        print(f"    Adsorbent         : €{_cv(self.psa.price_adsorbent):,.0f}")
        print(f"    Compressor        : €{_cv(self.psa.fixed_cost_comp):,.0f}")
        print(f"    Vacuum pump       : €{_cv(self.psa.fixed_cost_pump):,.0f}")
        print(f"  OPEX (electricity)  : €{_cv(self.psa.op_cost_psa) * op_hours:,.0f}/yr")

        # ==================================================================
        # RECYCLE
        # ==================================================================
        print("\n[RECYCLE]")
        print(f"  purge_fraction  : {_cv(self.purge_fraction):.4f}{_pf_label}")
        _recycled_co2 = _cv(self.psa.nf_CO2_psaL_out) * (1.0 - _cv(self.purge_fraction))
        _recycled_co = _cv(self.psa.nf_CO_psaL_out) * (1.0 - _cv(self.purge_fraction))
        _purged_co2 = _cv(self.psa.nf_CO2_psaL_out) * _cv(self.purge_fraction)
        _purged_co = _cv(self.psa.nf_CO_psaL_out) * _cv(self.purge_fraction)
        print(f"  CO2 recycled    : {_recycled_co2:.3f}  mol/s")
        print(f"  CO  recycled    : {_recycled_co:.3f}  mol/s")
        if _cv(self.purge_fraction) > 0:
            print(f"  CO2 purged      : {_purged_co2:.3f}  mol/s  (vented)")
            print(f"  CO  purged      : {_purged_co:.3f}  mol/s  (vented)")

        # ==================================================================
        # SYNGAS PRODUCT  (only if set_syngas_target was called)
        # ==================================================================
        if hasattr(self, "_syngas_target_set"):
            _nf_CO = _cv(self.nf_CO_syngas)
            _nf_H2 = _cv(self.nf_H2_syngas)
            _nf_CO2 = _cv(self.nf_CO2_syngas)
            _nf_tot = _cv(self.nf_syngas)
            _mf_kgs = _cv(self.mf_syngas)
            _mf_tpd = _mf_kgs * 3600.0 * 24.0 / 1000.0
            _tgt_tpd = _cv(self.mf_syngas_target) * 3600.0 * 24.0 / 1000.0
            _h2_co_actual = _nf_H2 / (_nf_CO + 1e-12)

            print("\n[SYNGAS PRODUCT]")
            print(f"  H2 source       : {self._H2_type}")
            print(f"  H2/CO target    : {_cv(self.syngas_ratio_param):.2f}  mol/mol")
            print(f"  H2/CO actual    : {_h2_co_actual:.4f}  mol/mol")
            print(sep2)
            print(f"  nf_CO           : {_nf_CO:.3f}  mol/s  (from PSA product)")
            print(
                f"  nf_H2           : {_nf_H2:.3f}  mol/s  (from electrolyzer{' + supplement' if self._H2_type != 'Electrolyzer' else ''})"
            )
            print(f"  nf_CO2 (trace)  : {_nf_CO2:.4f}  mol/s")
            print(f"  Total mole flow : {_nf_tot:.3f}  mol/s")
            _co2_tpd = _nf_CO2 * Constants.MW_CO2 * 3600 * 24 / 1000
            print(f"  Mass flow       : {_mf_tpd:.3f}  t/day  (CO+H2 only)")
            print(f"  CO2 impurity    : {_co2_tpd:.3f}  t/day  (not counted in target)")
            print(f"  Target mass flow: {_tgt_tpd:.3f}  t/day")
            print(f"  Deviation       : {abs(_mf_tpd - _tgt_tpd) / (_tgt_tpd + 1e-12) * 100:.2f}  %")

            if self._H2_type == "Buy":
                print(sep2)
                print(f"  nf_H2_buy       : {_cv(self.nf_H2_buy):.3f}  mol/s")
                print(f"  Annual H2 cost  : €{_cv(self.cost_H2_buy):,.0f}/yr")

            elif self._H2_type == "H2Electrolyzer":
                print(sep2)
                print("\n[H2 WATER ELECTROLYZER]")
                print(f"  Power           : {_cv(self.h2_elec.power_hp_mw):.2f}  MW")
                print(f"  H2 produced     : {_cv(self.h2_elec.H2_out_mol_per_sec):.3f}  mol/s")
                _h2_tpd = _cv(self.h2_elec.H2_out_kg_per_sec) * 3600.0 * 24.0 / 1000.0
                print(f"  H2 produced     : {_h2_tpd:.3f}  t/day")
                print(f"  CAPEX/kW        : €{_cv(self.h2_elec.capex_electrolyzer_per_kw):.0f}/kW")
                print(f"  CAPEX           : €{_cv(self.h2_elec.CAPEX):,.0f}")
                print(f"  OPEX            : €{_cv(self.h2_elec.OPEX):,.0f}/yr")
                print(f"  Maintenance     : €{_cv(self.h2_elec.MAINTENANCE):,.0f}/yr")

            print(f"  Annual CO2 cost : €{_cv(self.cost_CO2_feed):,.0f}/yr")

        # ==================================================================
        # COST BREAKDOWN TABLES
        # ==================================================================
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

        _maint_psa = _econ.maintenance_factor_rotating * (
            _cv(self.psa.fixed_cost_comp) + _cv(self.psa.fixed_cost_pump)
        ) + _econ.maintenance_factor_static * _cv(self.psa.fixed_cost_pv)

        capex_rows = [
            ("CO2 Electrolyzer", _cv(self.elec.CAPEX)),
            ("PSA pressure vessel", _cv(self.psa.fixed_cost_pv)),
            ("PSA adsorbent", _cv(self.psa.price_adsorbent)),
            ("PSA compressor", _cv(self.psa.fixed_cost_comp)),
            ("PSA vacuum pump", _cv(self.psa.fixed_cost_pump)),
        ]
        if hasattr(self, "_syngas_target_set") and self._H2_type == "H2Electrolyzer":
            capex_rows.append(("H2 Electrolyzer", _cv(self.h2_elec.CAPEX)))

        opex_rows = [
            ("CO2 Electrolyzer electricity+maint.", _cv(self.elec.OPEX) * op_hours),
            ("PSA electricity (comp+pump)", _cv(self.psa.op_cost_psa) * op_hours),
            ("PSA maintenance", _maint_psa),
        ]
        if hasattr(self, "_syngas_target_set"):
            opex_rows.append(("CO2 fresh feed cost", _cv(self.cost_CO2_feed)))
            if self._H2_type == "Buy":
                opex_rows.append(("H2 purchase cost", _cv(self.cost_H2_buy)))
            elif self._H2_type == "H2Electrolyzer":
                opex_rows.append(
                    (
                        "H2 Electrolyzer electricity+maint.",
                        (_cv(self.h2_elec.OPEX) or 0.0) + (_cv(self.h2_elec.MAINTENANCE) or 0.0),
                    )
                )

        _cost_table("CAPEX BREAKDOWN", capex_rows, "€")
        _cost_table("OPEX BREAKDOWN  (€/yr)", opex_rows, "€/yr")

        # ==================================================================
        # TOTALS
        # ==================================================================
        print("\n" + sep)
        print(f"{'COMBINED FLOWSHEET TOTALS':^68}")
        print(sep)
        _total_capex = _cv(self.total_capex)
        _total_opex = _cv(self.total_opex)
        _total_lc = _cv(self.total_lifecycle_cost)
        print(f"  Total CAPEX                    : €{_total_capex:,.0f}")
        print(f"  Total Annual OPEX              : €{_total_opex:,.0f}/yr")
        print(
            f"  NPV Cost ({plant_life} yr, {_econ.interest_rate*100:.1f}%)          : €{_total_lc:,.0f}  [CAPEX + PVF×OPEX, PVF={_econ.pvf:.2f}]"
        )
        if hasattr(self, "_syngas_target_set"):
            _mf_tpd = _cv(self.mf_syngas) * 3600.0 * 24.0 / 1000.0
            if _mf_tpd > 0:
                _lc_per_t = _total_lc / (_mf_tpd * 365.0 * plant_life)
                print(f"  Lifecycle cost / syngas        : €{_lc_per_t:,.2f}/t")
        print(sep + "\n")
