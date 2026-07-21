import pyomo.environ as pyo
from configuration import Economics
from prettytable import PrettyTable

# Unit conversion: 1 Barrer = 3.346e-16 mol·m/(m²·s·Pa)
BARRER = 3.346e-16  # mol·m/(m²·s·Pa)


class GasPermeationMembrane(pyo.ConcreteModel):
    """
    Discretized gas permeation membrane model using the solution-diffusion model (SDM).

    Reference:
        Heßelmann et al. (2023), Adv. Sustainable Syst. 2300077, Eq. 9
        Model structure adapted from Scholz et al. (2015), J. Membr. Sci. 474, 1

    Separates CO2/CO/H2 from the cathodic outlet stream of the CO2 electrolyzer.
    Cross-flow configuration, discretized along the membrane length into N segments.

    SDM flux (Eq. 9):
        Ji [mol/s] = (Q_i / delta) × (yr,i × p_r - yp,i × p_p) × (A_mem / N)
    where:
        Q_i   : permeability of species i [mol·m/(m²·s·Pa)]
        delta : selective layer thickness [m], assumed 0.5 µm for all materials
        yr,i  : retentate mole fraction of i [-]
        yp,i  : permeate mole fraction of i  [-]
        p_r   : retentate pressure [Pa]
        p_p   : permeate pressure  [Pa]
        A_mem : total membrane area [m²]
        N     : number of discretization segments

    Permeate composition (cross-flow):
        yp,i,k × (J_CO2,k + J_CO,k + J_H2,k) = J_i,k

    Capital cost (Eq. 14):
        CAPEX = BC_membrane × A_mem

    Inputs (fix or connect from upstream model):
        nf_r_CO2[1], nf_r_CO[1], nf_r_H2[1]  – feed molar flows  [mol/s]
        p_r                                    – retentate pressure [Pa]
        p_p                                    – permeate pressure  [Pa]

    Outputs:
        nf_r_CO2[N+1], nf_r_CO[N+1], nf_r_H2[N+1]  – retentate outlet [mol/s]
        nf_p_CO2_out, nf_p_CO_out, nf_p_H2_out       – permeate outlet  [mol/s]
    """

    # Permeabilities [Barrer] for each membrane material
    # Permeance [mol/(m²·s·Pa)] = P [Barrer] × BARRER / delta
    # Selective layer thickness delta = 0.5 µm assumed for all (Heßelmann et al.)
    #
    # Values taken directly from Heßelmann et al. (2023) Supporting Information, Table S1:
    #
    # Polyimide: H2=50, CO=1, CO2=13  → CO2/CO=13, H2/CO2=3.8 (glassy: diffusivity-dominated)
    #   Source: Aitken et al. (1992), Macromolecules 25, 3424 [ref. 2 in SI]
    # PEBAX:    H2=32.11, CO=1.39, CO2=55.85 → CO2/CO=40, CO2/H2=1.74 (rubbery: solubility-dominated)
    #   Source: David et al. (2011), J. Membr. Sci. 378, 359 [ref. 3 in SI]
    # Matrimid: H2=13.53, CO=3.46, CO2=0.25 → H2/CO2=54 (H2-selective glassy)
    #   Source: [ref. 4 in SI]
    # PRISM:    H2=14, CO=5.6, CO2=0.25 → H2/CO2=56 (H2-selective)
    #   Source: [ref. 5 in SI]
    MATERIAL_PERMEABILITIES = {
        "Polyimide": {"CO2": 13.0, "CO": 1.0, "H2": 50.0},
        "PEBAX": {"CO2": 55.85, "CO": 1.39, "H2": 32.11},
        "Matrimid": {"CO2": 0.25, "CO": 3.46, "H2": 13.53},
        "PRISM": {"CO2": 0.25, "CO": 5.6, "H2": 14.0},
    }

    def __init__(self, N_segments: int = 5, membrane_material: str = "Polyimide", co2_depletion_target: float = None):
        super().__init__()

        if membrane_material not in self.MATERIAL_PERMEABILITIES:
            raise ValueError(f"membrane_material must be one of {list(self.MATERIAL_PERMEABILITIES.keys())}")

        self._N = N_segments
        self.membrane_material = membrane_material
        self._co2_depletion_target = co2_depletion_target

        self.build_sets()
        self.build_parameters()
        self.build_variables()
        self.build_constraints()

    def build_sets(self):
        self.segments = pyo.RangeSet(1, self._N)  # discretization segments k = 1…N
        self.flow_nodes = pyo.RangeSet(1, self._N + 1)  # retentate flow nodes   k = 1…N+1

    def build_parameters(self):
        P = self.MATERIAL_PERMEABILITIES[self.membrane_material]

        # Number of discretization segments
        self.N_seg = pyo.Param(initialize=self._N)

        # Selective layer thickness δ [m]
        # Heßelmann et al.: "δ is assumed to be 0.5 µm for all membrane materials"
        self.delta = pyo.Param(initialize=0.5e-6)  # m

        # Permeabilities Q_i [mol·m/(m²·s·Pa)]
        # Permeance = Q_i / delta → used in SDM flux equation
        self.Q_CO2 = pyo.Param(initialize=P["CO2"] * BARRER)  # mol·m/(m²·s·Pa)
        self.Q_CO = pyo.Param(initialize=P["CO"] * BARRER)
        self.Q_H2 = pyo.Param(initialize=P["H2"] * BARRER)

        _econ = Economics()

        # Area-specific purchased equipment cost of the membrane module [€/m²]
        # (= K_B level — delivered module cost, NOT installed).
        # Source: Li Zhao et al., Multi-stage gas separation membrane processes
        # used in post-combustion capture, Energetic and economic analyses.
        self.BC_membrane = pyo.Param(initialize=50.0)  # €/m²

        # Lang/Hand installation factor K_B → K_I for membrane modules.
        # Membrane modules are pre-assembled modular units → treated as
        # "other machines" per Mitsos VL12 slide 30: f_H = 2.5.
        # The ×2 Guthrie overhead factor is applied in CAPEX_def below,
        # giving effective installed+overhead cost = 2 × 2.5 × 50 = 250 €/m².
        self.f_H_membrane = pyo.Param(initialize=2.5)
        self.base_frame_cost = pyo.Param(
            initialize=250000.0
        )  # € for the frame supporting the membrane, from my own assumptions and rough estimates based on some online sources for membrane module costs (e.g., https://www.airliquide.com/membrane-gas-separation-modules) and the fact that the frame cost should be significantly lower than the membrane cost (since the membrane is the main cost driver here)

        # Membrane lifetime for replacement cost calculation [years]
        # Heßelmann et al.: "lifetime of 5 years was assumed"
        self.t_lifetime_membrane = pyo.Param(initialize=5.0)  # years

        # Plant lifetime [years] — from Economics
        self.t_plant_life = pyo.Param(initialize=_econ.plant_life_years)

        # Annual maintenance cost as fraction of CAPEX [—] — from Economics
        self.maint_factor = pyo.Param(initialize=_econ.maintenance_factor_membrane)

        self.CEPCI_scaling = pyo.Param(
            initialize=_econ.CEPCI_2024 / _econ.CEPCI_1993
        )  # dimensionless scaling factor to update cost data from 1993 to 2024 prices

    def build_variables(self):
        # Membrane area [m²]
        self.A_mem = pyo.Var(initialize=5000.0, bounds=(1.0, 1e7))  # m²

        # Feed-side (retentate) pressure [Pa]
        self.p_r = pyo.Var(initialize=10e5, bounds=(1e5, 200e5))  # Pa

        # Permeate-side pressure [Pa]
        self.p_p = pyo.Var(initialize=1e5, bounds=(1e3, 20e5))  # Pa

        # Retentate molar flows at each flow node [mol/s]
        # Node 1 = feed inlet; Node N+1 = retentate outlet
        self.nf_r_CO2 = pyo.Var(self.flow_nodes, initialize=100.0, bounds=(1e-6, 1e6))
        self.nf_r_CO = pyo.Var(self.flow_nodes, initialize=50.0, bounds=(1e-6, 1e6))
        self.nf_r_H2 = pyo.Var(self.flow_nodes, initialize=10.0, bounds=(1e-6, 1e6))

        # Retentate molar fractions at each segment [dimensionless]
        self.y_r_CO2 = pyo.Var(self.segments, initialize=0.65, bounds=(1e-8, 1.0))
        self.y_r_CO = pyo.Var(self.segments, initialize=0.30, bounds=(1e-8, 1.0))
        self.y_r_H2 = pyo.Var(self.segments, initialize=0.05, bounds=(1e-8, 1.0))

        # Local permeate molar fractions at each segment (cross-flow) [dimensionless]
        self.y_p_CO2 = pyo.Var(self.segments, initialize=0.90, bounds=(1e-8, 1.0))
        self.y_p_CO = pyo.Var(self.segments, initialize=0.07, bounds=(1e-8, 1.0))
        self.y_p_H2 = pyo.Var(self.segments, initialize=0.03, bounds=(1e-8, 1.0))

        # Transmembrane molar fluxes per segment [mol/s]
        self.J_CO2 = pyo.Var(self.segments, initialize=10.0, bounds=(1e-9, 1e6))
        self.J_CO = pyo.Var(self.segments, initialize=0.50, bounds=(1e-9, 1e6))
        self.J_H2 = pyo.Var(self.segments, initialize=0.20, bounds=(1e-9, 1e6))

        # Total permeate outlet flows [mol/s]
        self.nf_p_CO2_out = pyo.Var(initialize=80.0, bounds=(1e-6, 1e6))
        self.nf_p_CO_out = pyo.Var(initialize=5.0, bounds=(1e-6, 1e6))
        self.nf_p_H2_out = pyo.Var(initialize=2.0, bounds=(1e-6, 1e6))

        # Stage cut: fraction of total feed that permeates [dimensionless]
        self.stage_cut = pyo.Var(initialize=0.50, bounds=(0.01, 0.99))

        # Capital cost [€] and replacement OPEX [€/h]
        self.CAPEX = pyo.Var(initialize=2.5e5, bounds=(1.0, 1e12))  # €
        self.OPEX = pyo.Var(initialize=1.0, bounds=(1e-3, 1e9))  # €/h
        self.CAPEX_membrane = pyo.Var(initialize=1.0, bounds=(1e-3, 1e9))  # €
        self.CAPEX_frame = pyo.Var(initialize=1.0, bounds=(1e-3, 1e9))  # €

    def build_constraints(self):
        N = self._N

        # --- Retentate mole fractions: y_r,i × N_total = N_i ---
        def y_r_CO2_def(m, k):
            return m.y_r_CO2[k] * (m.nf_r_CO2[k] + m.nf_r_CO[k] + m.nf_r_H2[k]) == m.nf_r_CO2[k]

        def y_r_CO_def(m, k):
            return m.y_r_CO[k] * (m.nf_r_CO2[k] + m.nf_r_CO[k] + m.nf_r_H2[k]) == m.nf_r_CO[k]

        def y_r_H2_def(m, k):
            return m.y_r_H2[k] * (m.nf_r_CO2[k] + m.nf_r_CO[k] + m.nf_r_H2[k]) == m.nf_r_H2[k]

        self.y_r_CO2_def = pyo.Constraint(self.segments, rule=y_r_CO2_def)
        self.y_r_CO_def = pyo.Constraint(self.segments, rule=y_r_CO_def)
        self.y_r_H2_def = pyo.Constraint(self.segments, rule=y_r_H2_def)

        # --- Solution-diffusion model flux (Heßelmann et al. 2023, Eq. 9) ---
        # Ji [mol/s] = (Q_i / delta) × (yr,i × p_r − yp,i × p_p) × (A_mem / N)
        def J_CO2_def(m, k):
            return m.J_CO2[k] == (m.Q_CO2 / m.delta) * (m.y_r_CO2[k] * m.p_r - m.y_p_CO2[k] * m.p_p) * m.A_mem / N

        def J_CO_def(m, k):
            return m.J_CO[k] == (m.Q_CO / m.delta) * (m.y_r_CO[k] * m.p_r - m.y_p_CO[k] * m.p_p) * m.A_mem / N

        def J_H2_def(m, k):
            return m.J_H2[k] == (m.Q_H2 / m.delta) * (m.y_r_H2[k] * m.p_r - m.y_p_H2[k] * m.p_p) * m.A_mem / N

        self.J_CO2_def = pyo.Constraint(self.segments, rule=J_CO2_def)
        self.J_CO_def = pyo.Constraint(self.segments, rule=J_CO_def)
        self.J_H2_def = pyo.Constraint(self.segments, rule=J_H2_def)

        # --- Cross-flow permeate mole fractions: yp,i × J_total = Ji ---
        def y_p_CO2_def(m, k):
            return m.y_p_CO2[k] * (m.J_CO2[k] + m.J_CO[k] + m.J_H2[k]) == m.J_CO2[k]

        def y_p_CO_def(m, k):
            return m.y_p_CO[k] * (m.J_CO2[k] + m.J_CO[k] + m.J_H2[k]) == m.J_CO[k]

        def y_p_H2_def(m, k):
            return m.y_p_H2[k] * (m.J_CO2[k] + m.J_CO[k] + m.J_H2[k]) == m.J_H2[k]

        self.y_p_CO2_def = pyo.Constraint(self.segments, rule=y_p_CO2_def)
        self.y_p_CO_def = pyo.Constraint(self.segments, rule=y_p_CO_def)
        self.y_p_H2_def = pyo.Constraint(self.segments, rule=y_p_H2_def)

        # --- Retentate mole balance across each segment ---
        def balance_CO2(m, k):
            return m.nf_r_CO2[k + 1] == m.nf_r_CO2[k] - m.J_CO2[k]

        def balance_CO(m, k):
            return m.nf_r_CO[k + 1] == m.nf_r_CO[k] - m.J_CO[k]

        def balance_H2(m, k):
            return m.nf_r_H2[k + 1] == m.nf_r_H2[k] - m.J_H2[k]

        self.balance_CO2 = pyo.Constraint(self.segments, rule=balance_CO2)
        self.balance_CO = pyo.Constraint(self.segments, rule=balance_CO)
        self.balance_H2 = pyo.Constraint(self.segments, rule=balance_H2)

        # --- Total permeate outlet flows (sum over all segments) ---
        self.permeate_CO2_total = pyo.Constraint(expr=self.nf_p_CO2_out == sum(self.J_CO2[k] for k in self.segments))
        self.permeate_CO_total = pyo.Constraint(expr=self.nf_p_CO_out == sum(self.J_CO[k] for k in self.segments))
        self.permeate_H2_total = pyo.Constraint(expr=self.nf_p_H2_out == sum(self.J_H2[k] for k in self.segments))

        # --- Stage cut: (total permeate) / (total feed) ---
        self.stage_cut_def = pyo.Constraint(
            expr=self.stage_cut * (self.nf_r_CO2[1] + self.nf_r_CO[1] + self.nf_r_H2[1])
            == (self.nf_p_CO2_out + self.nf_p_CO_out + self.nf_p_H2_out)
        )

        # --- Capital cost: K_B → K_I → total CAPEX ---
        # BC_membrane [€/m²] is purchased equipment cost (K_B).
        # K_I = f_H_membrane × K_B  (Lang/Hand, f_H = 2.5 for modules)
        # CAPEX = 2 × K_I           (Guthrie ×2 overhead, Mitsos VL12 slide 32)
        # Effective unit cost: 2 × 2.5 × 50 = 250 €/m²
        self.const_CAPEX_membrane = pyo.Constraint(
            expr=self.CAPEX_membrane == self.BC_membrane * self.A_mem * self.CEPCI_scaling
        )
        self.const_CAPEX_frame = pyo.Constraint(
            expr=self.CAPEX_frame == pow((self.A_mem / 2000), 0.7) * self.base_frame_cost * self.CEPCI_scaling
        )
        self.CAPEX_def = pyo.Constraint(
            expr=self.CAPEX == 2 * self.f_H_membrane * (self.CAPEX_frame + self.CAPEX_membrane)
        )

        # --- OPEX: membrane replacement + annual maintenance, per operating hour ---
        # Replacement (Heßelmann et al. Eq. 15): CAPEX amortized over 5-yr membrane lifetime
        # Maintenance: maint_factor [%/yr] × CAPEX, spread over operating hours per year
        self.OPEX_def = pyo.Constraint(
            expr=self.OPEX
            == self.CAPEX_membrane / (self.t_lifetime_membrane * Economics.operating_hours_per_year)
            + self.maint_factor * self.CAPEX / Economics.operating_hours_per_year
        )

        # --- Optional: CO2 depletion target in retentate (upper bound on CO2 mole fraction) ---
        # Depletion >= target  =>  y_ret_CO2 <= 1 - target/100
        # Expressed as: nf_r_CO2[N+1] <= alpha * (nf_r_CO2[N+1] + nf_r_CO[N+1] + nf_r_H2[N+1])
        if self._co2_depletion_target is not None:
            alpha = 1.0 - self._co2_depletion_target / 100.0
            self.co2_depletion_con = pyo.Constraint(
                expr=self.nf_r_CO2[N + 1] <= alpha * (self.nf_r_CO2[N + 1] + self.nf_r_CO[N + 1] + self.nf_r_H2[N + 1])
            )

    def calculate_performance_metrics(self):
        N = self._N
        nf_feed_total = self.nf_r_CO2[1].value + self.nf_r_CO[1].value + self.nf_r_H2[1].value
        nf_ret_total = self.nf_r_CO2[N + 1].value + self.nf_r_CO[N + 1].value + self.nf_r_H2[N + 1].value
        nf_perm_total = self.nf_p_CO2_out.value + self.nf_p_CO_out.value + self.nf_p_H2_out.value

        self.y_ret_CO2 = self.nf_r_CO2[N + 1].value / (nf_ret_total + 1e-12)
        self.y_ret_CO = self.nf_r_CO[N + 1].value / (nf_ret_total + 1e-12)
        self.y_ret_H2 = self.nf_r_H2[N + 1].value / (nf_ret_total + 1e-12)

        self.y_perm_CO2 = self.nf_p_CO2_out.value / (nf_perm_total + 1e-12)
        self.y_perm_CO = self.nf_p_CO_out.value / (nf_perm_total + 1e-12)
        self.y_perm_H2 = self.nf_p_H2_out.value / (nf_perm_total + 1e-12)

        # CO2 recovery in permeate (for CO2-selective membrane the permeate is CO2-rich)
        self.CO2_recovery = self.nf_p_CO2_out.value / (self.nf_r_CO2[1].value + 1e-12)

        # CO recovery in retentate (CO product should stay in retentate)
        self.CO_retention = self.nf_r_CO[N + 1].value / (self.nf_r_CO[1].value + 1e-12)

    def print_variables(self):
        self.calculate_performance_metrics()
        N = self._N

        print(f"\nGas Permeation Membrane ({self.membrane_material})")
        table = PrettyTable()
        table.field_names = ["     Variable     ", "                  Value                  ", "  Unit  "]

        table.add_row(["A_mem", self.A_mem.value, "m²"])
        table.add_row(["p_r", self.p_r.value / 1e5, "bar"])
        table.add_row(["p_p", self.p_p.value / 1e5, "bar"])
        table.add_row(["pressure ratio", self.p_r.value / self.p_p.value, "-"])
        table.add_row(["stage_cut", self.stage_cut.value, "-"])

        table.add_row(["nf_r_CO2 (feed)", self.nf_r_CO2[1].value, "mol/s"])
        table.add_row(["nf_r_CO  (feed)", self.nf_r_CO[1].value, "mol/s"])
        table.add_row(["nf_r_H2  (feed)", self.nf_r_H2[1].value, "mol/s"])

        table.add_row(["nf_r_CO2 (ret.)", self.nf_r_CO2[N + 1].value, "mol/s"])
        table.add_row(["nf_r_CO  (ret.)", self.nf_r_CO[N + 1].value, "mol/s"])
        table.add_row(["nf_r_H2  (ret.)", self.nf_r_H2[N + 1].value, "mol/s"])

        table.add_row(["nf_p_CO2 (perm.)", self.nf_p_CO2_out.value, "mol/s"])
        table.add_row(["nf_p_CO  (perm.)", self.nf_p_CO_out.value, "mol/s"])
        table.add_row(["nf_p_H2  (perm.)", self.nf_p_H2_out.value, "mol/s"])

        table.add_row(["y_ret_CO2", self.y_ret_CO2, "-"])
        table.add_row(["y_ret_CO", self.y_ret_CO, "-"])
        table.add_row(["y_ret_H2", self.y_ret_H2, "-"])

        table.add_row(["y_perm_CO2", self.y_perm_CO2, "-"])
        table.add_row(["y_perm_CO", self.y_perm_CO, "-"])
        table.add_row(["y_perm_H2", self.y_perm_H2, "-"])

        table.add_row(["CO2 recovery", self.CO2_recovery * 100, "%"])
        table.add_row(["CO retention", self.CO_retention * 100, "%"])
        table.add_row(["CO2 depletion ret.", 100 - self.y_ret_CO2 * 100, "%"])

        table.add_row(["CAPEX", self.CAPEX.value, "€"])
        table.add_row(["OPEX", self.OPEX.value, "€/h"])
        table.add_row(["membrane lifetime", pyo.value(self.t_lifetime_membrane), "yr"])
        table.add_row(["plant lifetime", pyo.value(self.t_plant_life), "yr"])
        table.add_row(["maintenance factor", pyo.value(self.maint_factor) * 100, "%/yr"])

        print(table)

        print("\nSegment-wise profiles:")
        seg_table = PrettyTable()
        seg_table.field_names = [
            "Seg",
            "y_r_CO2",
            "y_r_CO",
            "y_r_H2",
            "y_p_CO2",
            "J_CO2 [mol/s]",
            "J_CO [mol/s]",
            "J_H2 [mol/s]",
        ]
        for k in self.segments:
            seg_table.add_row(
                [
                    k,
                    f"{self.y_r_CO2[k].value:.4f}",
                    f"{self.y_r_CO[k].value:.4f}",
                    f"{self.y_r_H2[k].value:.4f}",
                    f"{self.y_p_CO2[k].value:.4f}",
                    f"{self.J_CO2[k].value:.4f}",
                    f"{self.J_CO[k].value:.6f}",
                    f"{self.J_H2[k].value:.6f}",
                ]
            )
        print(seg_table)
