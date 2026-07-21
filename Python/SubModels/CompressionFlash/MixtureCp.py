"""
Mixture molar heat capacity and enthalpy using NIST Shomate equations and
ideal mixing rule.

Shomate equations (NIST):
    Cp(T) = A + B*t + C*t^2 + D*t^3 + E/t^2              [J/mol·K]
    H(T) - H°298 = A*t + B*t²/2 + C*t³/3 + D*t⁴/4
                   - E/t + F - H_const                    [kJ/mol]
    where t = T [K] / 1000

Mixing rule (ideal gas):
    Cp_mix(T)       = sum_i( z_i * Cp_i(T) )
    delta_h_mix     = sum_i( z_i * [H_i(T_hot) - H_i(T_cold)] )  [kJ/mol]

Supported components and valid temperature ranges:
    CO2 :  298 – 1200 K
    CO  :  298 – 1300 K
    H2  :  298 – 1000 K

Sources: NIST WebBook / Chase (1998)
"""

import pyomo.environ as pyo


# NIST Shomate coefficients  {component: (A, B, C, D, E, F, H_const)}
# F and H_const are the enthalpy integration constants from NIST.
# H(T) - H°298 = A*t + B*t²/2 + C*t³/3 + D*t⁴/4 - E/t + F - H_const  [kJ/mol]
_SHOMATE = {
    "CO2": ( 24.99735,  55.18696,  -33.69137,   7.948387, -0.136638, -403.6075, -393.5224),
    "CO":  ( 25.56759,   6.096130,   4.054656,  -2.671301,  0.131021, -118.0089, -110.5271),
    "H2":  ( 33.066178, -11.363417,  11.432816,  -2.772874, -0.158558,  -9.980797,   0.0),
}


class MixtureCpModel(pyo.ConcreteModel):
    """
    Pyomo model that computes the molar heat capacity of a gas mixture
    at a given temperature using the Shomate equation and ideal mixing.

    Intended to be attached as a block inside a parent flowsheet model.
    The variable `T` can be linked to any upstream temperature variable
    via a parent-level equality constraint.

    Attributes exposed as Pyomo Expressions
    ----------------------------------------
    cp[i]   – pure-component Cp_i(T)   [J/mol·K]
    cp_mix  – mixture Cp(T)            [J/mol·K]

    Variable
    --------
    T       – temperature [K], free variable so parent can link to it
    """

    def __init__(self, z_dict, T_init=298.15, name="MixtureCp"):
        """
        Parameters
        ----------
        z_dict  : dict   {component: mole_fraction}  must sum to 1
        T_init  : float  initial guess / default temperature [K]
        """
        super().__init__(name=name)

        unknown = set(z_dict) - set(_SHOMATE)
        if unknown:
            raise ValueError(
                f"No Shomate coefficients available for: {unknown}. "
                f"Supported: {set(_SHOMATE)}"
            )

        # ------------------------------------------------------------------
        # Set and composition
        # ------------------------------------------------------------------
        self.I = pyo.Set(initialize=list(z_dict.keys()))
        self.z = pyo.Param(self.I, initialize=z_dict)

        # ------------------------------------------------------------------
        # Shomate coefficients stored as Params
        # ------------------------------------------------------------------
        self.sh_A      = pyo.Param(self.I, initialize={i: _SHOMATE[i][0] for i in z_dict})
        self.sh_B      = pyo.Param(self.I, initialize={i: _SHOMATE[i][1] for i in z_dict})
        self.sh_C      = pyo.Param(self.I, initialize={i: _SHOMATE[i][2] for i in z_dict})
        self.sh_D      = pyo.Param(self.I, initialize={i: _SHOMATE[i][3] for i in z_dict})
        self.sh_E      = pyo.Param(self.I, initialize={i: _SHOMATE[i][4] for i in z_dict})
        self.sh_F      = pyo.Param(self.I, initialize={i: _SHOMATE[i][5] for i in z_dict})
        self.sh_Hconst = pyo.Param(self.I, initialize={i: _SHOMATE[i][6] for i in z_dict})

        # ------------------------------------------------------------------
        # Temperature variables — free so a parent model can link them
        # T       : used for Cp evaluation (e.g. at feed / compressor inlet)
        # T_hot   : hot-side of HX1 (compressor outlet, linked via constraint)
        # T_cold  : cold-side of HX1 = hot-side of HX2 (intermediate T)
        # T_cold2 : cold-side of HX2 (flash temperature)
        # ------------------------------------------------------------------
        self.T      = pyo.Var(initialize=T_init,  bounds=(200.0, 2000.0))  # K
        self.T_hot  = pyo.Var(initialize=T_init,  bounds=(200.0, 2000.0))  # K
        self.T_cold = pyo.Var(initialize=303.15,  bounds=(200.0, 2000.0))  # K
        self.T_cold2 = pyo.Var(initialize=238.15, bounds=(200.0, 2000.0))  # K

        # ------------------------------------------------------------------
        # Expressions
        # ------------------------------------------------------------------
        @self.Expression(self.I)
        def cp(m, i):
            """Pure-component Cp_i(T) via Shomate equation [J/mol·K]."""
            t = m.T / 1000.0
            return (
                m.sh_A[i]
                + m.sh_B[i] * t
                + m.sh_C[i] * t**2
                + m.sh_D[i] * t**3
                + m.sh_E[i] / t**2
            )

        @self.Expression()
        def cp_mix(m):
            """Mixture Cp using ideal mixing rule [J/mol·K]."""
            return sum(m.z[i] * m.cp[i] for i in m.I)

        def _shomate_h(m, i, T_var):
            """Shomate enthalpy H_i(T) - H°298 [kJ/mol]."""
            t = T_var / 1000.0
            return (
                m.sh_A[i] * t
                + m.sh_B[i] * t**2 / 2.0
                + m.sh_C[i] * t**3 / 3.0
                + m.sh_D[i] * t**4 / 4.0
                - m.sh_E[i] / t
                + m.sh_F[i]
                - m.sh_Hconst[i]
            )

        @self.Expression(self.I)
        def h_hot(m, i):
            """Pure-component enthalpy at T_hot [kJ/mol]."""
            return _shomate_h(m, i, m.T_hot)

        @self.Expression(self.I)
        def h_cold(m, i):
            """Pure-component enthalpy at T_cold [kJ/mol]."""
            return _shomate_h(m, i, m.T_cold)

        @self.Expression()
        def delta_h_mix(m):
            """Mixture molar enthalpy difference H(T_hot) - H(T_cold) [kJ/mol]. HX1."""
            return sum(m.z[i] * (m.h_hot[i] - m.h_cold[i]) for i in m.I)

        @self.Expression(self.I)
        def h_cold2(m, i):
            """Pure-component enthalpy at T_cold2 [kJ/mol]."""
            return _shomate_h(m, i, m.T_cold2)

        @self.Expression()
        def delta_h_mix2(m):
            """Mixture molar enthalpy difference H(T_cold) - H(T_cold2) [kJ/mol]. HX2."""
            return sum(m.z[i] * (m.h_cold[i] - m.h_cold2[i]) for i in m.I)

        # Dummy objective for standalone use
        self.obj = pyo.Objective(expr=0)

    # ------------------------------------------------------------------
    # Standalone helpers
    # ------------------------------------------------------------------
    def solve(self, solver_name="ipopt"):
        """Solve standalone (fixes T, evaluates Cp expressions)."""
        solver = pyo.SolverFactory(solver_name)
        return solver.solve(self)

    def report(self):
        print(f"\n--- MixtureCp at T = {pyo.value(self.T) - 273.15:.2f} °C ---")
        for i in self.I:
            print(f"  Cp_{i:<4} = {pyo.value(self.cp[i]):.4f} J/mol·K")
        print(f"  Cp_mix  = {pyo.value(self.cp_mix):.4f} J/mol·K")


# ----------------------------------------------------------------------
# Standalone test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    z = {"CO2": 0.68, "CO": 0.26, "H2": 0.06}

    model = MixtureCpModel(z_dict=z, T_init=298.15)
    model.T.fix(298.15)
    model.solve()
    model.report()

    model.T.fix(500.0)
    model.solve()
    model.report()
