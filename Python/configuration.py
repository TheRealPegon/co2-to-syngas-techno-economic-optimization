from dataclasses import dataclass
from math import log, pi


@dataclass
class HeatXchangerEconomics:
    # Aspen-derived linear reference values (kept for comparison, not used in Guthrie CAPEX)
    capex_unit_cost_water: float = 201.0  # €/kW_thermal
    capex_unit_cost_NH3: float = 515.0  # €/kW_thermal

    # OPEX utilities
    # cooling_cost_kwh_water: float = 0.0012  # €/kWh_thermal from claude cause aspen is kinda broken
    cooling_cost_kwh_water: float = 0.00131  # €/kWh_thermal from turton et al. 2018 same as in amine wash flowsheet
    cooling_cost_kwh_NH3: float = 0.0116  # €/kWh_thermal from aspen Refrigeration utility
    steam_price_kwh: float = 23.4  # €/GJ — kept in sync with AmineWashParameters.steam_cost_eur_per_gj

    # Heat transfer parameters for Guthrie/Turton area-based CAPEX
    # U values: typical ranges for shell-and-tube exchangers from
    #   Perry's Chemical Engineers' Handbook, 9th ed., Table 11-10
    #   gas–liquid (forced convection both sides): 200–800 W/m²K
    U_overall_water: float = 600.0  # W/m²K  gas–water, shell-and-tube (Perry's, Table 11-10)
    U_overall_NH3: float = 350.0  # W/m²K  gas–boiling refrigerant (Perry's, Table 11-10)
    # ΔT_lm: conservative engineering assumption; actual value depends on stage inlet T
    delta_T_lm_water: float = 30.0  # K  assumed LMTD for water-cooled intercoolers
    delta_T_lm_NH3: float = 15.0  # K  assumed LMTD for NH3 refrigeration HX


@dataclass
class Constants:
    F: float = 96485  # in C/mol
    R: float = 8.314  # in J/(mol*K)
    T: float = 298  # in K (25 °C)
    P: float = 1e5  # in Pa
    z_CO: int = 2  # number of electrons for CO production
    z_H2: int = 2  # number of electrons for H2 production
    q_CO2_s: float = 6.043  # in mol/kg
    Bs_CO2: float = 0.08008  # in 1/kPa
    ns_CO2: float = 1.887
    Henry_constant_CO2: float = q_CO2_s * Bs_CO2  # in mol/(kg*kPa)
    q_CO_s: float = 3.890  # in mol/kg
    Bs_CO: float = 0.002123  # in 1/kPa
    ns_CO: float = 1.220
    bulk_density: float = 680  # in kg/m^3?
    saturation_adsorbent: float = 0.95
    # Park et al. (2000) non-isothermal kinetics — Eq. 16
    # β'(P) = exp(park_beta_intercept + park_beta_slope * ln(P_kPa))
    # k_app = park_ha_rhosCs / (1 + β')
    # t_ads(P) = -ln(1 - saturation_adsorbent) / k_app  — computed in PSAModel as Pyomo expression
    # Park et al. CO2 kinetics (Table 9, 293 K) — NLS fit in linear beta' scale
    park_ha_rhosCs: float = 0.0118  # s⁻¹, mean of Table 9 at 293 K
    park_beta_slope: float = 0.7335  # power-law slope of β'_CO2 vs P_kPa (NLS, 293 K)
    park_beta_intercept: float = -3.7816  # power-law intercept (NLS, 293 K)
    # Park et al. CO kinetics (Table 9, 293 K) — NLS fit in linear beta' scale
    park_ha_rhosCs_CO: float = 0.0232  # s⁻¹, mean of Table 9 at 293 K
    park_beta_slope_CO: float = 1.0411  # power-law slope of β'_CO vs P_kPa (NLS, 293 K)
    park_beta_intercept_CO: float = -5.3079  # power-law intercept (NLS, 293 K)
    # Bed utilisation factor: fraction of the theoretical equilibrium working capacity
    # (q_CO2_H - q_CO2_L) that is actually realised in a real PSA cycle.
    # The surrogate model assumes full equilibrium loading across the entire bed, which
    # is never achieved in practice because the adsorption front breaks through before
    # the bed is fully loaded. Typical values from literature are 0.4–0.7 depending on
    # cycle design, feed velocity, and bed geometry.
    # TODO: justify chosen value against PSA literature (e.g. Ruthven 1984,
    #       Sircar 2002, or zeolite 13X CO2 capture studies).
    bed_utilisation: float = 0.9
    MW_H2: float = 2.016 / 1000  # in kg/mol
    MW_CO2: float = 44.01 / 1000  # in kg/mol
    MW_CO: float = 28.01 / 1000  # in kg/mol


@dataclass
class Economics:
    price_power: float = 0.08  # in €/kWh
    price_CO2_feed: float = 22 * 44.01 / 1e6  # €/mol ≈ 22 €/t CO₂
    convert_usd_to_euro: float = 0.96  # conversion factor USD to Euro
    interest_rate: float = 0.085  # 8.5 % — consistent with CCS/CCU literature (Nwaoha 2018, Kim & Léonard 2025)
    operating_days_per_year: int = 350  # days/year
    operating_hours_per_year: float = 350 * 24  # h/year = 8400

    plant_life_years: int = 20

    # Maintenance cost as fraction of installed CAPEX per year
    # Source: Peters, Timmerhaus & West, "Plant Design and Economics for Chemical Engineers", 5th ed.
    maintenance_factor_rotating: float = 0.05  # 5 %/yr — compressors, pumps
    maintenance_factor_static: float = 0.03  # 3 %/yr — heat exchangers, vessels
    maintenance_factor_membrane: float = (
        0.01  # from   Multi-stage gas separation membrane processes used in post-combustion capture: Energetic and economic analyses Li Zhao∗, Ernst Riensche, Ludger Blum, Detlef Stolten
    )

    # https://toweringskills.com/financial-analysis/cost-indices/
    CEPCI_2024: float = 797.9  # CEPCI for the year 2023, calculated based on CEPCI values from 1950 to 2023
    CEPCI_1993: float = 359.2  # CEPCI for the year 1993,https://www.scribd.com/document/368638227/Cepci-Index-1950-2015
    CEPCI_1969: float = 119.0  # CEPCI for the year 1969, basis year for Guthrie (1969) HX correlation
    CEPCI_1974: float = 144.1  # CEPCI for the year 1974, basis year for Guthrie (1974) compressor/PSA correlations
    CEPCI_2001: float = 397.0  # CEPCI for the year 2001, basis year for Turton et al. (2012) correlations

    # Present Value Factor: discounts a constant annual OPEX stream to today's value
    # PVF = (1 - (1+r)^-n) / r
    # At r=8.5%, n=20: PVF = 9.46  (vs undiscounted multiplier of 20)
    @property
    def pvf(self) -> float:
        r = self.interest_rate
        n = self.plant_life_years
        return (1 - (1 + r) ** (-n)) / r

    # Annuity factor (Capital Recovery Factor): converts CAPEX to equivalent annual cost
    # CRF = r(1+r)^n / ((1+r)^n - 1)  =  1 / PVF
    @property
    def annuity_factor_plant_life(self) -> float:
        return sum(1 / (1 + self.interest_rate) ** y for y in range(1, self.plant_life_years + 1))


@dataclass
class ElectrolyzerParametersSurrogate:
    E_anode: float = 1.23  # in V

    # Butler_volmer kinetics
    BV_CO: float = 0.33
    BV_H2: float = 0.33
    BV_ECO: float = 0.11  # in V
    BV_iCO: float = 3.3e-4  # in A/m2
    BV_iH2: float = 3.4e-6  # in A/m2
    BV_CO2ref: float = 34  # in mol/m3
    BV_mCO2: float = 1

    Channel_H = 1e-3  # Channel height in m
    Channel_W = 1e-2  # Channel Width in m
    Channel_L = 0.1  # Channel Length in m
    Lm = 115e-6  # Membrane thickness in m
    reacting_area: float = Channel_W * Channel_L  # Reacting Area of 1 channel in m2

    sigma_el: float = 5.5  # in S/m
    sigma_m: float = 9.3  # in S/m


@dataclass
class ElectrolyzerParameters:
    current_density: float = 3000  # in A/m^2 = 300 mA/cm^2
    voltage: float = 3  # in V
    area_elec: float = 5000  # in m^2 (0.5 m^2 x 10000 cells)

    current: float = area_elec * current_density  # in A
    power_elec: float = current * voltage / 1000  # in kW
    op_cost_elec: float = power_elec * Economics.price_power  # in €/h (assuming 0.08 €/kWh)

    # CO2 electrolyzer stack manufacturing cost [€/m²].
    # Source: Otto & Steenbuck (2025), Table 4.2 at 500 000 m²/a production volume.
    # This is the pure stack production cost (K_B level): materials, manufacturing
    # labour, production-line equipment, and production-facility buildings.
    # It does NOT include BOP (power electronics, gas handling, cooling, controls),
    # site installation, or plant-level indirect costs.
    # Note: Bagemihl (2024) reports 920 $/m² bare module cost → 920 × 0.96 × 2 = 1766 €/m²
    # installed (×2 bare→installed factor from Prozessentwicklung in der Verfahrenstechnik,
    # Vorlesungseinheit 21). Otto & Steenbuck value used here as it includes the full
    # K_B cost breakdown at projected high-volume production scale.
    base_cost_elec: float = 4670  # €/m² — Otto & Steenbuck (2025), K_B stack cost
    # base_cost_elec: float = (
    #    920 * Economics.convert_usd_to_euro
    # )  # €/m² — Bagemihl (2024) bare module cost, converted from USD

    # Installation multiplier: stack K_B → installed system cost K_I.
    # Based on the Timmerhaus-West method as used in Otto & Steenbuck (2025),
    # Eq. 3.19: C_fixed,elec = (f_direct + f_indirect) × C_area × System_Area
    # with f_direct = 3.6 (direct installation: BOP, piping, civil, electrical)
    # and f_indirect = 1.44 (indirect: engineering, contingency, overhead).
    # Combined factor: 3.6 + 1.44 = 5.04

    # I do not trust this Timmerhaus-West installation factor as it came from Tahjana BA and I cant verify the source.
    # as an alternative I use *2 multiplier from Prozessentwicklung in der Verfahrenstechnik, Vorlesungseinheit 21, which is a more conservative estimate for the stack installation factor.
    # This scales the stack manufacturing cost to a fully installed system cost,
    # consistent with the K_I level used by all other submodels (Guthrie f_H).
    # CAPEX [€] = f_installation × base_cost_elec × total_area
    f_installation_elec: float = 2


@dataclass
class ElectrolyzerVariablesSurrogate:
    Channel_H = ElectrolyzerParametersSurrogate.Channel_H
    Channel_W = ElectrolyzerParametersSurrogate.Channel_W  # in m

    c_in_CO2_lb: float = 0.0  # in mol/m^3
    c_in_CO2_ub: float = 10000  # in mol/m^3

    c_in_CO_lb: float = 0.0  # in mol/m^3
    c_in_CO_ub: float = 10000  # in mol/m^3

    CO2share: float = 0.9  # dimensionless
    CO2share_lb: float = 0.8  # dimensionless
    CO2share_ub: float = 1.0  # dimensionless

    # v: float = 10 * 1e-6 / 60 * 1 / Channel_H / Channel_W  # in m/s
    v: float = 0.08  # in m/s
    v_lb: float = 5 * 1e-6 / 60 * 1 / Channel_H / Channel_W  # in m/s
    v_ub: float = 50 * 1e-6 / 60 * 1 / Channel_H / Channel_W  # in m/s

    E_appl: float = -1.3  # in V
    E_appl_lb: float = -1.4  # in V
    E_appl_ub: float = -1  # in V

    Vcell: float = 4.42  # in V
    Vcell_lb: float = 2.5  # in V
    Vcell_ub: float = 6  # in V

    eta_actA: float = 1.24
    eta_actA_lb: float = 0.00001
    eta_actA_ub: float = 3

    eta_ohm: float = 0.5
    eta_ohm_lb: float = 0.00001
    eta_ohm_ub: float = 3

    CD: float = 3000
    CD_lb: float = 1
    CD_ub: float = 10000

    total_area: float = 5000  # Total Electrolyzer Area in m2
    n: float = total_area / ElectrolyzerParametersSurrogate.reacting_area  # Number of channels in the electrolyzer


@dataclass
class ElectrolyzerVariables:
    nf_CO2_elec_in: float = 160  # in mol/s
    nf_CO2_elec_in_lb: float = 0.001
    nf_CO2_elec_in_ub: float = 10000

    nf_CO_elec_in: float = 10  # in mol/s
    nf_CO_elec_in_lb: float = 0.001
    nf_CO_elec_in_ub: float = 10000

    nf_CO2_elec_out: float = 100  # in mol/s
    nf_CO2_elec_out_lb: float = 0.001
    nf_CO2_elec_out_ub: float = 10000

    nf_CO_elec_out: float = 10  # in mol/s
    nf_CO_elec_out_lb: float = 0.001
    nf_CO_elec_out_ub: float = 10000

    nf_CO_elec: float = 10  # in mol/s
    nf_CO_elec_lb: float = 0.001
    nf_CO_elec_ub: float = 10000

    nf_H2_elec_out: float = 10  # in mol/s
    nf_H2_elec_out_lb: float = 0.001
    nf_H2_elec_out_ub: float = 10000

    lambda_val: float = 2  # dimensionless
    lambda_lb: float = 1
    lambda_ub: float = 7

    FE_CO: float = 0.9  # Faradaic efficiency for CO
    FE_CO_lb: float = 0.0001
    FE_CO_ub: float = 1

    FE_H2: float = 0.1  # Faradaic efficiency for H2
    FE_H2_lb: float = 0.0001
    FE_H2_ub: float = 1


@dataclass
class H2Parameters:
    C_F_el: float = 1.75  # installation factor (civil works + site), Mucci et al. 2023
    CE_PCI: float = 797.9  # Update factor for the year 2023 (calculated on CEPCI values) deprecated
    electrolyzer_efficiency: float = 0.64
    hhv_h2_kwh_per_kg: float = 39.4  # Higher Heating Value of H2 in kWh/kg
    actual_kwh_per_kg_h2: float = hhv_h2_kwh_per_kg / electrolyzer_efficiency


@dataclass
class PSAParameters:
    y_CO_product: float = 0.98  # targeted CO purity
    y_CO2_product: float = 0.02
    y_CO_psaH_out: float = y_CO_product
    y_CO2_psaH_out: float = y_CO2_product
    y_CO_psaL_out: float = y_CO_product
    y_CO2_psaL_out: float = y_CO2_product

    p_L: float = 100  # in kPa
    voidage: float = 0.4  # dimensionless
    efficiency_comp: float = 0.78  # dimensionless — centrifugal compressor
    efficiency_pump: float = 0.40  # dimensionless — liquid ring vacuum pump (lower than centrifugal)
    # Heat capacity ratio for the desorption gas (CO2-rich, ~90% CO2 / 10% CO at 25°C).
    # CO2: γ=1.289, CO: γ=1.400 → molar-average γ ≈ 1.30 at feed composition.
    # Used for isentropic vacuum pump power (replaces isothermal approximation).
    gamma_mix: float = 1.30  # dimensionless

    # Zeolite 13X pellets (PSA/CO2-capture grade), bulk industrial price.
    # Literature range: €2–5/kg. Sources: Ruthven et al. (1994), Grande (2012),
    # and supplier quotes for CO2-capture grade 13X (BASF, Zeochem, etc.).
    # €1.05/kg (previous value) was a raw powder commodity price — not pelletised
    # PSA-grade material, which carries a ~3x premium for forming and activation.
    price_zeoperkg: float = 3.0  # in €/kg

    # Zeolite replacement interval: pelletised 13X degrades under repeated thermal
    # and pressure cycling. 5-year replacement is a conservative industrial estimate.
    # (Ref: Grande 2012, Adsorption 18:415–433)
    adsorbent_lifetime_years: float = 5.0

    # Guthrie parameter for pressure vessel
    factor_material: float = 2.3  # stainless 304 Clad
    factor_installation: float = 2.921
    price_index: float = 1  # price index
    # Guthrie ×2 overhead factor (Mitsos VL12 slide 32: K = 2×K_I).
    # Covers buildings, infrastructure, engineering, working capital, startup.
    # Replaces the old indirect_cost_factor = 1.38 for consistency with
    # AmineWash and membrane submodels.
    indirect_cost_factor_pv: float = 2.0

    # Guthrie parameter for compressor
    factor_installation_comp: float = 1.565
    indirect_cost_factor_comp: float = 2.0  # same ×2 Guthrie overhead factor

    # Turton et al. (2012) liquid ring vacuum pump correlation.
    # log10(Cp_USD_2001) = K1 + K2*log10(W_kW) + K3*(log10(W_kW))^2
    # Valid range: 1–1000 kW shaft power.  CEPCI basis: 397 (year 2001).
    # Source: Turton et al., "Analysis, Synthesis, and Design of Chemical Processes",
    #         4th ed. (2012), Appendix A, Table A.1.
    turton_K1_pump: float = 3.8696
    turton_K2_pump: float = 0.3161
    turton_K3_pump: float = 0.1220
    turton_FBM_pump: float = 1.5  # bare module factor for vacuum equipment
    indirect_cost_factor_pump: float = 2.0  # same ×2 Guthrie overhead factor


@dataclass
class PSAVariables:
    nf_CO2_feed: float = 72.69909143963791  # in mol/s
    nf_CO2_feed_lb: float = 10
    nf_CO2_feed_ub: float = 1000

    p_H: float = 500.0  # in kPa
    p_H_lb: float = 100  # must be strictly > p_L_ub (100 kPa) to ensure a pressure swing exists
    p_H_ub: float = 100000

    p_L: float = 100  # in kPa — desorption at atmospheric (no vacuum pump)
    p_L_lb: float = 100  # minimum 1 bar: desorption is atmospheric, no vacuum needed
    p_L_ub: float = 1000

    nf_CO2_psaH_in: float = 254.6380469027694  # in mol/s
    nf_CO2_psaH_in_lb: float = 0.001
    nf_CO2_psaH_in_ub: float = 10000

    nf_CO_psaH_in: float = 88.69131518544398  # in mol/s
    nf_CO_psaH_in_lb: float = 0.001
    nf_CO_psaH_in_ub: float = 10000

    y_CO2_psaH_in: float = 0.7416727930113473  # dimensionless
    y_CO2_psaH_in_lb: float = 0.0001
    y_CO2_psaH_in_ub: float = 1

    y_CO_psaH_in: float = 0.2583272069886527  # dimensionless
    y_CO_psaH_in_lb: float = 0.0001
    y_CO_psaH_in_ub: float = 1

    nf_CO2_psaH_out: float = 0.2907963657585523  # in mol/s
    nf_CO2_psaH_out_lb: float = 0.001
    nf_CO2_psaH_out_ub: float = 10

    nf_CO_psaH_out: float = 72.40829507387933  # in mol/s
    nf_CO_psaH_out_lb: float = 0.001
    nf_CO_psaH_out_ub: float = 1000

    nf_CO2_product: float = nf_CO2_psaH_out  # in mol/s
    nf_CO2_product_lb: float = nf_CO2_psaH_out_lb
    nf_CO2_product_ub: float = nf_CO2_psaH_out_ub

    nf_CO_product: float = nf_CO_psaH_out  # in mol/s
    nf_CO_product_lb: float = nf_CO_psaH_out_lb
    nf_CO_product_ub: float = nf_CO_psaH_out_ub

    nf_CO2_psaL_out: float = 254.34725053701078  # in mol/s
    nf_CO2_psaL_out_lb: float = 0.001
    nf_CO2_psaL_out_ub: float = 1000

    nf_CO_psaL_out: float = 16.283020111564664  # in mol/s
    nf_CO_psaL_out_lb: float = 0.001
    nf_CO_psaL_out_ub: float = 100

    y_CO2_psaL_out: float = 0.8  # dimensionless
    y_CO2_psaL_out_lb: float = 0.0001
    y_CO2_psaL_out_ub: float = 1

    y_CO_psaL_out: float = 0.2  # dimensionless
    y_CO_psaL_out_lb: float = 0.0001
    y_CO_psaL_out_ub: float = 1

    nf_CO2_LR: float = 360.17400309733733  # in mol/s
    nf_CO2_LR_lb: float = 0.001
    nf_CO2_LR_ub: float = 1e5

    nf_CO_LR: float = 1.0990599777193355  # in mol/s
    nf_CO_LR_lb: float = 0.001
    nf_CO_LR_ub: float = 1000

    nf_CO2_HR: float = 614.5212536343482  # in mol/s
    nf_CO2_HR_lb: float = 0.001
    nf_CO2_HR_ub: float = 10000

    nf_CO_HR: float = 17.382080089284  # in mol/s
    nf_CO_HR_lb: float = 0.001
    nf_CO_HR_ub: float = 1000

    mf_ads: float = 143.66962759926938  # in kg/s
    mf_ads_lb: float = 10
    mf_ads_ub: float = 1e6

    q_CO2_H: float = 4.473980741401526  # in mol/kg
    q_CO2_H_lb: float = 0.0001
    q_CO2_H_ub: float = 6.092

    q_CO2_L: float = 2.6369664885644744  # in mol/kg
    q_CO2_L_lb: float = 0.0001
    q_CO2_L_ub: float = 6.092

    q_CO_H: float = 0.1174402599122145  # in mol/kg
    q_CO_H_lb: float = 0.000001
    q_CO_H_ub: float = 3.966

    q_CO_L: float = 0.007928423644688829  # in mol/kg
    q_CO_L_lb: float = 0.000001
    q_CO_L_ub: float = 3.966

    power_comp_in: float = 8000  # in kW
    power_comp_in_lb: float = 0.001
    power_comp_in_ub: float = 1e6

    power_comp_out: float = 8000  # in kW
    power_comp_out_lb: float = 0.001
    power_comp_out_ub: float = 1e6

    power_pump_in: float = 100  # in kW
    power_pump_in_lb: float = 0.001
    power_pump_in_ub: float = 1e6

    power_pump_out: float = 8000  # in kW
    power_pump_out_lb: float = 0.001
    power_pump_out_ub: float = 1e6

    op_cost_psa: float = 900  # in €/h
    op_cost_psa_lb: float = 0.001
    op_cost_psa_ub: float = 1e6

    price_adsorbent: float = 1e5  # in €
    price_adsorbent_lb: float = 1
    price_adsorbent_ub: float = 1e8

    m_ads: float = 3400  # in kg
    m_ads_lb: float = 0.001
    m_ads_ub: float = 1e8

    V_pv_min: float = 5  # in m^3
    V_pv_min_lb: float = 0.01
    V_pv_min_ub: float = 1000

    Diameter: float = 1  # in m
    Diameter_lb: float = 0.3048  # 1 ft  — Guthrie (1974) correlation lower limit
    Diameter_ub: float = 100.0  # temporarily unbounded — investigating model behaviour

    Height: float = 2  # in m
    Height_lb: float = 1.2192  # 4 ft   — Guthrie (1974) correlation lower limit
    Height_ub: float = 1000.0  # temporarily unbounded — investigating model behaviour

    V_pv: float = 5  # in m^3
    V_pv_lb: float = 0.01
    V_pv_ub: float = 1000

    factor_pressure: float = 1  # dimensionless
    factor_pressure_lb: float = 0.9  # smooth_max constraint enforces Fp >= 1; lb is a loose guard only
    factor_pressure_ub: float = 11.75

    base_cost_pv: float = 9e3  # in USD(1974) — corrected from 9e6 (was 1000x too high, matched old 1e6 coefficient)
    base_cost_pv_lb: float = 1
    base_cost_pv_ub: float = 1e10

    installed_cost_pv: float = 1  # in USD(1974)
    installed_cost_pv_lb: float = 1
    installed_cost_pv_ub: float = 1e6

    fixed_cost_pv: float = 95e3  # in EUR — corrected from 95e6 (was 1000x too high, matched old 1e6 coefficient)
    fixed_cost_pv_lb: float = 1
    fixed_cost_pv_ub: float = 1e8

    base_cost_comp: float = 1  # in €
    base_cost_comp_lb: float = 1
    base_cost_comp_ub: float = 1e9

    F_a_cost_comp: float = 5e5  # in €
    F_a_cost_comp_lb: float = 1
    F_a_cost_comp_ub: float = 1e7

    installed_cost_comp: float = 1  # in € (default, no initialize in code)
    installed_cost_comp_lb: float = 1
    installed_cost_comp_ub: float = 1e8

    fixed_cost_comp: float = 28e5  # in € (default, no initialize in code)
    fixed_cost_comp_lb: float = 1
    fixed_cost_comp_ub: float = 1e8

    base_cost_pump: float = 1e5  # in €
    base_cost_pump_lb: float = 1
    base_cost_pump_ub: float = 1e8

    installed_cost_pump: float = 1e5  # in € (default, no initialize in code)
    installed_cost_pump_lb: float = 1
    installed_cost_pump_ub: float = 1e8

    fixed_cost_pump: float = 1e5  # in € (default, no initialize in code)
    fixed_cost_pump_lb: float = 1
    fixed_cost_pump_ub: float = 1e8


# =============================================================================
# Absorber Surrogate Model — configuration
# =============================================================================


@dataclass
class AbsorberSurrogateParameters:
    # ── Surrogate fit coefficients ────────────────────────────────────────────
    # delta_P [Pa] = surr_dP_a * V_mol_s ** surr_dP_b
    surr_dP_a: float = 15.7606
    surr_dP_b: float = 0.6854

    # T_liq_bot [°C] = surr_T_a - surr_T_b * exp(-surr_T_c * F_co2)
    surr_T_a: float = 48.4033
    surr_T_b: float = 19.3532
    surr_T_c: float = 0.0748

    # H_m [m] = surr_H_a * F_co2 ** surr_H_b
    surr_H_a: float = 1.0528
    surr_H_b: float = 0.3951

    # D_m [m] = surr_D_a * F_co2 ** surr_D_b
    surr_D_a: float = 0.5264
    surr_D_b: float = 0.3951

    # Q_intercooler [W] = F_co2 * (surr_Q_a + surr_Q_b * (T_lean - 35))
    # Physical decomposition: a = baseline heat/mol CO2 at 35°C [W·s/mol]
    #                         b = sensible heat slope [W·s/mol/°C]
    surr_Q_a: float = 1.1879e4
    surr_Q_b: float = 5.1319e3

    # A_intercooler [m²] = surr_A_a * F_co2 ** surr_A_b * T_lean ** surr_A_c
    surr_A_a: float = 2.4795e-2
    surr_A_b: float = 0.9943
    surr_A_c: float = 1.6035

    # ── Seider vessel geometry ────────────────────────────────────────────────
    # L_total = 3 * L_packed  (doc §1)
    # Wall thickness and density (Seider §1)
    t_wall_in: float = 1.25  # in (wall thickness)
    D_extra_in: float = 1.25  # in (D_out = D + 1.25 in)
    rho_steel_lb_in3: float = 0.286  # lb/in³ (carbon steel)
    W_ref_vessel_lb: float = 45000.0  # lb (Seider reference weight)
    in_per_m: float = 39.3701  # in/m

    # ── Seider vessel cost (Seider 2013, §1) ──────────────────────────────────
    vessel_K1: float = 10.5449
    vessel_K2: float = -0.4672
    vessel_K3: float = 0.05482
    f_M_vessel: float = 1.7  # 304 SS material factor

    # ── Packing and distributors (Seider §1) ─────────────────────────────────
    packing_cost_per_ft3: float = 250.0  # $/ft³
    DR_cost_per_ft2: float = 140.0  # $/ft²
    ft3_per_m3: float = 35.3147
    ft2_per_m2: float = 10.7639

    # ── Seider blower cost (Seider 2013, §9) ─────────────────────────────────
    blower_K1: float = 7.0187  # centrifugal blower correlation
    blower_K2: float = 0.79
    P_ref_blower_Hp: float = 100.0  # reference power [Hp]
    n_blowers: int = 2  # two blowers, one per half of gas flow
    f_M_blower: float = 1.0  # carbon steel (no material factor in §9)
    efficiency_blower: float = 0.75  # isentropic efficiency assumption
    R_gas: float = 8.314  # J/mol/K
    T_gas_K: float = 313.15  # 40 °C inlet gas temperature assumption
    P_gas_Pa: float = 1e5  # 1 bar inlet pressure assumption
    HP_to_W: float = 745.7  # W per Hp
    p_elec_USD_per_J: float = 2.315e-8  # $/J (= 83.3 $/MWh, from doc OPEX §3)

    # ── Seider HX cost (Seider 2013, §3) ─────────────────────────────────────
    HX_K1: float = 12.0310
    HX_K2: float = -0.8709
    HX_K3: float = 0.09005
    A_ref_HX_ft2: float = 150.0  # ft² reference area
    f_M_HX: float = 1.7  # 304 SS material factor

    # ── Economics ─────────────────────────────────────────────────────────────
    CEPCI_2024: float = 797.9
    CEPCI_2013: float = 567.0  # Seider (2013) basis year
    # Cooling water price: 0.347 $/GJ  (doc OPEX §1)
    p_cw_USD_per_GJ: float = 0.347
    operating_hours_per_year: float = 8400.0  # hr/yr
    convert_usd_to_euro: float = 0.96
    # Installation factors (doc table)
    f_indirect_labor: float = 0.20
    f_foundation: float = 0.05 * (1 + 1.33)
    f_struct_steel: float = 0.05 * (1 + 0.50)
    f_instruments: float = 0.06 * (1 + 0.40)
    f_piping: float = 0.45 * (1 + 0.50)

    @property
    def f_installation_total(self) -> float:
        return 1 + self.f_indirect_labor + self.f_foundation + self.f_struct_steel + self.f_instruments + self.f_piping

    @property
    def cepci_ratio(self) -> float:
        return self.CEPCI_2024 / self.CEPCI_2013


@dataclass
class AbsorberSurrogateVariables:
    # ── Surrogate outputs ─────────────────────────────────────────────────────
    H_m: float = 5.0
    H_m_lb: float = 0.1
    H_m_ub: float = 50.0
    D_m: float = 2.5
    D_m_lb: float = 0.05
    D_m_ub: float = 25.0
    T_liq_bot_C: float = 47.0
    T_liq_bot_C_lb: float = 30.0
    T_liq_bot_C_ub: float = 60.0
    Q_intercooler_W: float = 1e6
    Q_intercooler_W_lb: float = 1.0
    Q_intercooler_W_ub: float = 1e8
    A_intercooler_m2: float = 400.0
    A_intercooler_m2_lb: float = 1.0
    A_intercooler_m2_ub: float = 1e5

    # ── Vessel sizing intermediates ───────────────────────────────────────────
    W_vessel_lb_val: float = 1.0
    W_vessel_lb_lb: float = 1.0
    W_vessel_lb_ub: float = 1e8
    C_vessel_USD: float = 1e5
    C_vessel_USD_lb: float = 1.0
    C_vessel_USD_ub: float = 1e9
    C_packing_USD: float = 1e5
    C_packing_USD_lb: float = 1.0
    C_packing_USD_ub: float = 1e9
    C_DR_USD: float = 1e4
    C_DR_USD_lb: float = 1.0
    C_DR_USD_ub: float = 1e8
    C_bare_abs_USD: float = 3e5
    C_bare_abs_USD_lb: float = 1.0
    C_bare_abs_USD_ub: float = 1e9

    # ── HX sizing intermediates ───────────────────────────────────────────────
    C_bare_HX_USD: float = 1e5
    C_bare_HX_USD_lb: float = 1.0
    C_bare_HX_USD_ub: float = 1e9

    # ── CAPEX totals ──────────────────────────────────────────────────────────
    CAPEX_abs_EUR: float = 5e5
    CAPEX_abs_EUR_lb: float = 1.0
    CAPEX_abs_EUR_ub: float = 1e10

    # ── Blower sizing ─────────────────────────────────────────────────────────
    delta_P_Pa: float = 350.0
    delta_P_Pa_lb: float = 1.0
    delta_P_Pa_ub: float = 1e5
    P_blower_W: float = 1e4
    P_blower_W_lb: float = 1.0
    P_blower_W_ub: float = 1e8
    P_blower_Hp: float = 15.0
    P_blower_Hp_lb: float = 0.01
    P_blower_Hp_ub: float = 1e5
    C_bare_blowers_USD: float = 1e5
    C_bare_blowers_USD_lb: float = 1.0
    C_bare_blowers_USD_ub: float = 1e9

    # ── OPEX ──────────────────────────────────────────────────────────────────
    OPEX_cw_EUR_yr: float = 1e4
    OPEX_cw_EUR_yr_lb: float = 0.0
    OPEX_cw_EUR_yr_ub: float = 1e9
    OPEX_blower_EUR_yr: float = 1e4
    OPEX_blower_EUR_yr_lb: float = 0.0
    OPEX_blower_EUR_yr_ub: float = 1e9


# =============================================================================
# MEA amine wash (CO2 capture) — Kim & Léonard (2025)
# =============================================================================


@dataclass
class AmineWashParameters:
    # Shortcut-correlation parameters — Kim & Léonard (2025), Table 14
    # TEC [M€2023] = tec_α + (tec_β·y_CO2^n + tec_γ)·F^m
    # y_CO2 = mol fraction (0.05–0.50),  F in 10³ Nm³/h
    tec_alpha: float = 2.1673
    tec_beta: float = 0.8092
    tec_gamma: float = -0.00332
    tec_n: float = 0.5291
    tec_m: float = 0.8391

    # Specific reboiler duty [GJ/tCO2] = α·exp(n·y_CO2) + β·exp(m·y_CO2)
    reb_alpha: float = 1.471
    reb_beta: float = 3.560
    reb_n: float = -35.83
    reb_m: float = -0.0158

    # Specific electrical duty [kWh/tCO2]
    elec_alpha: float = 10.420
    elec_beta: float = 2.164
    elec_n: float = -23.49
    elec_m: float = -1.2350

    # Specific cooling duty [GJ/tCO2]
    cool_alpha: float = 10.040
    cool_beta: float = 2.905
    cool_n: float = -33.73
    cool_m: float = 0.2108

    # Lang/Hand installation factor: TEC (= K_B, purchased equipment) → K_I
    # (installed module cost, including piping, civil, instrumentation,
    # electrical, insulation, labour).
    #
    # Equipment mix (Kim & Léonard 2025, Fig. A1, capture plant excl. compression):
    #   Absorber + stripper + washer/separator  ≈ 73 %  → f_H = 4.0
    #   Heat exchangers (lean/rich, coolers)    ≈ 18 %  → f_H = 3.5
    #   Pump + blower                           ≈  9 %  → f_H = 2.5
    #
    #   f_H_weighted = 0.73×4.0 + 0.18×3.5 + 0.09×2.5
    #                = 2.92 + 0.63 + 0.23 = 3.78  → rounded to 3.8
    #
    # CAPEX = 2 × f_installed × TEC  (the ×2 Guthrie overhead factor covering
    # buildings, infrastructure, engineering, working capital and startup is
    # applied inside AmineWash.build_constraints, NOT here).
    f_installed: float = 3.8

    # Utility costs (Nwaoha 2018, updated to 2023 via CEPCI)
    # LP steam [€/GJ] — electricity-derived: price_power / (3.6e-3 × η_boiler)
    # = 0.08 / (3.6e-3 × 0.95) ≈ 23.4 €/GJ (electric boiler). Single source of
    # truth for steam cost across the flowsheet.
    steam_cost_eur_per_gj: float = 23.4
    cooling_cost_eur_per_gj: float = 0.23  # cooling water [€/GJ]

    # MEA chemical makeup — Fig. 7b benchmark value
    mea_makeup_eur_per_tco2: float = 0.87  # €/tCO2

    # Annual maintenance as fraction of CAPEX — matches maintenance_factor_static
    # used by HeatEXModel and FlashModel in the rest of the flowsheet
    maintenance_factor: float = 0.03  # 3 %/yr


@dataclass
class AmineWashVariables:
    # Benchmark initial values: 0.7 Mt/y captured, 11.5 % CO2, 8215 h/y
    nf_CO2_in: float = 597.0
    nf_CO2_in_lb: float = 10.0
    nf_CO2_in_ub: float = 15000.0
    nf_inert_in: float = 4600.0
    nf_inert_in_lb: float = 10.0
    nf_inert_in_ub: float = 200000.0
    nf_CO2_out: float = 59.7
    nf_CO2_out_lb: float = 0.0
    nf_CO2_out_ub: float = 5000.0
    nf_CO2_cap: float = 537.0
    nf_CO2_cap_lb: float = 0.01
    nf_CO2_cap_ub: float = 15000.0
    y_CO2_in: float = 0.115
    y_CO2_in_lb: float = 0.05
    y_CO2_in_ub: float = 0.50
    F_thousands: float = 396.0
    F_thousands_lb: float = 4.03
    F_thousands_ub: float = 1613.81
    scale_tph: float = 85.2
    scale_tph_lb: float = 0.1
    scale_tph_ub: float = 1500.0
    TEC: float = 42.0
    TEC_lb: float = 0.01
    TEC_ub: float = 10000.0
    CAPEX: float = 209e6
    CAPEX_lb: float = 1e4
    CAPEX_ub: float = 1e12
    Q_reb_spec: float = 3.58
    Q_reb_spec_lb: float = 0.5
    Q_reb_spec_ub: float = 10.0
    W_elec_spec: float = 2.5
    W_elec_spec_lb: float = 0.1
    W_elec_spec_ub: float = 20.0
    Q_cool_spec: float = 3.5
    Q_cool_spec_lb: float = 0.1
    Q_cool_spec_ub: float = 20.0
    Q_reb_kW: float = 85000.0
    Q_reb_kW_lb: float = 1.0
    Q_reb_kW_ub: float = 1e9
    W_elec_kW: float = 12000.0
    W_elec_kW_lb: float = 1.0
    W_elec_kW_ub: float = 1e8
    Q_cool_kW: float = 85000.0
    Q_cool_kW_lb: float = 1.0
    Q_cool_kW_ub: float = 1e9
    op_cost: float = 6300.0
    op_cost_lb: float = 0.0
    op_cost_ub: float = 1e9
