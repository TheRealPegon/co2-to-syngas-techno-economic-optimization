import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import math
from prettytable import PrettyTable

model = pyo.ConcreteModel(name="CAPEX+OPEXmin")

# 1. Price
model.convert_usd_to_euro = pyo.Param(initialize=0.96)
model.price_CO = pyo.Param(initialize=model.convert_usd_to_euro * 0.81 * 28.01 / 1000)  # in €/mol
model.price_H2 = pyo.Param(initialize=model.convert_usd_to_euro * 3 * 2.016 / 1000)  # in €/mol
model.price_CO2 = pyo.Param(initialize=0.05 * 44.01 / 1000)  # in €/mol
model.price_power = pyo.Param(initialize=0.08)  # in €/kWh

# 2. Electrolyser
#  2.1.Parameters
model.current_density = pyo.Param(initialize=3000)  # in A/m^2 = 300 mA/cm^2
model.voltage = pyo.Param(initialize=3)  # in V
model.area_elec = pyo.Param(initialize=5000)  # in m^2    (0.5 m^2 x 10000 cells)
model.current = pyo.Param(initialize=model.area_elec * model.current_density)  # in A
model.power_elec = pyo.Param(initialize=model.current * model.voltage / 1000)  # in kW
model.op_cost_elec = pyo.Param(initialize=model.power_elec * model.price_power)  # in €/h
# Capital Cost (Peters-Timmerhaus-West-Methode)
model.base_cost_elec = pyo.Param(initialize=327)  # in USD/kW
model.direct_cost_factor_elec = pyo.Param(initialize=3.6)
model.indirect_cost_factor_elec = pyo.Param(initialize=1.44)
model.fixed_cost_elec = pyo.Param(
    initialize=model.convert_usd_to_euro
    * (model.direct_cost_factor_elec + model.indirect_cost_factor_elec)
    * model.base_cost_elec
    * model.power_elec
)
#  2.2 Variables
model.nf_CO2_elec_in = pyo.Var(bounds=(0.001, 10000), initialize=160)  # in mol/s
model.nf_CO_elec_in = pyo.Var(bounds=(0.001, 10000), initialize=10)  # in mol/s
model.nf_CO2_elec_out = pyo.Var(bounds=(0.001, 10000), initialize=1000)  # in mol/s
model.nf_CO_elec_out = pyo.Var(bounds=(0.001, 10000), initialize=10)  # in mol/s
model.nf_CO_elec = pyo.Var(bounds=(0.001, 1000), initialize=10)  # in mol/s
model.nf_H2_elec_out = pyo.Var(bounds=(0.001, 1000), initialize=10)  # in mol/s
model.lamda = pyo.Var(bounds=(1, 7), initialize=2)
model.FE_CO = pyo.Var(bounds=(0.0001, 1), initialize=0.9)
model.FE_H2 = pyo.Var(bounds=(0.0001, 1), initialize=0.1)
#  2.3 Constraint
# Electrical Equation
model.current_const1 = pyo.Constraint(expr=model.nf_CO_elec == model.current * model.FE_CO / (2 * 96485))
model.current_const2 = pyo.Constraint(expr=model.nf_H2_elec_out == model.current * model.FE_H2 / (2 * 96485))
model.FE_CO_const = pyo.Constraint(expr=model.FE_CO == (model.lamda**2) / ((model.lamda**2) + 1.5))
model.FE_H2_const = pyo.Constraint(
    expr=model.FE_H2 == -0.0003 * (model.lamda**3) + 0.0102 * (model.lamda**2) - 0.0995 * model.lamda + 0.3129
)
# Material Balances (C-balance)
model.lamda_definition = pyo.Constraint(expr=model.nf_CO2_elec_in == model.lamda * model.nf_CO_elec)
model.balance1 = pyo.Constraint(
    expr=model.nf_CO2_elec_in + model.nf_CO_elec_in == model.nf_CO2_elec_out + model.nf_CO_elec_out
)
model.balance2 = pyo.Constraint(expr=model.nf_CO2_elec_out <= model.nf_CO2_elec_in)
model.balance3 = pyo.Constraint(expr=model.nf_CO2_elec_out == model.nf_CO2_elec_in - model.nf_CO_elec)
model.balance4 = pyo.Constraint(expr=model.nf_CO_elec_out == model.nf_CO_elec_in + model.nf_CO_elec)

# 3. Pressure Swing Adsorption
#  3.1. Parameters
model.R = pyo.Param(initialize=8.314)  # in J/(mol K)
model.T = pyo.Param(initialize=298)  # in K
model.y_CO_product = pyo.Param(initialize=0.996)  # targeted CO purity
model.y_CO2_product = pyo.Param(initialize=0.004)
model.y_CO_psaH_out = pyo.Param(initialize=model.y_CO_product)
model.y_CO2_psaH_out = pyo.Param(initialize=model.y_CO2_product)
model.y_CO_sw = pyo.Param(initialize=model.y_CO_product)
model.y_CO2_sw = pyo.Param(initialize=model.y_CO2_product)
model.p_L = pyo.Param(initialize=100)  # in kPa
model.q_CO2_s = pyo.Param(initialize=6.043)  # in mol/kg
model.Bs_CO2 = pyo.Param(initialize=0.08008)  # in 1/kPa
model.ns_CO2 = pyo.Param(initialize=1.887)
model.Henry_constant_CO2 = pyo.Param(initialize=model.q_CO2_s * model.Bs_CO2)  # in mol/(kg kPa)
model.q_CO_s = pyo.Param(initialize=3.890)  # in mol/kg
model.Bs_CO = pyo.Param(initialize=0.002123)  # in 1/kPa
model.ns_CO = pyo.Param(initialize=1.220)
model.bulk_density = pyo.Param(initialize=680)
model.voidage = pyo.Param(initialize=0.4)
model.efficiency_comp = pyo.Param(initialize=0.78)
model.saturation_adsorbent = pyo.Param(initialize=0.95)
model.t_ads = pyo.Param(initialize=-(0.4977 + log(1 - model.saturation_adsorbent)) / (math.pi * 0.0167))
model.price_zeoperkg = pyo.Param(initialize=1.05)  # in €/kg
# Guthries parameter for pressure vessel
model.factor_material = pyo.Param(initialize=2.3)  # stainless 304 Clad
model.factor_installation = pyo.Param(initialize=2.921)
model.price_index = pyo.Param(initialize=1)  # price index
model.indirect_cost_factor_pv = pyo.Param(initialize=1.38)
# Guthries parameter for compressor
model.factor_installation_comp = pyo.Param(initialize=1.565)
model.indirect_cost_factor_comp = pyo.Param(initialize=1.38)
#  3.2. Variables
model.nf_CO2_feed = pyo.Var(bounds=(0.001, 10000), initialize=100)  # in mol/s
model.p_H = pyo.Var(bounds=(100, 2000), initialize=1000)  # in kPa
model.nf_CO2_psaH_in = pyo.Var(bounds=(0.001, 10000), initialize=1000)  # in mol/s
model.nf_CO_psaH_in = pyo.Var(bounds=(0.001, 10000), initialize=200)  # in mol/s
model.y_CO2_psaH_in = pyo.Var(bounds=(0.0001, 1), initialize=0.6)
model.y_CO_psaH_in = pyo.Var(bounds=(0.0001, 1), initialize=0.4)
model.nf_CO2_psaH_out = pyo.Var(bounds=(0.001, 10000), initialize=20)  # in mol/s
model.nf_CO_psaH_out = pyo.Var(bounds=(0.001, 10000), initialize=1000)  # in mol/s
model.nf_CO2_product = pyo.Var(bounds=(0.001, 10000), initialize=0.001)  # in mol/s
model.nf_CO_product = pyo.Var(bounds=(0.001, 10000), initialize=100)  # in mol/s
model.nf_CO2_sw = pyo.Var(bounds=(0.001, 10000), initialize=0.1)  # in mol/s
model.nf_CO_sw = pyo.Var(bounds=(0.001, 10000), initialize=0.1)  # in mol/s
model.nf_CO2_psaL_out = pyo.Var(bounds=(0.001, 10000), initialize=100)  # in mol/s
model.nf_CO_psaL_out = pyo.Var(bounds=(0.001, 10000), initialize=0.1)  # in mol/s
model.nf_CO2_LR = pyo.Var(bounds=(0.001, 10000), initialize=1)  # in mol/s
model.nf_CO_LR = pyo.Var(bounds=(0.001, 10000), initialize=1)  # in mol/s
model.nf_CO2_HR = pyo.Var(bounds=(0.001, 10000), initialize=100)  # in mol/s
model.nf_CO_HR = pyo.Var(bounds=(0.001, 10000), initialize=1)  # in mol/s
model.mf_ads = pyo.Var(bounds=(0.001, 10000), initialize=100)  # in kg/s
model.q_CO2_H = pyo.Var(bounds=(0.0001, 6.092), initialize=5)  # in mol/kg
model.q_CO2_L = pyo.Var(bounds=(0.000001, 6.092), initialize=1)  # in mol/kg
model.q_CO_H = pyo.Var(bounds=(0.000001, 3.966), initialize=0.15)  # in mol/kg
model.q_CO_L = pyo.Var(bounds=(0.000001, 3.966), initialize=0.1)  # in mol/kg
model.power_comp_in = pyo.Var(bounds=(0.001, 100000), initialize=8000)  # in kW
model.power_comp_out = pyo.Var(bounds=(0.001, 100000), initialize=8000)  # in kW
model.op_cost_psa = pyo.Var(bounds=(0.001, 1000000), initialize=900)  # in €/h
model.price_adsorbent = pyo.Var(bounds=(1, 100000000), initialize=10000)  # in €
model.m_ads = pyo.Var(bounds=(0.001, 1000000), initialize=3400)  # in kg
model.V_pv_min = pyo.Var(bounds=(0.01, 1000), initialize=5)  # in m^3
model.Diameter = pyo.Var(bounds=(0.3048, 3.048), initialize=1)  # in m
model.Height = pyo.Var(bounds=(1.2192, 30.48), initialize=2)  # in m
model.V_pv = pyo.Var(bounds=(0.01, 1000), initialize=5)  # in m^3
model.factor_pressure = pyo.Var(bounds=(1, 11.75), initialize=1)
model.base_cost_pv = pyo.Var(bounds=(1, 1000000000), initialize=500000)  # in €
model.installed_cost_pv = pyo.Var(bounds=(1, 1000000000))  # in €
model.fixed_cost_pv = pyo.Var(bounds=(1, 1000000000))  # in €
model.base_cost_comp = pyo.Var(bounds=(1, 10000000), initialize=500000)  # in €
model.F_a_cost_comp = pyo.Var(bounds=(1, 10000000), initialize=500000)  # in €
model.installed_cost_comp = pyo.Var(bounds=(1, 100000000))  # in €
model.fixed_cost_comp = pyo.Var(bounds=(1, 100000000))  # in €
#  3.3 Constraint
# mole fration
model.y_CO2_psaH_in_Const = pyo.Constraint(
    expr=model.y_CO2_psaH_in == model.nf_CO2_psaH_in / (model.nf_CO2_psaH_in + model.nf_CO_psaH_in)
)
model.y_CO_psaH_in_Const = pyo.Constraint(
    expr=model.y_CO_psaH_in == model.nf_CO_psaH_in / (model.nf_CO2_psaH_in + model.nf_CO_psaH_in)
)
model.y_CO2_psaH_out_const = pyo.Constraint(
    expr=model.y_CO2_product == model.nf_CO2_psaH_out / (model.nf_CO2_psaH_out + model.nf_CO_psaH_out)
)
model.y_CO_psaH_out_const = pyo.Constraint(
    expr=model.y_CO_product == model.nf_CO_psaH_out / (model.nf_CO2_psaH_out + model.nf_CO_psaH_out)
)
model.y_CO2_product_const = pyo.Constraint(
    expr=model.y_CO2_product == model.nf_CO2_product / (model.nf_CO2_product + model.nf_CO_product)
)
model.y_CO_product_const = pyo.Constraint(
    expr=model.y_CO_product == model.nf_CO_product / (model.nf_CO2_product + model.nf_CO_product)
)
# Sips-model
model.q_CO2_H_const = pyo.Constraint(
    expr=model.q_CO2_H
    == model.q_CO2_s
    * (model.Bs_CO2 * model.y_CO2_psaH_in * model.p_H) ** (1 / model.ns_CO2)
    / (
        1
        + (model.Bs_CO2 * model.y_CO2_psaH_in * model.p_H) ** (1 / model.ns_CO2)
        + (model.Bs_CO * model.y_CO_psaH_in * model.p_H) ** (1 / model.ns_CO)
    )
)
model.q_CO_H_const = pyo.Constraint(
    expr=model.q_CO_H
    == model.q_CO_s
    * (model.Bs_CO * model.y_CO_psaH_in * model.p_H) ** (1 / model.ns_CO)
    / (
        1
        + (model.Bs_CO2 * model.y_CO2_psaH_in * model.p_H) ** (1 / model.ns_CO2)
        + (model.Bs_CO * model.y_CO_psaH_in * model.p_H) ** (1 / model.ns_CO)
    )
)
model.q_CO2_L_const = pyo.Constraint(
    expr=model.q_CO2_L
    == model.q_CO2_s
    * (model.Bs_CO2 * model.y_CO2_psaH_out * model.p_L) ** (1 / model.ns_CO2)
    / (
        1
        + (model.Bs_CO2 * model.y_CO2_psaH_out * model.p_L) ** (1 / model.ns_CO2)
        + (model.Bs_CO * model.y_CO_psaH_out * model.p_L) ** (1 / model.ns_CO)
    )
)
model.q_CO_L_const = pyo.Constraint(
    expr=model.q_CO_L
    == model.q_CO_s
    * (model.Bs_CO * model.y_CO_psaH_out * model.p_L) ** (1 / model.ns_CO)
    / (
        1
        + (model.Bs_CO2 * model.y_CO2_psaH_out * model.p_L) ** (1 / model.ns_CO2)
        + (model.Bs_CO * model.y_CO_psaH_out * model.p_L) ** (1 / model.ns_CO)
    )
)
model.q_const1 = pyo.Constraint(expr=model.q_CO2_H >= model.q_CO_H)
# sweep gas
model.nf_CO2_sw_const = pyo.Constraint(
    expr=model.nf_CO2_sw >= model.y_CO2_sw * model.Henry_constant_CO2 * model.p_L * model.mf_ads
)
model.nf_CO_sw_const = pyo.Constraint(
    expr=model.nf_CO_sw >= model.y_CO_sw * model.Henry_constant_CO2 * model.p_L * model.mf_ads
)
# Retentat
model.nf_CO2_LR_const = pyo.Constraint(
    expr=model.nf_CO2_LR
    == model.mf_ads
    * (
        model.q_CO2_L * model.saturation_adsorbent
        + model.voidage * model.y_CO2_psaH_out * model.p_L * 1000 / (model.bulk_density * model.R * model.T)
    )
)
model.nf_CO_LR_const = pyo.Constraint(
    expr=model.nf_CO_LR
    == model.mf_ads
    * (
        model.q_CO_L * model.saturation_adsorbent
        + model.voidage * model.y_CO_psaH_out * model.p_L * 1000 / (model.bulk_density * model.R * model.T)
    )
)
model.nf_CO2_HR_const = pyo.Constraint(
    expr=model.nf_CO2_HR
    == model.mf_ads
    * (
        model.q_CO2_H * model.saturation_adsorbent
        + model.voidage * model.y_CO2_psaH_in * model.p_H * 1000 / (model.bulk_density * model.R * model.T)
    )
)
model.nf_CO_HR_const = pyo.Constraint(
    expr=model.nf_CO_HR
    == model.mf_ads
    * (
        model.q_CO_H * model.saturation_adsorbent
        + model.voidage * model.y_CO_psaH_in * model.p_H * 1000 / (model.bulk_density * model.R * model.T)
    )
)
# Connector Elctrolyser to PSA
model.connector1 = pyo.Constraint(expr=model.nf_CO2_psaH_in == model.nf_CO2_elec_out)
model.connector2 = pyo.Constraint(expr=model.nf_CO_psaH_in == model.nf_CO_elec_out)
# Balance
model.nf_CO2_psaH_out_const = pyo.Constraint(
    expr=model.nf_CO2_psaH_out == model.nf_CO2_psaH_in - (model.nf_CO2_HR - model.nf_CO2_LR)
)
model.nf_CO_psaH_out_const = pyo.Constraint(
    expr=model.nf_CO_psaH_out == model.nf_CO_psaH_in - (model.nf_CO_HR - model.nf_CO_LR)
)
model.nf_CO2_product_const = pyo.Constraint(expr=model.nf_CO2_product == model.nf_CO2_psaH_out - model.nf_CO2_sw)
model.nf_CO_product_const = pyo.Constraint(expr=model.nf_CO_product == model.nf_CO_psaH_out - model.nf_CO_sw)
model.nf_CO2_psaL_out_const = pyo.Constraint(
    expr=model.nf_CO2_psaL_out == model.nf_CO2_sw + (model.nf_CO2_HR - model.nf_CO2_LR)
)
model.nf_CO_psaL_out_const = pyo.Constraint(
    expr=model.nf_CO_psaL_out == model.nf_CO_sw + (model.nf_CO_HR - model.nf_CO_LR)
)
# Connector
model.connector3 = pyo.Constraint(expr=model.nf_CO2_psaL_out == model.nf_CO2_elec_in - model.nf_CO2_feed)
model.connector4 = pyo.Constraint(expr=model.nf_CO_psaL_out == model.nf_CO_elec_in)
# Power
model.power_comp_out_const = pyo.Constraint(
    expr=(
        model.power_comp_out
        == (model.nf_CO2_psaH_in + model.nf_CO_psaH_in) * model.R * model.T * log(model.p_H / 100) / 1000
    )
)  # divided  by 1000 to convert into kW
model.power_comp_in_const = pyo.Constraint(
    expr=model.power_comp_in == (model.power_comp_out / model.efficiency_comp)
)  # divided  by 1000 to convert into kW
model.op_cost_psa_const = pyo.Constraint(expr=model.op_cost_psa == model.power_comp_in * model.price_power)
# Capital cost adsorbent
model.m_ads_const = pyo.Constraint(expr=model.m_ads == model.mf_ads * model.t_ads / (2 * model.saturation_adsorbent))
model.price_adsorbent_const = pyo.Constraint(
    expr=model.price_adsorbent == model.m_ads * model.price_zeoperkg
)  # multiplied by 2, because 2 pressure vessel necessary for PSA
# Capital cost pressure vessel
model.V_pv_min_const = pyo.Constraint(expr=model.V_pv_min == model.m_ads / model.bulk_density)
model.V_pv_const1 = pyo.Constraint(expr=model.V_pv == math.pi * (model.Diameter**2) * model.Height / 4)
model.V_pv_const2 = pyo.Constraint(expr=model.V_pv >= model.V_pv_min)
model.factor_pressure_const = pyo.Constraint(
    expr=model.factor_pressure == 2e-13 * model.p_H**3 - 1e-08 * model.p_H**2 + 0.0006 * model.p_H + 0.9081
)
model.base_cost_pv_const = pyo.Constraint(
    expr=model.base_cost_pv
    == 10**6 * model.convert_usd_to_euro * (model.Height / 1.2192) ** 0.81 * (model.Diameter / 0.9144) ** 1.05
)
model.installed_cost_pv_const = pyo.Constraint(
    expr=model.installed_cost_pv
    == model.factor_installation
    * model.factor_material
    * model.factor_pressure
    * model.price_index
    * model.base_cost_pv
)
model.fixed_cost_pv_const = pyo.Constraint(
    expr=model.fixed_cost_pv == model.indirect_cost_factor_pv * model.installed_cost_pv * 2
)  # multiplied by 2, because 2 pressure vessel necessary for PSA
# Capiltal cost compressor
model.base_cost_comp_const = pyo.Constraint(
    expr=model.base_cost_comp == 23000 * model.convert_usd_to_euro * (model.power_comp_out / 73.55) ** 0.77
)
model.F_a_cost_comp_const = pyo.Constraint(
    expr=model.F_a_cost_comp == 776.5 * model.convert_usd_to_euro * model.base_cost_comp**0.493
)
model.installed_cost_comp_const = pyo.Constraint(
    expr=model.installed_cost_comp == model.factor_installation_comp * (model.base_cost_comp + model.F_a_cost_comp)
)
model.fixed_cost_comp_const = pyo.Constraint(
    expr=model.fixed_cost_comp == model.indirect_cost_factor_comp * model.installed_cost_comp
)

# 4. Operating Time and Lifetime
model.lifetime = pyo.Param(initialize=20)  # in years
model.operating_day_per_year = pyo.Param(initialize=350)  # in days/a
model.operating_hour = pyo.Param(initialize=model.lifetime * model.operating_day_per_year * 24)  # in h

# 5. CAPEX
model.CAPEX = pyo.Var(bounds=(0.0001, 1000000000))  # in €
model.CAPEX_const = pyo.Constraint(
    expr=model.CAPEX == model.fixed_cost_elec + model.fixed_cost_pv + model.price_adsorbent + model.fixed_cost_comp
)

model.objective = pyo.Objective(
    expr=model.CAPEX
    + model.operating_hour
    * (
        model.op_cost_elec
        + model.op_cost_psa
        + 3600
        * (
            model.nf_CO2_feed * model.price_CO2
            - model.nf_CO_product * model.price_CO
            - model.nf_H2_elec_out * model.price_H2
        )
    ),
    sense=pyo.minimize,
)

# Count Variables
vars = list(model.component_data_objects(pyo.Var, active=True))
# Count Parameters
params = list(model.component_data_objects(pyo.Param, active=True))
# Count Constraints
cons = list(model.component_data_objects(pyo.Constraint, active=True))

print(f"--- Model Statistics ---")
print(f"Number of Variables:   {len(vars)}")
print(f"Number of Parameters:  {len(params)}")
print(f"Number of Constraints: {len(cons)}")

solver = pyo.SolverFactory("multistart").solve(model)

print(f"net value after 20 years: {-model.objective.expr()} €")
print(f"CAPEX: {model.CAPEX.value} €")
print(f"Profit/a: {-(model.objective.expr() - model.CAPEX.value)/model.lifetime} €/a")
print(
    f"OPEX/a: {350 * 24 * (3600 * model.nf_CO2_feed.value * model.price_CO2 + model.op_cost_elec.value + model.op_cost_psa.value) }  €/a"
)
print(f"Profit/h: {-(model.objective.expr() - model.CAPEX.value)/(model.operating_hour)} €/h")
print(
    f"OPEX/h: {3600 * model.nf_CO2_feed.value * model.price_CO2 + model.op_cost_elec.value + model.op_cost_psa.value}  €/h"
)
print(f"CO2 cost per hour: {3600 * model.nf_CO2_feed.value * model.price_CO2}  €/h")
print(f"CO sales per hour: {3600 * model.nf_CO_product.value * model.price_CO}  €/h")
print(f"H2 sales per hour: {3600 * model.nf_H2_elec_out.value * model.price_H2}  €/h")

print("Electrolyser")
table_elec = PrettyTable()
table_elec.field_names = ["     Variables     ", "                   Values                  ", "  Unit  "]
table_elec.add_row(["lamda", model.lamda.value, "-"])
table_elec.add_row(["nf_CO2_feed", model.nf_CO2_feed.value, "mol/s"])
table_elec.add_row(["nf_CO2_elec_in ", model.nf_CO2_elec_in.value, "mol/s"])
table_elec.add_row(["nf_CO_elec", model.nf_CO_elec.value, "mol/s"])
table_elec.add_row(["nf_CO_elec_in", model.nf_CO_elec_in.value, "mol/s"])
table_elec.add_row(["nf_CO2_elec_out", model.nf_CO2_elec_out.value, "mol/s"])
table_elec.add_row(["nf_CO_elec_out", model.nf_CO_elec_out.value, "mol/s"])
table_elec.add_row(["nf_H2_elec_out", model.nf_H2_elec_out.value, "mol/s"])
table_elec.add_row(["FE_CO", model.FE_CO.value, "-"])
table_elec.add_row(["FE_H2", model.FE_H2.value, "-"])
table_elec.add_row(["op_cost_elec", model.op_cost_elec.value, "€/h"])
table_elec.add_row(["fixed_cost_elec", model.fixed_cost_elec.value, "€"])
print(table_elec)

print("PSA")
table_psa = PrettyTable()
table_psa.field_names = ["     Variables     ", "                   Values                  ", "  Unit  "]
table_psa.add_row(["mf_ads", model.mf_ads.value, "kg/s"])
table_psa.add_row(["p_H", model.p_H.value, "kPa"])
table_psa.add_row(["nf_CO2_psaH_in", model.nf_CO2_psaH_in.value, "mol/s"])
table_psa.add_row(["nf_CO_psaH_in", model.nf_CO_psaH_in.value, "mol/s"])
table_psa.add_row(["y_CO2_psaH_in", model.y_CO2_psaH_in.value, "-"])
table_psa.add_row(["y_CO_psaH_in", model.y_CO_psaH_in.value, "-"])
table_psa.add_row(["nf_CO2_psaH_out", model.nf_CO2_psaH_out.value, "mol/s"])
table_psa.add_row(["nf_CO_psaH_out", model.nf_CO_psaH_out.value, "mol/s"])
table_psa.add_row(["y_CO2_psaH_out", model.y_CO2_psaH_out.value, "-"])
table_psa.add_row(["y_CO_psaH_out", model.y_CO_psaH_out.value, "-"])
table_psa.add_row(["nf_CO2_sw", model.nf_CO2_sw.value, "mol/s"])
table_psa.add_row(["nf_CO_sw", model.nf_CO_sw.value, "mol/s"])
table_psa.add_row(["nf_CO2_psaL_out", model.nf_CO2_psaL_out.value, "mol/s"])
table_psa.add_row(["nf_CO_psaL_out", model.nf_CO_psaL_out.value, "mol/s"])
table_psa.add_row(["q_CO2_H", model.q_CO2_H.value, "mol/kg"])
table_psa.add_row(["q_CO_H", model.q_CO_H.value, "mol/kg"])
table_psa.add_row(["q_CO2_L", model.q_CO2_L.value, "mol/kg"])
table_psa.add_row(["q_CO_L", model.q_CO_L.value, "mol/kg"])
table_psa.add_row(["nf_CO2_HR", model.nf_CO2_HR.value, "mol/s"])
table_psa.add_row(["nf_CO_HR", model.nf_CO_HR.value, "mol/s"])
table_psa.add_row(["nf_CO2_LR", model.nf_CO2_LR.value, "mol/s"])
table_psa.add_row(["nf_CO_LR", model.nf_CO_LR.value, "mol/s"])
table_psa.add_row(["nf_CO2_product", model.nf_CO2_product.value, "mol/s"])
table_psa.add_row(["nf_CO_product", model.nf_CO_product.value, "mol/s"])
table_psa.add_row(["power_comp_in", model.power_comp_in.value, "kW"])
table_psa.add_row(["power_comp_out", model.power_comp_out.value, "kW"])
table_psa.add_row(["op_cost_psa", model.op_cost_psa.value, "€/h"])
table_psa.add_row(["Diameter", model.Diameter.value, "€"])
table_psa.add_row(["Height", model.Height.value, "€"])
table_psa.add_row(["Fp", model.factor_pressure.value, "-"])
table_psa.add_row(["fixed_cost_pv", model.fixed_cost_pv.value, "€"])
table_psa.add_row(["fixed_cost_comp", model.fixed_cost_comp.value, "€"])
table_psa.add_row(["cost_adsorbent", model.price_adsorbent.value, "€"])
print(table_psa)
