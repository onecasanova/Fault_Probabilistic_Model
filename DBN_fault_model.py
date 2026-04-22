from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD


# model = DiscreteBayesianNetwork([
#     ("Aging", "ShortCircuit"),
#     ("Aging", "ThermalRunaway"),
#     ("ShortCircuit", "BatteryFailure"),
#     ("ThermalRunaway", "BatteryFailure")
# ])



cpd_aging = TabularCPD("Aging", 2, [[0.9], [0.1]])

cpd_short = TabularCPD(
    variable="ShortCircuit", 
    variable_card=2,
    values =[
        [0.95, 0.7], # no short
        [0.05, 0.3],  # short
        ],
    evidence=["Aging"],
    evidence_card=[2],
    state_names={'ShortCircuit': ['no', 'yes'], 'Aging': ['yes', 'no']}
)

cpd_thermal = TabularCPD(
    "ThermalRunaway", 2,
    [[0.97, 0.6],
     [0.03, 0.4]],
    evidence=["Aging"],
    evidence_card=[2]
)

cpd_battery = TabularCPD(
    "BatteryFailure", 2,
    [
        [0.99, 0.3, 0.2, 0.01],  # no failure
        [0.01, 0.7, 0.8, 0.99],  # failure
    ],
    evidence=["ShortCircuit", "ThermalRunaway"],
    evidence_card=[2, 2]
)

# model.add_cpds(cpd_aging, cpd_short, cpd_thermal, cpd_battery)

print(cpd_thermal)