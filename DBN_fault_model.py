from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


#define graph (DAG)
dag = [
    ("Aging", "ShortCircuit"), #aging --> shortcircuit
    ("Aging", "ThermalRunaway"),
    ("ShortCircuit", "BatteryFailure"), #shortcurcuit --> batteryfailure
    ("ThermalRunaway", "BatteryFailure") #TR --> batteryfailure
]
model = DiscreteBayesianNetwork(dag)


cpd_aging = TabularCPD(
    variable="Aging", 
    variable_card=2, 
    values = [[0.9], [0.1]]
    #no evidence because it is root cause
)

cpd_short = TabularCPD(
    variable="ShortCircuit", 
    variable_card=2,
    values =[
        [0.95, 0.7], # no short
        [0.05, 0.3],  # short
    ],
    evidence=["Aging"],
    evidence_card=[2],
)

cpd_thermal = TabularCPD(
    variable="ThermalRunaway", 
    variable_card=2,
    values =[
        [0.97, 0.6], #no TR
        [0.03, 0.4]  #TR
    ],
    evidence=["Aging"],
    evidence_card=[2]
)

cpd_battery = TabularCPD(
    variable="BatteryFailure", 
    variable_card=2,
    values =[
        [0.99, 0.3, 0.2, 0.01],  # no failure
        [0.01, 0.7, 0.8, 0.99],  # failure
    ],
    evidence=["ShortCircuit", "ThermalRunaway"],
    evidence_card=[2, 2]
)

model.add_cpds(cpd_aging, cpd_short, cpd_thermal, cpd_battery)
# print(cpd_thermal)


# inference
infer = VariableElimination(model)

result = infer.query(["BatteryFailure"], evidence={"Aging": 1})
print(result)

# print(model.get_cpds())
# for cpd in model.get_cpds():
#     print(cpd)
