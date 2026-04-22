from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import matplotlib.pyplot as plt
import os


def sample_binary(p_true):
    return 1 if np.random.rand() < p_true else 0


# Create DBN
model = DBN()
model.add_edges_from([
    (("Aging", 0), ("ShortCircuit", 0)),
    (("Aging", 0), ("ThermalRunaway", 0)),
    (("ShortCircuit", 0), ("BatteryFailure", 0)),
    (("ThermalRunaway", 0), ("BatteryFailure", 0)),
    (("Aging", 0), ("Aging", 1)),
    (("ShortCircuit", 0), ("ShortCircuit", 1)),
    (("ThermalRunaway", 0), ("ThermalRunaway", 1)),
    (("ShortCircuit", 1), ("BatteryFailure", 1)),
    (("ThermalRunaway", 1), ("BatteryFailure", 1)),
    (("BatteryFailure", 0), ("BatteryFailure", 1)),
])

cpd_aging_0 = TabularCPD(("Aging", 0), 2, [[0.9], [0.1]])

cpd_short_0 = TabularCPD(
    ("ShortCircuit", 0), 2,
    [[0.98, 0.80], #no short
     [0.02, 0.20]], #short
    evidence=[("Aging", 0)],
    evidence_card=[2]
)

cpd_thermal_0 = TabularCPD(
    ("ThermalRunaway", 0), 2,
    [[0.97, 0.70], #no TR
     [0.03, 0.30]], #TR
    evidence=[("Aging", 0)],
    evidence_card=[2]
)

cpd_battery_0 = TabularCPD(
    ("BatteryFailure", 0), 2,
    [[0.99, 0.30, 0.20, 0.01],
     [0.01, 0.70, 0.80, 0.99]],
    evidence=[("ShortCircuit", 0), ("ThermalRunaway", 0)],
    evidence_card=[2, 2]
)

cpd_aging_t = TabularCPD(
    ("Aging", 1), 2,
    [[0.95, 0.05],
     [0.05, 0.95]],
    evidence=[("Aging", 0)],
    evidence_card=[2]
)

cpd_short_t = TabularCPD(
    ("ShortCircuit", 1), 2,
    [[0.98, 0.70, 0.40, 0.10],
     [0.02, 0.30, 0.60, 0.90]],
    evidence=[("ShortCircuit", 0), ("Aging", 1)],
    evidence_card=[2, 2]
)

cpd_thermal_t = TabularCPD(
    ("ThermalRunaway", 1), 2,
    [[0.99, 0.75, 0.50, 0.10],
     [0.01, 0.25, 0.50, 0.90]],
    evidence=[("ThermalRunaway", 0), ("Aging", 1)],
    evidence_card=[2, 2]
)

cpd_battery_t = TabularCPD(
    ("BatteryFailure", 1), 2,
    [[0.99, 0.00, 0.30, 0.00, 0.20, 0.00, 0.01, 0.00],
     [0.01, 1.00, 0.70, 1.00, 0.80, 1.00, 0.99, 1.00]],
    evidence=[("ShortCircuit", 1), ("ThermalRunaway", 1), ("BatteryFailure", 0)],
    evidence_card=[2, 2, 2]
)

model.add_cpds(
    cpd_aging_0,
    cpd_short_0,
    cpd_thermal_0,
    cpd_battery_0,
    cpd_aging_t,
    cpd_short_t,
    cpd_thermal_t,
    cpd_battery_t,
)

model.check_model()
# print(model.edges(), '\n')


#get prob of state 1
def get_prob_true(cpd, *parent_states):
    return cpd.values[(1, *parent_states)]

# Sample initial state from t=0 CPDs
def sample_initial_state():
    aging = sample_binary(cpd_aging_0.values[1])
    short = sample_binary(get_prob_true(cpd_short_0, aging))
    thermal = sample_binary(get_prob_true(cpd_thermal_0, aging))
    battery = sample_binary(get_prob_true(cpd_battery_0, short, thermal))

    return {
        "Aging": aging,
        "ShortCircuit": short,
        "ThermalRunaway": thermal,
        "BatteryFailure": battery,
    }


# Sample next state from transition CPDs
def sample_next_state(state):
    aging_t = state["Aging"]
    short_t = state["ShortCircuit"]
    thermal_t = state["ThermalRunaway"]
    battery_t = state["BatteryFailure"]

    aging_next = sample_binary(get_prob_true(cpd_aging_t, aging_t))
    short_next = sample_binary(get_prob_true(cpd_short_t, short_t, aging_next))
    thermal_next = sample_binary(get_prob_true(cpd_thermal_t, thermal_t, aging_next))
    battery_next = sample_binary(get_prob_true(cpd_battery_t, short_next, thermal_next, battery_t))

    return {
        "Aging": aging_next,
        "ShortCircuit": short_next,
        "ThermalRunaway": thermal_next,
        "BatteryFailure": battery_next,
    }

#monte carlo
def monte_carlo(num_runs=5000, num_steps=50):
    battery_hist = np.zeros((num_runs, num_steps + 1))

    for i in range(num_runs):
        state = sample_initial_state()
        battery_hist[i, 0] = state["BatteryFailure"]

        for t in range(1, num_steps + 1):
            state = sample_next_state(state)
            battery_hist[i, t] = state["BatteryFailure"]

    return battery_hist.mean(axis=0)


if __name__ == "__main__":
    failure_prob = monte_carlo()
    print(failure_prob)



#plotting
plt.plot(failure_prob)
plt.xlabel("Time Step")
plt.ylabel("P(Battery Failure)")
plt.title("Battery Failure Probability Over Time")
plt.grid(True)

#save figure
folder_path = "img/"
file_name = "battery_sys_fail.png"
os.makedirs(folder_path, exist_ok=True)
plt.savefig(os.path.join(folder_path, file_name))