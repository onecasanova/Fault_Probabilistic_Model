from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import os
import time
from contextlib import contextmanager

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sample_binary(p_true):
    return 1 if np.random.rand() < p_true else 0


IMG_DIR = "img"
NUM_RUNS = 5000
NUM_STEPS = 50
SEED = int(time.time()) % 10000


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
    [[0.98, 0.85, 0.40, 0.10],
     [0.02, 0.15, 0.60, 0.90]],
    evidence=[("ShortCircuit", 0), ("Aging", 1)],
    evidence_card=[2, 2]
)

cpd_thermal_t = TabularCPD(
    ("ThermalRunaway", 1), 2,
    [[0.99, 0.90, 0.50, 0.10],
     [0.01, 0.10, 0.50, 0.90]],
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


def set_prob_true(cpd, parent_states, p_true):
    p_true = float(p_true)
    if not 0 <= p_true <= 1:
        raise ValueError(f"Probability must be in [0, 1], got {p_true}")

    key = (slice(None), *parent_states)
    cpd.values[key] = [1 - p_true, p_true]


@contextmanager
def temporary_cpd_values():
    cpds = [
        cpd_aging_0,
        cpd_short_0,
        cpd_thermal_0,
        cpd_battery_0,
        cpd_aging_t,
        cpd_short_t,
        cpd_thermal_t,
        cpd_battery_t,
    ]
    saved_values = [cpd.values.copy() for cpd in cpds]
    try:
        yield
    finally:
        for cpd, values in zip(cpds, saved_values):
            cpd.values = values


def apply_parameter_change(parameter, value):
    parameter_setters = {
        # P(Aging_{t+1}=1 | Aging_t=0). Lowering this slows aging accumulation.
        "aging_onset": lambda v: set_prob_true(cpd_aging_t, (0,), v),
        # P(BatteryFailure_{t+1}=1 | no short, no thermal, no prior failure).
        "base_battery_hazard": lambda v: set_prob_true(cpd_battery_t, (0, 0, 0), v),
        # Same baseline condition at t=0.
        "initial_base_battery_failure": lambda v: set_prob_true(cpd_battery_0, (0, 0), v),
        # Fault formation under aging, when the fault was absent at the previous step.
        "short_given_aging": lambda v: set_prob_true(cpd_short_t, (0, 1), v),
        "thermal_given_aging": lambda v: set_prob_true(cpd_thermal_t, (0, 1), v),
        # Fault persistence under aging.
        "short_persistence_given_aging": lambda v: set_prob_true(cpd_short_t, (1, 1), v),
        "thermal_persistence_given_aging": lambda v: set_prob_true(cpd_thermal_t, (1, 1), v),
        # Battery failure response when no prior battery failure exists.
        "battery_given_short": lambda v: set_prob_true(cpd_battery_t, (1, 0, 0), v),
        "battery_given_thermal": lambda v: set_prob_true(cpd_battery_t, (0, 1, 0), v),
        "battery_given_short_and_thermal": lambda v: set_prob_true(cpd_battery_t, (1, 1, 0), v),
    }

    if parameter not in parameter_setters:
        valid = ", ".join(sorted(parameter_setters))
        raise ValueError(f"Unknown parameter '{parameter}'. Valid options: {valid}")
    parameter_setters[parameter](value)

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
def monte_carlo(num_runs=NUM_RUNS, num_steps=NUM_STEPS, seed=SEED):
    if seed is not None:
        np.random.seed(seed)

    battery_hist = np.zeros((num_runs, num_steps + 1))

    for i in range(num_runs):
        state = sample_initial_state()
        battery_hist[i, 0] = state["BatteryFailure"]

        for t in range(1, num_steps + 1):
            state = sample_next_state(state)
            battery_hist[i, t] = state["BatteryFailure"]

    return battery_hist.mean(axis=0)


def plot_failure_curve(failure_prob, file_name="battery_sys_fail.png"):
    plt.figure()
    plt.plot(failure_prob)
    plt.xlabel("Time Step")
    plt.ylabel("P(Battery Failure)")
    plt.title("Battery Failure Probability Over Time")
    plt.grid(True)
    os.makedirs(IMG_DIR, exist_ok=True)
    plt.savefig(os.path.join(IMG_DIR, file_name))
    plt.close()


def sensitivity_analysis(parameter, values, num_runs=NUM_RUNS, num_steps=NUM_STEPS, seed=SEED):
    results = {}
    with temporary_cpd_values():
        for value in values:
            apply_parameter_change(parameter, value)
            results[value] = monte_carlo(
                num_runs=num_runs,
                num_steps=num_steps,
                seed=seed,
            )
    return results


def plot_sensitivity(parameter, values, num_runs=NUM_RUNS, num_steps=NUM_STEPS, seed=SEED):
    results = sensitivity_analysis(
        parameter,
        values,
        num_runs=num_runs,
        num_steps=num_steps,
        seed=seed,
    )

    plt.figure()
    for value, failure_prob in results.items():
        plt.plot(failure_prob, label=f"{parameter}={value:g}")
    plt.xlabel("Time Step")
    plt.ylabel("P(Battery Failure)")
    plt.title(f"Sensitivity: {parameter}")
    plt.grid(True)
    plt.legend()
    os.makedirs(IMG_DIR, exist_ok=True)
    file_name = f"sensitivity_{parameter}.png"
    plt.savefig(os.path.join(IMG_DIR, file_name))
    plt.close()
    return results


if __name__ == "__main__":
    print(f"Using random seed: {SEED}")
    failure_prob = monte_carlo()
    print(failure_prob)
    plot_failure_curve(failure_prob)

    plot_sensitivity(
        "base_battery_hazard",
        values=[0.001, 0.003, 0.01, 0.03],
        num_runs=1000,
    )
    plot_sensitivity(
        "aging_onset",
        values=[0.005, 0.01, 0.02, 0.05],
        num_runs=1000,
    )
