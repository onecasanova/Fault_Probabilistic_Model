from itertools import product
import os
import time
from graphlib import TopologicalSorter
from contextlib import contextmanager

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DynamicBayesianNetwork as DBN

from BN_fault_model import GATES, ROOT_PRIORS


# Binary state convention used throughout:
# 0 = nominal / no failure
# 1 = failed / fault present

OR_PARENT_TRIGGER_PROB = 0.55
AND_GATE_TRIGGER_PROB = 0.75

IMG_DIR = "img"
NUM_RUNS = 5000
NUM_STEPS = 50
SEED = int(time.time()) % 10000

TRACKED_SIGNALS = [
    "PowerSupplyFailure",
    "ControlDriveFailure",
    "ThrusterFailure",
    "PropulsionSystemFailure",
]

ROOT_PERSISTENCE = {
    "Aging": 0.98,
    "MaterialFlaw": 0.95,
    "Dendrites": 0.90,
    "ConnectorCorrosion": 0.90,
    "ComponentWear": 0.90,
    "SeawaterCorrosion": 0.90,
    "Abrasion": 0.80,
    "MateDemateWear": 0.85,
    "InadequateSealing": 0.95,
    "VibrationFatigue": 0.90,
    "InsulationBreakdown": 0.90,
    "LubricationLoss": 0.85,
    "SandSiltIngress": 0.70,
    "PoorThermalPath": 0.90,
    "Fatigue": 0.90,
    "FastenerFatigue": 0.85,
    "AssemblyError": 0.95,
    "Contamination": 0.75,
    "SealDegradation": 0.90,
    "ManufacturingDefect": 0.95,
    "FatigueCrack": 0.90,
    "ImproperAssembly": 0.95,
    "ORingAging": 0.90,
    "SurfaceDefect": 0.95,
    "ConnectorPinFailure": 0.90,
    "ExcessiveMatingCycles": 0.85,
    "Corrosion": 0.90,
}

GATE_PERSISTENCE = {
    "CellCapacityDegradation": 0.95,
    "ShortCircuit": 0.85,
    "BMSCommsFault": 0.75,
    "ThermalRunaway": 0.98,
    "BatteryFailure": 0.99,
    "VoltageRegulationFailure": 0.85,
    "PowerSupplyModuleFailure": 0.90,
    "DCDCConverterFault": 0.90,
    "CableInsulationCorrosion": 0.90,
    "ConnectorCorrosionFailure": 0.85,
    "MechanicalCableBreak": 0.98,
    "CableConnectorFailure": 0.90,
    "PowerSupplyFailure": 0.95,
    "FeedbackSensorFailure": 0.90,
    "ControlDriveFailure": 0.90,
    "WindingShortCircuitFailure": 0.95,
    "BearingSeizureWear": 0.90,
    "MotorOverheating": 0.30,
    "WaterIngress": 0.90,
    "MotorFailure": 0.95,
    "BladeDeformationFracture": 0.95,
    "ForeignObjectEntanglement": 0.70,
    "PropLooseningDetachment": 0.90,
    "PropellerFailure": 0.95,
    "ShaftMisalignment": 0.90,
    "ShaftBearingFailure": 0.90,
    "SealFailure": 0.90,
    "ShaftSealFailure": 0.95,
    "HousingBreach": 0.98,
    "ORingGasketIssues": 0.85,
    "ThrusterInterfaceConnectorFailure": 0.90,
    "HousingInterfaceFailure": 0.95,
    "ThrusterFailure": 0.97,
    "PropulsionSystemFailure": 0.98,
}

GATE_ORDER = tuple(
    variable
    for variable in TopologicalSorter(
        {child: tuple(parents) for child, (_, parents) in GATES.items()}
    ).static_order()
    if variable in GATES
)


def node(variable, time_slice):
    return (variable, time_slice)


def sample_binary(p_true):
    return 1 if np.random.rand() < p_true else 0


def set_probability(mapping, key, value):
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError(f"Probability must be in [0, 1], got {value}")
    mapping[key] = value


@contextmanager
def temporary_parameters():
    global OR_PARENT_TRIGGER_PROB, AND_GATE_TRIGGER_PROB

    saved_or_parent_trigger_prob = OR_PARENT_TRIGGER_PROB
    saved_and_gate_trigger_prob = AND_GATE_TRIGGER_PROB
    saved_root_persistence = ROOT_PERSISTENCE.copy()
    saved_gate_persistence = GATE_PERSISTENCE.copy()

    try:
        yield
    finally:
        OR_PARENT_TRIGGER_PROB = saved_or_parent_trigger_prob
        AND_GATE_TRIGGER_PROB = saved_and_gate_trigger_prob
        ROOT_PERSISTENCE.clear()
        ROOT_PERSISTENCE.update(saved_root_persistence)
        GATE_PERSISTENCE.clear()
        GATE_PERSISTENCE.update(saved_gate_persistence)


def apply_parameter_change(parameter, value):
    global OR_PARENT_TRIGGER_PROB, AND_GATE_TRIGGER_PROB

    if parameter == "or_parent_trigger_prob":
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {value}")
        OR_PARENT_TRIGGER_PROB = value
        return

    if parameter == "and_gate_trigger_prob":
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {value}")
        AND_GATE_TRIGGER_PROB = value
        return

    parameter_setters = {
        "aging_persistence": lambda v: set_probability(ROOT_PERSISTENCE, "Aging", v),
        "connector_corrosion_persistence": lambda v: set_probability(
            ROOT_PERSISTENCE, "ConnectorCorrosion", v
        ),
        "thermal_runaway_persistence": lambda v: set_probability(
            GATE_PERSISTENCE, "ThermalRunaway", v
        ),
        "battery_failure_persistence": lambda v: set_probability(
            GATE_PERSISTENCE, "BatteryFailure", v
        ),
        "power_supply_failure_persistence": lambda v: set_probability(
            GATE_PERSISTENCE, "PowerSupplyFailure", v
        ),
        "motor_failure_persistence": lambda v: set_probability(
            GATE_PERSISTENCE, "MotorFailure", v
        ),
        "thruster_failure_persistence": lambda v: set_probability(
            GATE_PERSISTENCE, "ThrusterFailure", v
        ),
        "propulsion_failure_persistence": lambda v: set_probability(
            GATE_PERSISTENCE, "PropulsionSystemFailure", v
        ),
        "control_drive_failure_persistence": lambda v: set_probability(
            GATE_PERSISTENCE, "ControlDriveFailure", v
        ),
    }

    if parameter not in parameter_setters:
        valid = ", ".join(
            [
                "and_gate_trigger_prob",
                "or_parent_trigger_prob",
                *sorted(parameter_setters.keys()),
            ]
        )
        raise ValueError(f"Unknown parameter '{parameter}'. Valid options: {valid}")

    parameter_setters[parameter](value)


def root_cpd(variable, p_failure, time_slice=0):
    return TabularCPD(
        variable=node(variable, time_slice),
        variable_card=2,
        values=[[1 - p_failure], [p_failure]],
    )


def root_transition_cpd(variable, p_failure):
    persistence = ROOT_PERSISTENCE.get(variable, p_failure)
    # Columns are previous state 0, then previous state 1.
    return TabularCPD(
        variable=node(variable, 1),
        variable_card=2,
        values=[
            [1 - p_failure, 1 - persistence],
            [p_failure, persistence],
        ],
        evidence=[node(variable, 0)],
        evidence_card=[2],
    )


def gate_initial_cpd(variable, gate, parents):
    fail_probs = []
    for states in product([0, 1], repeat=len(parents)):
        fail_probs.append(gate_failure_probability(variable, gate, states, previous_failed=None))

    return TabularCPD(
        variable=node(variable, 0),
        variable_card=2,
        values=[[1 - p for p in fail_probs], fail_probs],
        evidence=[node(parent, 0) for parent in parents],
        evidence_card=[2] * len(parents),
    )


def gate_transition_cpd(variable, gate, parents):
    fail_probs = []
    for states in product([0, 1], repeat=len(parents) + 1):
        parent_states = states[:-1]
        previous_failed = states[-1]
        fail_probs.append(gate_failure_probability(variable, gate, parent_states, previous_failed))

    evidence = [node(parent, 1) for parent in parents] + [node(variable, 0)]
    return TabularCPD(
        variable=node(variable, 1),
        variable_card=2,
        values=[[1 - p for p in fail_probs], fail_probs],
        evidence=evidence,
        evidence_card=[2] * len(evidence),
    )


def gate_trigger_probability(gate, parent_states):
    if gate == "OR":
        active_count = sum(parent_states)
        if active_count == 0:
            return 0.0
        return 1 - (1 - OR_PARENT_TRIGGER_PROB) ** active_count
    elif gate == "AND":
        return AND_GATE_TRIGGER_PROB if all(parent_states) else 0.0
    else:
        raise ValueError(f"Unsupported gate type: {gate}")


def gate_failure_probability(variable, gate, parent_states, previous_failed):
    trigger_probability = gate_trigger_probability(gate, parent_states)
    persistence_probability = GATE_PERSISTENCE.get(variable, 0.0) if previous_failed else 0.0
    return 1 - (1 - trigger_probability) * (1 - persistence_probability)


def build_model():
    variables = sorted(set(ROOT_PRIORS) | set(GATES))
    model = DBN()
    model.add_nodes_from(variables)

    intra_edges = []
    temporal_edges = []
    for child, (_, parents) in GATES.items():
        intra_edges.extend((node(parent, 0), node(child, 0)) for parent in parents)
    temporal_edges.extend((node(variable, 0), node(variable, 1)) for variable in variables)

    model.add_edges_from(intra_edges + temporal_edges)

    cpds = [root_cpd(variable, probability) for variable, probability in ROOT_PRIORS.items()]
    cpds.extend(
        gate_initial_cpd(child, gate, parents) for child, (gate, parents) in GATES.items()
    )
    cpds.extend(
        root_transition_cpd(variable, probability)
        for variable, probability in ROOT_PRIORS.items()
    )
    cpds.extend(
        gate_transition_cpd(child, gate, parents) for child, (gate, parents) in GATES.items()
    )

    model.add_cpds(*cpds)
    model.initialize_initial_state()
    model.check_model()
    return model


def sample_initial_state():
    state = {}

    for variable, probability in ROOT_PRIORS.items():
        state[variable] = sample_binary(probability)

    for variable in GATE_ORDER:
        gate, parents = GATES[variable]
        parent_states = [state[parent] for parent in parents]
        p_failure = gate_failure_probability(variable, gate, parent_states, previous_failed=None)
        state[variable] = sample_binary(p_failure)

    return state


def sample_next_state(previous_state):
    state = {}

    for variable, probability in ROOT_PRIORS.items():
        if previous_state[variable]:
            p_failure = ROOT_PERSISTENCE.get(variable, probability)
        else:
            p_failure = probability
        state[variable] = sample_binary(p_failure)

    for variable in GATE_ORDER:
        gate, parents = GATES[variable]
        parent_states = [state[parent] for parent in parents]
        p_failure = gate_failure_probability(variable, gate, parent_states, previous_state[variable])
        state[variable] = sample_binary(p_failure)

    return state


def monte_carlo(num_runs=NUM_RUNS, num_steps=NUM_STEPS, seed=SEED):
    if seed is not None:
        np.random.seed(seed)

    histories = {
        signal: np.zeros((num_runs, num_steps + 1))
        for signal in TRACKED_SIGNALS
    }

    for run_idx in range(num_runs):
        state = sample_initial_state()
        for signal in TRACKED_SIGNALS:
            histories[signal][run_idx, 0] = state[signal]

        for step in range(1, num_steps + 1):
            state = sample_next_state(state)
            for signal in TRACKED_SIGNALS:
                histories[signal][run_idx, step] = state[signal]

    return {signal: history.mean(axis=0) for signal, history in histories.items()}


def plot_failure_curves(results, file_name="propulsion_system_fail.png"):
    plt.figure()
    for signal, failure_prob in results.items():
        plt.plot(failure_prob, label=signal)

    plt.xlabel("Time Step")
    plt.ylabel("P(Failure)")
    plt.title("Propulsion System Failure Probability Over Time")
    plt.grid(True)
    plt.legend()
    os.makedirs(IMG_DIR, exist_ok=True)
    plt.savefig(os.path.join(IMG_DIR, file_name))
    plt.close()


def sensitivity_analysis(parameter, values, num_runs=NUM_RUNS, num_steps=NUM_STEPS, seed=SEED):
    results = {}
    with temporary_parameters():
        for value in values:
            apply_parameter_change(parameter, value)
            results[value] = monte_carlo(
                num_runs=num_runs,
                num_steps=num_steps,
                seed=seed,
            )
    return results


def plot_sensitivity(
    parameter,
    values,
    tracked_signal="PropulsionSystemFailure",
    num_runs=NUM_RUNS,
    num_steps=NUM_STEPS,
    seed=SEED,
):
    results = sensitivity_analysis(
        parameter,
        values,
        num_runs=num_runs,
        num_steps=num_steps,
        seed=seed,
    )

    plt.figure()
    for value, signal_results in results.items():
        plt.plot(signal_results[tracked_signal], label=f"{parameter}={value:g}")

    plt.xlabel("Time Step")
    plt.ylabel(f"P({tracked_signal})")
    plt.title(f"Sensitivity: {tracked_signal}")
    plt.grid(True)
    plt.legend()
    os.makedirs(IMG_DIR, exist_ok=True)
    file_name = f"sensitivity_{tracked_signal}_{parameter}.png"
    plt.savefig(os.path.join(IMG_DIR, file_name))
    plt.close()
    return results


if __name__ == "__main__":
    model = build_model()
    results = monte_carlo()

    print(f"Using random seed: {SEED}")
    print("Dynamic Bayesian network built and validated.")
    print(f"Variables per time slice: {len(set(ROOT_PRIORS) | set(GATES))}")
    print(f"Root cause priors: {len(ROOT_PRIORS)}")
    print(f"Fault gate nodes: {len(GATES)}")
    print(f"CPDs: {len(model.get_cpds())}")
    print(
        "Persistence settings: "
        f"{len(ROOT_PERSISTENCE)} persistent root causes, "
        f"{len(GATE_PERSISTENCE)} persistent derived faults"
    )
    for signal in TRACKED_SIGNALS:
        print(f"{signal} final probability: {results[signal][-1]:.4f}")

    plot_failure_curves(results)

    plot_sensitivity(
        "or_parent_trigger_prob",
        values=[0.35, 0.45, 0.55, 0.65],
        tracked_signal="PropulsionSystemFailure",
        num_runs=1000,
    )
    plot_sensitivity(
        "thruster_failure_persistence",
        values=[0.75, 0.85, 0.95, 0.99],
        tracked_signal="ThrusterFailure",
        num_runs=1000,
    )
    plot_sensitivity(
        "battery_failure_persistence",
        values=[0.85, 0.92, 0.97, 0.99],
        tracked_signal="PowerSupplyFailure",
        num_runs=1000,
    )
