from itertools import product

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DynamicBayesianNetwork as DBN

from BN_fault_model import GATES, ROOT_PRIORS


# Binary state convention used throughout:
# 0 = nominal / no failure
# 1 = failed / fault present

ROOT_PERSISTENCE = 0.95
GATE_PERSISTENCE = 0.900


def node(variable, time_slice):
    return (variable, time_slice)


def root_cpd(variable, p_failure, time_slice=0):
    return TabularCPD(
        variable=node(variable, time_slice),
        variable_card=2,
        values=[[1 - p_failure], [p_failure]],
    )


def root_transition_cpd(variable, p_failure):
    # Columns are previous state 0, then previous state 1.
    return TabularCPD(
        variable=node(variable, 1),
        variable_card=2,
        values=[
            [1 - p_failure, 1 - ROOT_PERSISTENCE],
            [p_failure, ROOT_PERSISTENCE],
        ],
        evidence=[node(variable, 0)],
        evidence_card=[2],
    )


def gate_initial_cpd(variable, gate, parents):
    fail_probs = []
    for states in product([0, 1], repeat=len(parents)):
        fail_probs.append(gate_failure_probability(gate, states, previous_failed=None))

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
        fail_probs.append(gate_failure_probability(gate, parent_states, previous_failed))

    evidence = [node(parent, 1) for parent in parents] + [node(variable, 0)]
    return TabularCPD(
        variable=node(variable, 1),
        variable_card=2,
        values=[[1 - p for p in fail_probs], fail_probs],
        evidence=evidence,
        evidence_card=[2] * len(evidence),
    )


def gate_failure_probability(gate, parent_states, previous_failed):
    if gate == "OR":
        caused_now = any(parent_states)
    elif gate == "AND":
        caused_now = all(parent_states)
    else:
        raise ValueError(f"Unsupported gate type: {gate}")

    if caused_now:
        return 1.0
    if previous_failed:
        return GATE_PERSISTENCE
    return 0.0


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


if __name__ == "__main__":
    model = build_model()

    print("Dynamic Bayesian network built and validated.")
    print(f"Variables per time slice: {len(set(ROOT_PRIORS) | set(GATES))}")
    print(f"Root cause priors: {len(ROOT_PRIORS)}")
    print(f"Fault gate nodes: {len(GATES)}")
    print(f"CPDs: {len(model.get_cpds())}")
    print(f"Temporal persistence: roots={ROOT_PERSISTENCE}, gates={GATE_PERSISTENCE}")
