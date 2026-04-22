# Fault Probabilistic Model - Current State

Last reviewed: 2026-04-22

## Repository Purpose

This repository is exploring probabilistic fault modeling for marine robot power-system failures. It currently contains two modeling approaches:

- A Monte Carlo fault-tree style simulator in `fault_model.py`.
- A `pgmpy` Bayesian network prototype for battery failure in `BN_fault_model.py`.

There is also an early dynamic Bayesian network placeholder in `DBN_fault_model.py`, but it currently duplicates the static Bayesian network behavior.

## Current Files

- `fault_model.py`
  - Defines base probabilities for low-level power subsystem faults.
  - Models power system failure with OR/AND logic across:
    - battery failure,
    - power module failure,
    - power distribution/conversion failure,
    - cable/connector failure.
  - Runs a Monte Carlo simulation with `N = 100000`.
  - Current runtime issue: imports `matplotlib.pyplot`, but the local `auv` virtual environment does not have `matplotlib` installed. The import appears unused.

- `BN_fault_model.py`
  - Implements a static Bayesian network with `pgmpy`.
  - Graph structure:
    - `Aging -> ShortCircuit`
    - `Aging -> ThermalRunaway`
    - `ShortCircuit -> BatteryFailure`
    - `ThermalRunaway -> BatteryFailure`
  - Adds CPDs for `Aging`, `ShortCircuit`, `ThermalRunaway`, and `BatteryFailure`.
  - Runs variable elimination for `P(BatteryFailure | Aging = 1)`.
  - Verified output:
    - `BatteryFailure(0) = 0.5370`
    - `BatteryFailure(1) = 0.4630`

- `DBN_fault_model.py`
  - Currently identical in behavior to `BN_fault_model.py`.
  - Adds `from pgmpy.models import DynamicBayesianNetwork as DBN`, but no dynamic network structure or temporal CPDs are implemented yet.
  - Verified output matches `BN_fault_model.py`.

- `BN_example.py`
  - Untracked example file showing basic `TabularCPD` construction.
  - Appears to be the replacement or renamed form of the deleted tracked `BN.py`.

- `auv/`
  - Local Python virtual environment.
  - Contains `pgmpy` and `numpy`.
  - Does not currently contain `matplotlib`.

## Git Working Tree State

As of this review:

- `BN.py` is deleted from the working tree, but still tracked in `HEAD`.
- `DBN_fault_model.py` is modified only by adding the `DynamicBayesianNetwork` import.
- `BN_example.py` is untracked.
- `BN_fault_model.py` is untracked.
- `REPO_STATE.md` was added to record this summary.

Recent commit history:

- `f0cd681` - `finish inference for BN fault tolerance, will add other subsystems soon. rename files appropriately`
- `df680d6` - `first commit`

## Verified Commands

Run with the local virtual environment:

```bash
./auv/bin/python BN_fault_model.py
./auv/bin/python DBN_fault_model.py
```

Both commands complete successfully and print the same posterior distribution for `BatteryFailure`.

This command currently fails because `matplotlib` is missing:

```bash
./auv/bin/python fault_model.py
```

Observed error:

```text
ModuleNotFoundError: No module named 'matplotlib'
```

## Progress So Far

- A Monte Carlo fault tree exists for a broader power-system failure model.
- A working static Bayesian network exists for the battery failure subset.
- The Bayesian network performs inference conditioned on aging.
- Initial groundwork for a DBN file exists, but temporal modeling has not started.

## Suggested Next Steps

1. Decide whether `BN_example.py` should replace the deleted tracked `BN.py`, then stage the intended rename/delete state.
2. Remove the unused `matplotlib` import from `fault_model.py` or install `matplotlib` if plotting will be added soon.
3. Convert `DBN_fault_model.py` from a static duplicate into a real dynamic Bayesian network with time-indexed variables.
4. Expand the Bayesian network beyond battery failure to include the same power subsystems represented in `fault_model.py`.
5. Add a `requirements.txt` or `pyproject.toml` so the environment can be recreated without relying on the local `auv` directory.
