# Fault Probabilistic Model - Current State

Last reviewed: 2026-04-22

## Repository Purpose

This repository is exploring probabilistic fault modeling for marine robot propulsion-system failures. It currently contains three modeling inputs/approaches:

- A Monte Carlo fault-tree style simulator in `fault_model.py`.
- A full propulsion-system decomposition in `prop_system_decomp.md` and `prop_system_decomp.txt`.
- A `pgmpy` Bayesian network model in `BN_fault_model.py`.
- A `pgmpy` dynamic Bayesian network model in `DBN_fault_model.py`.

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
  - Implements the full static propulsion-system Bayesian network with `pgmpy`.
  - Encodes the decomposition using binary variables where `0 = nominal` and `1 = failed/fault present`.
  - Uses root prior CPDs for base causes and deterministic OR/AND gate CPDs for intermediate/top-level failures.
  - Covers:
    - power supply failure,
    - control and drive failure,
    - thruster failure,
    - top-level propulsion system failure.
  - Control/drive component failure nodes use component-level priors because the decomposition lists components but not lower-level causes.
  - Verified top-level output:
    - `PropulsionSystemFailure(0) = 0.6897`
    - `PropulsionSystemFailure(1) = 0.3103`

- `DBN_fault_model.py`
  - Implements a dynamic Bayesian network version of the same propulsion decomposition.
  - Adds intra-slice decomposition edges and temporal persistence edges from time slice `0` to time slice `1`.
  - Uses root persistence of `0.95` and fault-gate persistence of `0.90`.
  - The command-line entry point validates and summarizes the DBN instead of running full exact inference, because full exact inference over the complete temporal graph is slow.
  - Verified summary:
    - `Variables per time slice: 97`
    - `Root cause priors: 60`
    - `Fault gate nodes: 37`
    - `CPDs: 194`

- `prop_system_decomp.md` / `prop_system_decomp.txt`
  - Source decomposition for the propulsion system.
  - Defines power supply, control/drive, and thruster subsystems.
  - Provides detailed failure decomposition for power supply and thruster.
  - Provides component decomposition for control/drive.

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
- `DBN_fault_model.py` is modified into a real DBN implementation.
- `BN_example.py` is untracked.
- `BN_fault_model.py` is untracked and now contains the full propulsion static BN.
- `REPO_STATE.md` was added to record this summary.
- `prop_system_decomp.md` and `prop_system_decomp.txt` are untracked decomposition documents.

Recent commit history:

- `f0cd681` - `finish inference for BN fault tolerance, will add other subsystems soon. rename files appropriately`
- `df680d6` - `first commit`

## Verified Commands

Run with the local virtual environment:

```bash
./auv/bin/python BN_fault_model.py
./auv/bin/python DBN_fault_model.py
```

Both commands complete successfully. `BN_fault_model.py` prints selected exact-inference queries. `DBN_fault_model.py` builds, validates, and prints a summary of the complete temporal graph.

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
- A working static Bayesian network exists for the full propulsion decomposition.
- A working dynamic Bayesian network exists for the full propulsion decomposition with one-step temporal persistence.
- The static Bayesian network performs selected exact inference queries.
- The DBN validates structurally and avoids launching expensive full-graph exact inference by default.

## Suggested Next Steps

1. Decide whether `BN_example.py` should replace the deleted tracked `BN.py`, then stage the intended rename/delete state.
2. Review and tune the root priors in `BN_fault_model.py`; many are engineering placeholders.
3. Review whether deterministic OR/AND gate CPDs are the desired modeling choice or whether noisy gates should be used.
4. Remove the unused `matplotlib` import from `fault_model.py` or install `matplotlib` if plotting will be added soon.
5. Add a `requirements.txt` or `pyproject.toml` so the environment can be recreated without relying on the local `auv` directory.
