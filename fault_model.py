import numpy as np
import matplotlib.pyplot as plt


#fault probabilities
P = {
    "li_ion_degradation": 0.01,
    "material_flaw": 0.005,
    "aging": 0.01,
    "vibration": 0.005,
    "dendrites": 0.002,
    "connector_corrosion": 0.01,
    "overcharging": 0.003,
    "component_wear": 0.01,
    "thermal_stress": 0.008,
    "load_transient": 0.01,
    "regulator_failure": 0.002,
    "transformer_saturation": 0.002,
    "fet_failure": 0.005,
    "short_circuit": 0.01,
    "protection_failure": 0.005,
    "seawater_corrosion": 0.02,
    "abrasion": 0.01,
    "mate_demate": 0.01,
    "inadequate_sealing": 0.015,
    "vibration_fatigue": 0.01,
    "impact": 0.005,
}

#helper function, for random events
def event(p):
    return np.random.rand() < p


#Power subsystem
#Battery Failure
def battery_failure(P):

    '''
    subsystem for battery failure
    any() == OR
    all() == AND
    '''

    # Cell capacity degradation (OR)
    li_ion = event(P["li_ion_degradation"])
    cell_deg = li_ion  # only one listed

    # Short circuit (OR)
    short_circuit = any([
        event(P["material_flaw"]),
        event(P["aging"]),
        event(P["vibration"]),
        event(P["dendrites"])
    ])

    # BMS comms fault
    bms_fault = event(P["connector_corrosion"])

    # Thermal runaway (OR)
    thermal_runaway = any([
        event(P["overcharging"]),
        event(P["aging"])
    ])

    return any([
        cell_deg,
        short_circuit,
        bms_fault,
        thermal_runaway
    ])

#power supply module component
def power_module_failure(P):
    #votlage regulation failure

    voltage_reg = any([
        event(P["component_wear"]),
        event(P["thermal_stress"])
    ])

    over_under_voltage = any([
        event(P["regulator_failure"]),
        event(P["load_transient"])
    ])

    return any([voltage_reg, over_under_voltage])

#power distribution/conversion component
def power_distribution_failure(P):
    dcdc_failure = any([
        event(P["transformer_saturation"]),
        event(P["fet_failure"])
    ])

    load_rejection = event(0.01)  # placeholder

    overload_condition = all([   # AND gate
        event(P["short_circuit"]),
        event(P["protection_failure"])
    ])

    return any([
        dcdc_failure,
        load_rejection,
        overload_condition
    ])

#cable connector failure component
def cable_failure(P):
    insulation = any([
        event(P["seawater_corrosion"]),
        event(P["abrasion"])
    ])

    connector = any([
        event(P["mate_demate"]),
        event(P["inadequate_sealing"])
    ])

    mechanical = any([
        event(P["vibration_fatigue"]),
        event(P["impact"])
    ])

    return any([
        insulation,
        connector,
        mechanical
    ])


#top level system failure
def power_system_failure(P):
    return any([
        battery_failure(P),
        power_module_failure(P),
        power_distribution_failure(P),
        cable_failure(P)
    ])



#monte carlo simulation
N = 100000
failures = sum(power_system_failure(P) for _ in range(N))

#calculate overall probability of failure
print("Power system failure probability:", failures / N)
