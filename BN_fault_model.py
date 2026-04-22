from itertools import product

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


# Binary state convention used throughout:
# 0 = nominal / no failure
# 1 = failed / fault present

ROOT_PRIORS = {
    # Power supply causes
    "LiIonConcentrationDecrease": 0.010,
    "MaterialFlaw": 0.005,
    "Aging": 0.010,
    "Vibration": 0.005,
    "Dendrites": 0.002,
    "ConnectorCorrosion": 0.010,
    "Overcharging": 0.003,
    "ComponentWear": 0.010,
    "ThermalStress": 0.008,
    "RegulatorFailure": 0.002,
    "LoadTransient": 0.010,
    "TransformerSaturation": 0.002,
    "SwitchingFETFailure": 0.005,
    "LoadRejectionDeviceFailure": 0.010,
    "ProtectionDeviceFails": 0.005,
    "SeawaterCorrosion": 0.020,
    "Abrasion": 0.010,
    "MateDemateWear": 0.010,
    "InadequateSealing": 0.015,
    "VibrationFatigue": 0.010,
    "Impact": 0.005,
    # Control and drive component-level faults.
    # The decomposition document lists components but not lower-level causes.
    "ActuatorControllerFailure": 0.006,
    "ESCMotorDriverFailure": 0.008,
    "CurrentSensorFailure": 0.004,
    "SpeedSensorFailure": 0.004,
    "VoltageSensorFailure": 0.004,
    "TemperatureSensorFailure": 0.004,
    "SignalCableFailure": 0.006,
    "SignalConnectorFailure": 0.006,
    # Thruster causes
    "InsulationBreakdown": 0.006,
    "Overload": 0.010,
    "LubricationLoss": 0.006,
    "SandSiltIngress": 0.007,
    "ProlongedHighCurrentDemand": 0.010,
    "PoorThermalPath": 0.006,
    "PressureExceedance": 0.004,
    "ObstacleImpact": 0.006,
    "Cavitation": 0.005,
    "Fatigue": 0.008,
    "MarineDebris": 0.010,
    "Kelp": 0.006,
    "FishingGear": 0.004,
    "MonofilamentLine": 0.004,
    "FastenerFatigue": 0.006,
    "InadequateTorquing": 0.004,
    "AssemblyError": 0.004,
    "Contamination": 0.006,
    "PressureDifferentialExceedance": 0.004,
    "SealDegradation": 0.008,
    "MechanicalImpactFailure": 0.004,
    "ManufacturingDefect": 0.003,
    "FatigueCrack": 0.005,
    "PressureCycling": 0.006,
    "ImproperAssembly": 0.004,
    "ORingAging": 0.008,
    "SurfaceDefect": 0.003,
    "ConnectorPinFailure": 0.004,
    "MechanicalDamage": 0.005,
    "ExcessiveMatingCycles": 0.006,
    "Corrosion": 0.010,
}


GATES = {
    # Power supply subsystem
    "CellCapacityDegradation": ("OR", ["LiIonConcentrationDecrease"]),
    "ShortCircuit": ("OR", ["MaterialFlaw", "Aging", "Vibration", "Dendrites"]),
    "BMSCommsFault": ("OR", ["ConnectorCorrosion"]),
    "ThermalRunaway": ("OR", ["Overcharging", "Aging"]),
    "BatteryFailure": (
        "OR",
        ["CellCapacityDegradation", "ShortCircuit", "BMSCommsFault", "ThermalRunaway"],
    ),
    "VoltageRegulationFailure": ("OR", ["ComponentWear", "ThermalStress"]),
    "OverUnderVoltageOutput": ("OR", ["RegulatorFailure", "LoadTransient"]),
    "PowerSupplyModuleFailure": (
        "OR",
        ["VoltageRegulationFailure", "OverUnderVoltageOutput"],
    ),
    "DCDCConverterFault": ("OR", ["TransformerSaturation", "SwitchingFETFailure"]),
    "OverloadOvercurrentCondition": ("AND", ["ShortCircuit", "ProtectionDeviceFails"]),
    "PowerDistributionConversionFailure": (
        "OR",
        ["DCDCConverterFault", "LoadRejectionDeviceFailure", "OverloadOvercurrentCondition"],
    ),
    "CableInsulationCorrosion": ("OR", ["SeawaterCorrosion", "Abrasion"]),
    "ConnectorCorrosionFailure": ("OR", ["MateDemateWear", "InadequateSealing"]),
    "MechanicalCableBreak": ("OR", ["VibrationFatigue", "Impact"]),
    "CableConnectorFailure": (
        "OR",
        ["CableInsulationCorrosion", "ConnectorCorrosionFailure", "MechanicalCableBreak"],
    ),
    "PowerSupplyFailure": (
        "OR",
        [
            "BatteryFailure",
            "PowerSupplyModuleFailure",
            "PowerDistributionConversionFailure",
            "CableConnectorFailure",
        ],
    ),
    # Control and drive subsystem
    "FeedbackSensorFailure": (
        "OR",
        [
            "CurrentSensorFailure",
            "SpeedSensorFailure",
            "VoltageSensorFailure",
            "TemperatureSensorFailure",
        ],
    ),
    "ControlDriveFailure": (
        "OR",
        [
            "ActuatorControllerFailure",
            "ESCMotorDriverFailure",
            "FeedbackSensorFailure",
            "SignalCableFailure",
            "SignalConnectorFailure",
        ],
    ),
    # Thruster subsystem
    "WindingShortCircuitFailure": (
        "OR",
        ["InsulationBreakdown", "Overload", "WaterIngress"],
    ),
    "BearingSeizureWear": ("OR", ["LubricationLoss", "Overload", "SandSiltIngress"]),
    "MotorOverheating": (
        "OR",
        ["ProlongedHighCurrentDemand", "PoorThermalPath"],
    ),
    "WaterIngress": ("OR", ["SealFailure", "HousingBreach", "PressureExceedance"]),
    "MotorFailure": (
        "OR",
        [
            "WindingShortCircuitFailure",
            "BearingSeizureWear",
            "MotorOverheating",
            "WaterIngress",
        ],
    ),
    "BladeDeformationFracture": ("OR", ["ObstacleImpact", "Cavitation", "Fatigue"]),
    "ForeignObjectEntanglement": (
        "OR",
        ["MarineDebris", "Kelp", "FishingGear", "MonofilamentLine"],
    ),
    "PropLooseningDetachment": (
        "OR",
        ["FastenerFatigue", "Vibration", "InadequateTorquing"],
    ),
    "PropellerFailure": (
        "OR",
        [
            "BladeDeformationFracture",
            "ForeignObjectEntanglement",
            "PropLooseningDetachment",
        ],
    ),
    "ShaftMisalignment": ("OR", ["Impact", "AssemblyError"]),
    "ShaftBearingFailure": ("OR", ["LubricationLoss", "Contamination", "Overload"]),
    "SealFailure": (
        "AND",
        [
            "PressureDifferentialExceedance",
            "SealDegradation",
            "MechanicalImpactFailure",
        ],
    ),
    "ShaftSealFailure": (
        "OR",
        ["ShaftMisalignment", "ShaftBearingFailure", "SealFailure"],
    ),
    "HousingBreach": (
        "OR",
        ["ObstacleImpact", "ManufacturingDefect", "FatigueCrack", "PressureCycling"],
    ),
    "ORingGasketIssues": ("OR", ["ImproperAssembly", "ORingAging", "SurfaceDefect"]),
    "ThrusterInterfaceConnectorFailure": (
        "OR",
        [
            "ConnectorPinFailure",
            "MechanicalDamage",
            "ExcessiveMatingCycles",
            "Corrosion",
        ],
    ),
    "HousingInterfaceFailure": (
        "OR",
        ["HousingBreach", "ORingGasketIssues", "ThrusterInterfaceConnectorFailure"],
    ),
    "ThrusterFailure": (
        "OR",
        ["MotorFailure", "PropellerFailure", "ShaftSealFailure", "HousingInterfaceFailure"],
    ),
    # Top-level propulsion system
    "PropulsionSystemFailure": (
        "OR",
        ["PowerSupplyFailure", "ControlDriveFailure", "ThrusterFailure"],
    ),
}


def root_cpd(variable, p_failure):
    return TabularCPD(variable=variable, variable_card=2, values=[[1 - p_failure], [p_failure]])


def gate_cpd(variable, gate, parents):
    fail_probs = []
    for states in product([0, 1], repeat=len(parents)):
        if gate == "OR":
            fail_probs.append(float(any(states)))
        elif gate == "AND":
            fail_probs.append(float(all(states)))
        else:
            raise ValueError(f"Unsupported gate type for {variable}: {gate}")

    return TabularCPD(
        variable=variable,
        variable_card=2,
        values=[[1 - p for p in fail_probs], fail_probs],
        evidence=parents,
        evidence_card=[2] * len(parents),
    )


def build_model():
    edges = []
    for child, (_, parents) in GATES.items():
        edges.extend((parent, child) for parent in parents)

    model = DiscreteBayesianNetwork(edges)

    cpds = [root_cpd(variable, probability) for variable, probability in ROOT_PRIORS.items()]
    cpds.extend(gate_cpd(child, gate, parents) for child, (gate, parents) in GATES.items())

    model.add_cpds(*cpds)
    model.check_model()
    return model


if __name__ == "__main__":
    model = build_model()
    infer = VariableElimination(model)

    queries = [
        ("PropulsionSystemFailure", {}),
        ("PowerSupplyFailure", {}),
        ("ControlDriveFailure", {}),
        ("ThrusterFailure", {}),
        ("BatteryFailure", {"Aging": 1}),
        ("ThrusterFailure", {"WaterIngress": 1}),
    ]

    for variable, evidence in queries:
        print(f"\nP({variable}" + (f" | {evidence}" if evidence else "") + ")")
        print(infer.query([variable], evidence=evidence))
