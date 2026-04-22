from pgmpy.factors.discrete import TabularCPD

cpd = TabularCPD(
    variable='Grade', 
    variable_card=3, 
    values=[
        [0.85, 0.55, 0.30, 0.05], 
        [0.12, 0.35, 0.40, 0.15],
        [0.03, 0.10, 0.30, 0.80]
        ],
    evidence=['Intelligence', 'Exam Difficulty'],
    evidence_card=[2,2],
    state_names={'Grade': ['A', 'B', 'C'], 'Intelligence': ['High', 'Low'], 'Exam Difficulty': ['Easy', 'Hard']}
)

from pgmpy.factors.discrete import TabularCPD

cpd_grass = TabularCPD(
    variable='Grass', 
    variable_card=2, 
    # Row 1: Probabilities of being DRY for each parent combo
    # Row 2: Probabilities of being WET for each parent combo
    values=[[1.0, 0.1, 0.1, 0.01],  
            [0.0, 0.9, 0.9, 0.99]], 
    evidence=['Rain', 'Sprinkler'],
    evidence_card=[2, 2]
)

# print(cpd)
print(cpd_grass)


