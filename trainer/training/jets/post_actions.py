import numpy as np

"""
Dictionary of postprocessing operations for conditioning and target variables.
It is generated make_dataset function. Values of dictionary are list objects in which
sepcify preprocessing operation. Every operation has the following template

                       ["string", *pars]

where "string" tells which operation to perform and *pars its parameters. Such operations are

unsmearing: ["d", [inf, sup]]
transformation: ["i", func, [a, b]]  # func(x) - b / a

In the case of multiple operations, order follows the operation list indexing.
"""

target_dictionary = {
    "MJet_btagCSVV2": [["d", [-np.inf, -0.01], -1]],
    "MJet_btagDeepB": [["d", [-np.inf, -0.01], -1]], 
    "MJet_btagDeepCvB": [["d", [-np.inf, -0.01], -1]], 
    "MJet_btagDeepCvL": [["d", [-np.inf, -0.01], -1]],
    "chEmEF": [["d", [-np.inf, 0], 0]],
    "chHEF": [["i", np.tan, [50, -50]]],
    "cleanmask": [["c", 0.5, [0, 1]]],
    "etaMinusGen": [],
    "hadronFlavour": [["uhf"]], 
    "hfadjacentEtaStripsSize": [["c", 0.5, [0, 1]]],
    "hfcentralEtaStripSize": [["c", 0.5, [0, 1]]],
    "hfsigmaEtaEta": [["d", [-np.inf, 0], -1]],
    "hfsigmaPhiPhi": [["d", [-np.inf, 0], -1]],
    "jetId": [["uj"]],
    "mass": [],
    "muEF": [["d", [-np.inf, 0], 0]],
    "muonSubtrFactor": [["d", [-np.inf, 0], 0]],
    "nConstituents": [["d", None, None]],
    "nElectrons": [["d", None, None]],
    "nMuons": [["d", None, None]],
    "partonFlavour": [["upf"]], 
    "phiMinusGen": [],
    "ptRatio": [],
    "puId": [["upu"]],
}

# overwrite the dict adding Jet_ prefix
target_dictionary_jets = {}
for key, value in target_dictionary.items():
    target_dictionary_jets["Jet_" + key] = value

