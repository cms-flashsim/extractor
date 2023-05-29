import numpy as np

"""
Dictionary of postprocessing operations for conditioning and target variables.
It is generated make_dataset function. Values of dictionary are list objects in which
sepcify preprocessing operation. Every operation has the following template

                       ["string", *pars]

where "string" tells which operation to perform and *pars its parameters. Such operations are

unsmearing: ["d", [inf, sup]]
transformation: ["i", func, [a, b]]  # (func(x) - b) / a

In the case of multiple operations, order follows the operation list indexing.
"""

target_dictionary_muons = {
    "cleanmask": [["c", 0.5, [0, 1]]],
    "highPtId": [["d", None, None]],
    "highPurity": [["c", 0.5, [0, 1]]],
    "inTimeMuon": [["c", 0.5, [0, 1]]],
    "isStandalone": [["c", 0.5, [0, 1]]],
    "looseId": [["c", 0.5, [0, 1]]],
    "mediumPromptId": [["c", 0.5, [0, 1]]],
    "miniIsoId": [["d", None, None]],
    "multiIsoId": [["d", None, None]],
    "mvaId": [["d", None, None]],
    "mvaLowPtId": [["d", None, None]],
    "nStations": [["d", None, None]],
    "nTrackerLayers": [["d", None, None]],
    "pfIsoId": [["d", None, None]],
    "puppiIsoId": [["d", None, None]],
    "tightCharge": [["c", 1, [0, 2]]],
    "tightId": [["c", 0.5, [0, 1]]],
    "tkIsoId": [["c", 1.5, [1, 2]]],
    "triggrIdLoose": [["c", 0.5, [0, 1]]],
    "etaMinusGen": [
        ["i", np.tan, [100, 0]],
    ],
    "phiMinusGen": [
        ["i", np.tan, [80, 0]],
        ["pmp"],
    ],
    "ptRatio": [
        ["i", np.tan, [10, -10]],
    ],
    "dxy": [["i", np.tan, [150, 0]]],
    "dxyErr": [["i", np.expm1, [1, 0]]],
    "dxybs": [["i", np.tan, [50, 0]]],
    "dz": [["i", np.tan, [50, 0]]],
    "dzErr": [["i", np.exp, [1, 0.001]]],
    "ip3d": [["i", np.exp, [1, 0.001]]],
    "jetPtRelv2": [["d", [-np.inf, -4], np.log(0.001)], ["i", np.exp, [1, 0.001]]],
    "jetRelIso": [["i", np.exp, [1, 0.08]]],
    "pfRelIso04_all": [
        ["d", [-np.inf, -7.5], np.log(0.00001)],
        ["i", np.exp, [1, 0.00001]],
    ],
    "pfRelIso03_all": [
        ["d", [-np.inf, -7.5], np.log(0.00001)],
        ["i", np.exp, [1, 0.00001]],
    ],
    "pfRelIso03_chg": [
        ["d", [-np.inf, -7.5], np.log(0.00001)],
        ["i", np.exp, [1, 0.00001]],
    ],
    "miniPFRelIso_all": [
        ["d", [-np.inf, -6.5], np.log(0.001)],
        ["i", np.exp, [1, 0.001]],
    ],
    "miniPFRelIso_chg": [
        ["d", [-np.inf, -6.5], np.log(0.001)],
        ["i", np.exp, [1, 0.001]],
    ],
    "tkRelIso": [
        ["d", [-np.inf, -6.5], np.log(0.001)], 
        ["i", np.exp, [1, 0.001]]
    ],
    "ptErr": [["i", np.exp, [1, 0.001]]],
    "sip3d": [["i", np.exp, [1, 1]]],
    "isGlobal": [["c", 0.5, [0, 1]]],
    "isPFcand": [["c", 0.5, [0, 1]]],
    "isTracker": [["c", 0.5, [0, 1]]],
    "mediumId": [["c", 0.5, [0, 1]]],
    "softId": [["c", 0.5, [0, 1]]],
    "softMvaId": [["c", 0.5, [0, 1]]],
    "charge": [["c", 0.0, [-1, 1]]],
}

target_dictionary_jets = {}
for key, value in target_dictionary_muons.items():
    target_dictionary_jets["Muon_" + key] = value
