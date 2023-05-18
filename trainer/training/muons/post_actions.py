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
    "highPtId": [["c", 0.5, [0, 1]]],
    "highPurity": [["c", 0.5, [0, 1]]],
    "inTimeMuon": [["c", 0.5, [0, 1]]],
    "isStandalone": [["c", 0.5, [0, 1]]],
    "looseId": [["c", 0.5, [0, 1]]],
    "mediumPromptId": [["c", 0.5, [0, 1]]],
    "miniIsoId": [["c", 0.5, [0, 1]]],
    "multiIsoId": [["c", 0.5, [0, 1]]],
    "mvaId": [["c", 0.5, [0, 1]]],
    "mvaLowPtId": [["c", 0.5, [0, 1]]],
    "nStations": [["c", 0.5, [0, 1]]],
    "nTrackerLayers": [["c", 0.5, [0, 1]]],
    "pfIsoId": [["c", 0.5, [0, 1]]],
    "puppiIsoId": [["c", 0.5, [0, 1]]],
    "tightCharge": [["c", 0.5, [0, 1]]],
    "tightId": [["c", 0.5, [0, 1]]],
    "tkIsoId": [["c", 0.5, [0, 1]]],
    "triggrIdLoose": [["c", 0.5, [0, 1]]],
    "etaMiinusGen": [
        ["i", np.tan, [100, 0]],
        ["a", "MGenMuon_eta"],
        ["rename", "Muon_eta"],
    ],
    "phiMinusGen": [
        ["i", np.tan, [80, 0]],
        ["a", "MGenMuon_phi"],
        ["pmp"],
        ["rename", "Muon_phi"],
    ],
    "ptRatio": [
        ["i", np.tan, [10, -10]],
        ["m", "MGenMuon_pt"],
        ["rename", "Muon_pt"],
    ],
    "dxy": [["i", np.tan, [150, 0]]],
    "dxyErr": [["i", np.expm1, [1, 0]]],
    "dz": [["i", np.tan, [50, 0]]],
    "dzErr": [["i", np.exp, [1, 0.001]]],
    "ip3d": [["i", np.exp, [1, 0.001]]],
    "jetPtRelv2": [["d", [-np.inf, -4], -6.9], ["i", np.exp, [1, 0.001]]],
    "jetRelIso": [["i", np.exp, [1, 0.08]]],
    "pfRelIso04_all": [
        ["i", np.exp, [1, 0.08]],
        ["s"],
        ["a", "Muon_pfRelIso04_all"],
        ["rename", "Muon_pfRelIso04_all"],
    ],
    "pfRelIso03_all": [
        ["i", np.exp, [1, 0.08]],
        ["s"],
    ],
    "pfRelIso03_chg": [
        ["i", np.exp, [1, 0.08]],
        ["s"],
    ],
    "miniPFRelIso_all": [
        ["i", np.exp, [1, 0.08]],
        ["s"],
    ],
    "miniPFRelIso_chg": [
        ["i", np.exp, [1, 0.08]],
        ["s"],
    ],
    "ptErrRel": [["i", np.exp, [1, 0.001]]],
    "tkRelIso": [["i", np.exp, [1, 0.08]]],
    "ptErr": [["i", np.exp, [1, 0.001]]],
    "sip3d": [["i", np.exp, [1, 0.001]]],
    "isGlobal": [["c", 0.5, [0, 1]]],
    "isPFcand": [["c", 0.5, [0, 1]]],
    "isTracker": [["c", 0.5, [0, 1]]],
    "mediumId": [["c", 0.5, [0, 1]]],
    "softId": [["c", 0.5, [0, 1]]],
    "softMvaId": [["c", 0.5, [0, 1]]],
    "charge": [["c", 0.5, [-1, 1]]],
}

target_dictionary_muons = {
    "Muon_cleanmask": [["c", 0.5, [0, 1]]],
    "Muon_highPtId": [["c", 0.5, [0, 1]]],
    "Muon_highPurity": [["c", 0.5, [0, 1]]],
    "Muon_inTimeMuon": [["c", 0.5, [0, 1]]],
    "Muon_isStandalone": [["c", 0.5, [0, 1]]],
    "Muon_looseId": [["c", 0.5, [0, 1]]],
    "Muon_mediumPromptId": [["c", 0.5, [0, 1]]],
    "Muon_miniIsoId": [["c", 0.5, [0, 1]]],
    "Muon_multiIsoId": [["c", 0.5, [0, 1]]],
    "Muon_mvaId": [["c", 0.5, [0, 1]]],
    "Muon_"
    "Muon_etaMinusGen": [
        ["i", np.tan, [100, 0]],
        ["s"],
        ["a", "MGenMuon_eta"],
        ["rename", "Muon_eta"],
    ],
    "Muon_phiMinusGen": [
        ["i", np.tan, [80, 0]],
        ["s"],
        ["a", "MGenMuon_phi"],
        ["pmp"],
        ["rename", "Muon_phi"],
    ],
    "Muon_ptRatio": [
        ["i", np.tan, [10, -10]],
        ["s"],
        ["m", "MGenMuon_pt"],
        ["rename", "Muon_pt"],
    ],
    "Muon_dxy": [["i", np.tan, [150, 0]], ["s"]],
    "Muon_dxyErr": [["i", np.expm1, [1, 0]], ["s"]],
    "Muon_dz": [["i", np.tan, [50, 0]], ["s"]],
    "Muon_dzErr": [["i", np.exp, [1, 0.001]]],
    "Muon_ip3d": [["i", np.exp, [1, 0.001]]],
    "Muon_jetPtRelv2": [["d", [-np.inf, -4], -6.9], ["i", np.exp, [1, 0.001]]],
    "Muon_jetRelIso": [["i", np.exp, [1, 0.08]]],
    "Muon_pfRelIso04_all": [
        ["d", [-np.inf, -7.5], -11.51],
        ["i", np.exp, [1, 0.00001]],
    ],
    "Muon_pfRelIso03_all": [
        ["d", [-np.inf, -7.5], -11.51],
        ["i", np.exp, [1, 0.00001]],
    ],
    "Muon_pfRelIso03_chg": [
        ["d", [-np.inf, -7.5], -11.51],
        ["i", np.exp, [1, 0.00001]],
    ],
    "Muon_ptErr": [["i", np.exp, [1, 0.001]]],
    "Muon_sip3d": [["i", np.exp, [1, 1]]],
    "Muon_isGlobal": [["c", 0.5, [0, 1]]],
    "Muon_isPFcand": [["c", 0.5, [0, 1]]],
    "Muon_isTracker": [["c", 0.5, [0, 1]]],
    "Muon_mediumId": [["c", 0.5, [0, 1]]],
    "Muon_softId": [["c", 0.5, [0, 1]]],
    "Muon_softMvaId": [["c", 0.5, [0, 1]]],
    "Muon_charge": [["genow", "MGenMuon_charge"]],
}
