import numpy as np

"""
Dictionary of preprocessing operations for conditioning and target variables.
It is generated make_dataset function. Values of dictionary are list objects in which
sepcify preprocessing operation. Every operation has the following template

                       ["string", *pars]

where "string" tells which operation to perform and *pars its parameters. Such operations are

saturation: ["s", [inf, sup]]
gaussian smearing: ["g", sigma, [inf, sup]]
transformation: ["t", func, [a, b]]  # func(a * x + b)

In the case of multiple operations, order follows the operation list indexing.
"""


target_dictionary_muons = {
    "MMuon_cleanmask": [["u", 0.5, None]],
    "MMuon_highPtId": [["u", 0.5, None]],
    "MMuon_highPurity": [["u", 0.5, None]],
    "MMuon_inTimeMuon": [["u", 0.5, None]],
    "MMuon_isStandalone": [["u", 0.5, None]],
    "MMuon_looseId": [["u", 0.5, None]],
    "MMuon_mediumPromptId": [["u", 0.5, None]],
    "MMuon_miniIsoId": [["u", 0.5, None]],
    "MMuon_multiIsoId": [["u", 0.5, None]],
    "MMuon_mvaId": [["u", 0.5, None]],
    "MMuon_mvaLowPtId": [["u", 0.5, None]],
    "MMuon_nStations": [["u", 0.5, None]],
    "MMuon_nTrackerLayers": [["u", 0.5, None]],
    "MMuon_pfIsoId": [["u", 0.5, None]],
    "MMuon_puppiIsoId": [["u", 0.5, None]],
    "MMuon_tightCharge": [["u", 0.5, None]],
    "MMuon_tightId": [["u", 0.5, None]],
    "MMuon_tkIsoId": [["u", 0.5, None]],
    "MMuon_triggerIdLoose": [["u", 0.5, None]],
    "MMuon_etaMinusGen": [
        ["t", np.arctan, [100, 0]],
    ],
    "MMuon_phiMinusGen": [
        ["t", np.arctan, [80, 0]],
    ],
    "MMuon_ptRatio": [
        ["t", np.arctan, [10, -10]],
    ],
    "MMuon_dxy": [["t", np.arctan, [150, 0]]],
    "MMuon_dxyErr": [
        ["s", [-np.inf, 1]],
        ["t", np.log1p, [1, 0]],
    ],
    "MMuon_dxybs": [["t", np.arctan, [50, 0]]],
    "MMuon_dz": [["s", [-np.inf, 20]], ["t", np.arctan, [50, 0]]],
    "MMuon_dzErr": [["s", [-np.inf, 1]], ["t", np.log, [1, 0.001]]],
    "MMuon_ip3d": [["s", [-np.inf, 1]], ["t", np.log, [1, 0.001]]],
    "MMuon_jetPtRelv2": [
        ["s", [-np.inf, 200]],
        ["t", np.log, [1, 0.001]],
        ["gm", -6.9, 1, [-np.inf, -4]],
    ],  #!!!
    "MMuon_jetRelIso": [["s", [-np.inf, 100]], ["t", np.log, [1, 0.08]]],
    "MMuon_pfRelIso04_all": [
        ["s", [-np.inf, 70]],
        ["t", np.log, [1, 0.00001]],
        ["gm", -11.51, 1, [-np.inf, -7.5]],
    ],
    "MMuon_pfRelIso03_all": [
        ["s", [-np.inf, 100]],
        ["t", np.log, [1, 0.00001]],
        ["gm", -11.51, 1, [-np.inf, -7.5]],
    ],
    "MMuon_pfRelIso03_chg": [
        ["s", [-np.inf, 40]],
        ["t", np.log, [1, 0.00001]],
        ["gm", -11.51, 1, [-np.inf, -7.5]],
    ],
    "MMuon_miniPFRelIso_all": [
        ["s", [-np.inf, 100]],
        ["t", np.log, [1, 0.001]],
        ["gm", -8, 0.1, [-np.inf, -6.5]],
    ],
    "MMuon_miniPFRelIso_chg": [
        ["s", [-np.inf, 100]],
        ["t", np.log, [1, 0.001]],
        ["gm", -8, 0.1, [-np.inf, -6.5]],
    ],
    "MMuon_tkRelIso": [
        ["s", [-np.inf, 25]],
        ["t", np.log, [1, 0.001]],
        ["gm", -8, 0.1, [-np.inf, -6.5]],
    ],
    "MMuon_ptErr": [["s", [-np.inf, 300]], ["t", np.log, [1, 0.001]]],
    "MMuon_sip3d": [["s", [-np.inf, 1000]], ["t", np.log, [1, 1]]],
    "MMuon_isGlobal": [["u", 0.5, None]],
    "MMuon_isPFcand": [["u", 0.5, None]],
    "MMuon_isTracker": [["u", 0.5, None]],
    "MMuon_mediumId": [["u", 0.5, None]],
    "MMuon_softId": [["u", 0.5, None]],
    "MMuon_softMvaId": [["u", 0.5, None]],
    "MMuon_charge": [["u", 1, None]],
}
