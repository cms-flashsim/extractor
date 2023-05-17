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
    "Muon_etaMinusGen": [
        ["t", np.arctan, [100, 0]],
    ],
    "Muon_phiMinusGen": [
        ["t", np.arctan, [80, 0]],
    ],
    "Muon_ptRatio": [
        ["t", np.arctan, [10, -10]],
    ],
    "Muon_dxy": [["t", np.arctan, [150, 0]]],
    "Muon_dxyErr": [["s", [-np.inf, 1]], ["t", np.log1p, [1, 0]],],
    "Muon_dz": [["s", [-np.inf, 20]], ["t", np.arctan, [50, 0]]],
    "Muon_dzErr": [["s", [-np.inf, 1]], ["t", np.log, [1, 0.001]]],
    "Muon_ip3d": [["s", [-np.inf, 1]], ["t", np.log, [1, 0.001]]],
    "Muon_jetPtRelv2": [["s", [-np.inf, 200]], ["t", np.log, [1, 0.001]], ["gm", -6.9, 1, [-np.inf, -4]], ], #!!!
    "Muon_jetRelIso": [["s", [-np.inf, 100]], ["t", np.log, [1, 0.08]]],
    "Muon_pfRelIso04_all": [
        ["s", [-np.inf, 70]],
        ["t", np.log, [1, 0.00001]],
        ["gm", -11.51, 1, [-np.inf, -7.5]],
    ],
    "Muon_pfRelIso03_all": [
        ["s", [-np.inf, 100]],
        ["t", np.log, [1, 0.00001]],
        ["gm",-11.51, 1, [-np.inf, -7.5]], 
        
    ],
    "Muon_pfRelIso03_chg": [
        ["s", [-np.inf, 40]],
        ["t", np.log, [1, 0.00001]],
        ["gm", -11.51, 1, [-np.inf, -7.5]],
        
    ],
    "Muon_ptErr": [["s", [-np.inf, 300]], ["t", np.log, [1, 0.001]]],
    "Muon_sip3d": [["s", [-np.inf, 1000]], ["t", np.log, [1, 1]]],
    "Muon_isGlobal": [["u", 0.5, None]],
    "Muon_isPFcand": [["u", 0.5, None]],
    "Muon_isTracker": [["u", 0.5, None]],
    "Muon_mediumId": [["u", 0.5, None]],
    "Muon_softId": [["u", 0.5, None]],
    "Muon_softMvaId": [["u", 0.5, None]],
    "Muon_charge": [["genow", "MGenMuon_charge"]],
}