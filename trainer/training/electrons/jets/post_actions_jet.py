import numpy as np

"""
Dictionary of postprocessing operations for conditioning and target variables.
It is generated make_dataset function. Values of dictionary are list objects in which
sepcify preprocessing operation. Every operation has the following template

                       ["string", *pars]

where "string" tells which operation to perform and *pars its parameters. Such operations are

unsmearing: ["d", [inf, sup]]
transformation: ["i", func, [a, b]]  # func(x - b) / a

In the case of multiple operations, order follows the operation list indexing.
"""

target_dictionary = {
    "Electron_charge": [["c", 0, [-1, 1]]],
    "Electron_convVeto": [["d", None, None]],
    "Electron_deltaEtaSC": [["i", np.tan, [10, 0]]],
    "Electron_dr03EcalRecHitSumEt": [
        ["d", [-np.inf, -2], np.log(1e-3)],
        ["i", np.exp, [1, 1e-3]],
    ],
    "Electron_dr03HcalDepth1TowerSumEt": [
        ["d", [-np.inf, -2], np.log(1e-3)],
        ["i", np.exp, [1, 1e-3]],
    ],
    "Electron_dr03TkSumPt": [
        ["d", [-np.inf, -2], np.log(1e-3)],
        ["i", np.exp, [1, 1e-3]],
    ],
    "Electron_dr03TkSumPtHEEP": [
        ["d", [-np.inf, -2], np.log(1e-3)],
        ["i", np.exp, [1, 1e-3]],
    ],
    "Electron_dxy": [["i", np.tan, [10, 0]]],
    "Electron_dxyErr": [["i", np.exp, [1, 1e-3]]],
    "Electron_dz": [["i", np.tan, [10, 0]]],
    "Electron_dzErr": [["i", np.exp, [1, 1e-3]]],
    "Electron_eInvMinusPInv": [["i", np.tan, [10, 0]]],
    "Electron_energyErr": [["i", np.expm1, [1, 0]]],
    "Electron_etaMinusGen": [["i", np.tan, [10, 0]]],
    "Electron_hoe": [["d", [-np.inf, -6], np.log(1e-3)], ["i", np.exp, [1, 1e-3]]],
    "Electron_ip3d": [["i", np.exp, [1, 1e-3]]],
    "Electron_isPFcand": [["d", None, None]],
    "Electron_jetPtRelv2": [["i", np.expm1, [1, 0]]],
    "Electron_jetRelIso": [["i", np.exp, [10, 1e-2]]],
    "Electron_lostHits": [["d", None, None]],
    "Electron_miniPFRelIso_all": [
        ["d", [-np.inf, -5.5], np.log(1e-3)],
        ["i", np.exp, [1, 1e-3]],
    ],
    "Electron_miniPFRelIso_chg": [
        ["d", [-np.inf, -5.5], np.log(1e-3)],
        ["i", np.exp, [1, 1e-3]],
    ],
    "Electron_mvaFall17V1Iso": [],
    "Electron_mvaFall17V1Iso_WP80": [["d", None, None]],
    "Electron_mvaFall17V1Iso_WP90": [["d", None, None]],
    "Electron_mvaFall17V1Iso_WPL": [["d", None, None]],
    "Electron_mvaFall17V1noIso": [],
    "Electron_mvaFall17V1noIso_WP80": [["d", None, None]],
    "Electron_mvaFall17V1noIso_WP90": [["d", None, None]],
    "Electron_mvaFall17V1noIso_WPL": [["d", None, None]],
    "Electron_mvaFall17V2Iso": [],
    "Electron_mvaFall17V2Iso_WP80": [["d", None, None]],
    "Electron_mvaFall17V2Iso_WP90": [["d", None, None]],
    "Electron_mvaFall17V2Iso_WPL": [["d", None, None]],
    "Electron_mvaFall17V2noIso": [],
    "Electron_mvaFall17V2noIso_WP80": [["d", None, None]],
    "Electron_mvaFall17V2noIso_WP90": [["d", None, None]],
    "Electron_mvaFall17V2noIso_WPL": [["d", None, None]],
    "Electron_mvaTTH": [],
    "Electron_pfRelIso03_all": [
        ["d", [-np.inf, -5.5], np.log(1e-3)],
        ["i", np.exp, [1, 1e-3]],
    ],
    "Electron_pfRelIso03_chg": [
        ["d", [-np.inf, -5.5], np.log(1e-3)],
        ["i", np.exp, [1, 1e-3]],
    ],
    "Electron_phiMinusGen": [["i", np.tan, [10, 0]]],
    "Electron_ptRatio": [["i", np.expm1, [1, 0]]],
    "Electron_r9": [["i", np.exp, [1, 1e-2]]],
    "Electron_seedGain": [["d", None, None]],
    "Electron_sieie": [["i", np.exp, [10, 1e-1]]],
    "Electron_sip3d": [["i", np.expm1, [1, 0]]],
    "Electron_tightCharge": [["d", None, None]],
}
