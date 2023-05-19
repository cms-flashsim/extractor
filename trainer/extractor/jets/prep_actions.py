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

target_dictionary_jets = {
    "MJet_area": [],
    "MJet_bRegCorr": [],
    "MJet_bRegRes": [],
    "MJet_btagCSVV2": [["gm", -0.5, 0.1, [-np.inf, -0.01]]], # was -1 before smearing
    "MJet_btagDeepB": [["gm", -0.5, 0.1, [-np.inf, -0.01]]], # was -1 before smearing
    "MJet_btagDeepCvB": [["gm", -0.5, 0.1, [-np.inf, -0.01]]], # was -1 before smearing
    "MJet_btagDeepCvL": [["gm", -0.5, 0.1, [-np.inf, -0.01]]], # was -1 before smearing
    "MJet_btagDeepFlavB": [],
    "MJet_btagDeepFlavCvB": [],
    "MJet_btagDeepFlavCvL": [],
    "MJet_btagDeepFlavQG": [],
    "MJet_cRegCorr": [],
    "MJet_cRegRes": [],
    "MJet_chEmEF": [["gm", -0.5, 0.1, [-np.inf, 0]],], # was 0 before smearing
    "MJet_chFPV0EF": [],
    "MJet_chHEF": [["t", np.arctan, [50, -50]]],
    "MJet_cleanmask": [["u", 0.5, None]],
    "MJet_etaMinusGen": [],
    "MJet_hadronFlavour": [["u", 0.5, None]],
    "MJet_hfadjacentEtaStripsSize": [["u", 0.5, None]],
    "MJet_hfcentralEtaStripSize": [["u", 0.5, None]],
    "MJet_hfsigmaEtaEta": [["gm", -0.5, 0.1, [-np.inf, 0]],], # -1 before smearing
    "MJet_hfsigmaPhiPhi": [["gm", -0.5, 0.1, [-np.inf, 0]],], # -1 before smearing
    "MJet_jetId": [["u", 0.5, None]],
    "MJet_mass": [],
    "MJet_muEF": [["gm", -0.5, 0.1, [-np.inf, 0]],], # 0 before smearing
    "MJet_muonSubtrFactor": [["gm", -0.5, 0.1, [-np.inf, 0]],], # 0 before smearing
    "MJet_nConstituents": [["u", 0.5, None]],
    "MJet_nElectrons": [["u", 0.5, None]],
    "MJet_nMuons": [["u", 0.5, None]],
    "MJet_neEmEF": [], 
    "MJet_neHEF": [],
    "MJet_partonFlavour": [["u", 0.5, None]],
    "MJet_phiMinusGen": [],
    "MJet_ptRatio": [["manual_range", [0.1, 5]]],
    "MJet_puId": [["u", 0.5, None]],
    "MJet_puIdDisc": [],
    "MJet_qgl": [],
    "MJet_rawFactor": [],

}
