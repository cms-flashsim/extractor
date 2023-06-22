# conditioning and reco columns for jets

jet_cond = [
    "Pileup_nTrueInt",
    "MJet_pt",
    "MJet_eta",
    "MJet_phi",
]

jet_names = [
    "area",
    "bRegCorr",
    "bRegRes",
    "btagCSVV2",
    "btagDeepB",
    "btagDeepCvB",
    "btagDeepCvL",
    "btagDeepFlavB",
    "btagDeepFlavCvB",
    "btagDeepFlavCvL",
    "btagDeepFlavQG",
    "cRegCorr",
    "cRegRes",
    "chEmEF",
    "chFPV0EF",
    "chHEF",
    "cleanmask",
    "hadronFlavour",
    "hfadjacentEtaStripsSize",
    "hfcentralEtaStripSize",
    "hfsigmaEtaEta",
    "hfsigmaPhiPhi",
    "jetId",
    "mass",
    "muEF",
    "muonSubtrFactor",
    "nConstituents",
    "nElectrons",
    "nMuons",
    "neEmEF",
    "neHEF",
    "partonFlavour",
    "puId",
    "puIdDisc",
    "qgl",
    "rawFactor",
]

reco_columns = [f"MJet_{name}" for name in jet_names]