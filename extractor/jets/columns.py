# conditioning and reco columns for jets

jet_cond = [
    "MClosestMuon_dr",
    "MClosestMuon_pt",
    "MClosestMuon_deta",
    "MClosestMuon_dphi",
    "MSecondClosestMuon_dr",
    "MSecondClosestMuon_pt",
    "MSecondClosestMuon_deta",
    "MSecondClosestMuon_dphi",
    "GenJet_eta",
    "GenJet_mass",
    "GenJet_phi",
    "GenJet_pt",
    "GenJet_EncodedPartonFlavour_light",
    "GenJet_EncodedPartonFlavour_gluon",
    "GenJet_EncodedPartonFlavour_c",
    "GenJet_EncodedPartonFlavour_b",
    "GenJet_EncodedPartonFlavour_undefined",
    "GenJet_EncodedHadronFlavour_b",
    "GenJet_EncodedHadronFlavour_c",
    "GenJet_EncodedHadronFlavour_light",
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
    "etaMinusGen",
    "hadronFlavour",
    "hfadjacentEtaStripsSize",
    "hfcentralEtaStripSize",
    "hfsigmaEtaEta",
    "hfsigmaPhiPhi",
    "jetId",
    "massRatio",
    "muEF",
    "muonSubtrFactor",
    "nConstituents",
    "nElectrons",
    "nMuons",
    "neEmEF",
    "neHEF",
    "partonFlavour",
    "phifiltered",
    "phiMinusGen",
    "ptRatio",
    "puId",
    "puIdDisc",
    "qgl",
    "rawFactor",
]

reco_columns = [f"MJet_{name}" for name in jet_names]

# NOTE we are calling the ratio/minus variables with the same name as the original
# for i, name in enumerate(reco_columns):
#     if name == "MElectron_pt":
#         reco_columns[i] = "MElectron_ptRatio"
#     elif name == "MElectron_phi":
#         reco_columns[i] = "MElectron_phiMinusGen"
#     elif name == "MElectron_eta":
#         reco_columns[i] = "MElectron_etaMinusGen"
