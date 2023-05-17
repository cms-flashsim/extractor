# conditioning and reco columns for muons

muon_cond = [
    "MGenMuon_eta",
    "MGenMuon_phi",
    "MGenMuon_pt",
    "MGenMuon_charge",
    "MGenPart_statusFlags0",
    "MGenPart_statusFlags1",
    "MGenPart_statusFlags2",
    "MGenPart_statusFlags3",
    "MGenPart_statusFlags4",
    "MGenPart_statusFlags5",
    "MGenPart_statusFlags6",
    "MGenPart_statusFlags7",
    "MGenPart_statusFlags8",
    "MGenPart_statusFlags9",
    "MGenPart_statusFlags10",
    "MGenPart_statusFlags11",
    "MGenPart_statusFlags12",
    "MGenPart_statusFlags13",
    "MGenPart_statusFlags14",
    "ClosestJet_dr",
    "ClosestJet_deta",
    "ClosestJet_dphi",
    "ClosestJet_pt",
    "ClosestJet_mass",
    "ClosestJet_EncodedPartonFlavour_light",
    "ClosestJet_EncodedPartonFlavour_gluon",
    "ClosestJet_EncodedPartonFlavour_c",
    "ClosestJet_EncodedPartonFlavour_b",
    "ClosestJet_EncodedPartonFlavour_undefined",
    "ClosestJet_EncodedHadronFlavour_b",
    "ClosestJet_EncodedHadronFlavour_c",
    "ClosestJet_EncodedHadronFlavour_light",
    "Pileup_gpudensity",
    "Pileup_nPU",
    "Pileup_nTrueInt",
    "Pileup_pudensity",
    "Pileup_sumEOOT",
    "Pileup_sumLOOT",
]
        .Define("MMuon_charge", "Muon_charge[MuonMask]")
        .Define("MMuon_cleanmask", "Muon_cleanmask[MuonMask]")
        .Define("MMuon_dxy", "Muon_dxy[MuonMask]")
        .Define("MMuon_dxyErr", "Muon_dxyErr[MuonMask]")
        .Define("MMuon_dxybs", "Muon_dxybs[MuonMask]")
        .Define("MMuon_dz", "Muon_dz[MuonMask]")
        .Define("MMuon_dzErr", "Muon_dzErr[MuonMask]")
        .Define("MMuon_etaMinusGen", "Muon_eta[MuonMask]-MGenMuon_eta")
        .Define("MMuon_highPtId", "Muon_highPtId[MuonMask]")
        .Define("MMuon_highPurity", "Muon_highPurity[MuonMask]")
        .Define("MMuon_inTimeMuon", "Muon_inTimeMuon[MuonMask]")
        .Define("MMuon_ip3d", "Muon_ip3d[MuonMask]")
        .Define("MMuon_isGlobal", "Muon_isGlobal[MuonMask]")
        .Define("MMuon_isPFcand", "Muon_isPFcand[MuonMask]")
        .Define("MMuon_isStandalone", "Muon_isStandalone[MuonMask]")
        .Define("MMuon_isTracker", "Muon_isTracker[MuonMask]")
        .Define("MMuon_jetPtRelv2", "Muon_jetPtRelv2[MuonMask]")
        .Define("MMuon_jetRelIso", "Muon_jetRelIso[MuonMask]")
        .Define("MMuon_looseId", "Muon_looseId[MuonMask]")
        .Define("MMuon_mediumId", "Muon_mediumId[MuonMask]")
        .Define("MMuon_mediumPromptId", "Muon_mediumPromptId[MuonMask]")
        .Define("MMuon_miniIsoId", "Muon_miniIsoId[MuonMask]")
        .Define("MMuon_miniPFRelIso_all", "Muon_miniPFRelIso_all[MuonMask]")
        .Define("MMuon_miniPFRelIso_chg", "Muon_miniPFRelIso_chg[MuonMask]")
        .Define("MMuon_multiIsoId", "Muon_multiIsoId[MuonMask]")
        .Define("MMuon_mvaId", "Muon_mvaId[MuonMask]")
        .Define("MMuon_mvaLowPt", "Muon_mvaLowPt[MuonMask]")
        .Define("MMuon_mvaLowPtId", "Muon_mvaLowPtId[MuonMask]")
        .Define("MMuon_mvaTTH", "Muon_mvaTTH[MuonMask]")
        .Define("MMuon_nStations", "Muon_nStations[MuonMask]")
        .Define("MMuon_nTrackerLayers", "Muon_nTrackerLayers[MuonMask]")
        .Define("MMuon_pfIsoId", "Muon_pfIsoId[MuonMask]")
        .Define("MMuon_pfRelIso03_all", "Muon_pfRelIso03_all[MuonMask]")
        .Define("MMuon_pfRelIso03_chg", "Muon_pfRelIso03_chg[MuonMask]")
        .Define("MMuon_pfRelIso04_all", "Muon_pfRelIso04_all[MuonMask]")
        .Define("MMuon_filteredphi", "Muon_phi[MuonMask]")
        .Define("MMuon_phiMinusGen", "DeltaPhi(MMuon_filteredphi, MGenMuon_phi)")
        .Define("MMuon_ptRatio", "Muon_pt[MuonMask]/MGenMuon_pt")
        .Define("MMuon_ptErr", "Muon_ptErr[MuonMask]")
        .Define("MMuon_puppiIsoId", "Muon_puppiIsoId[MuonMask]")
        .Define("MMuon_segmentComp", "Muon_segmentComp[MuonMask]")
        .Define("MMuon_sip3d", "Muon_sip3d[MuonMask]")
        .Define("MMuon_softId", "Muon_softId[MuonMask]")
        .Define("MMuon_softMva", "Muon_softMva[MuonMask]")
        .Define("MMuon_softMvaId", "Muon_softMvaId[MuonMask]")
        .Define("MMuon_tightCharge", "Muon_tightCharge[MuonMask]")
        .Define("MMuon_tightId", "Muon_tightId[MuonMask]")
        .Define("MMuon_tkIsoId", "Muon_tkIsoId[MuonMask]")
        .Define("MMuon_tkRelIso", "Muon_tkRelIso[MuonMask]")
        .Define("MMuon_triggerIdLoose", "Muon_triggerIdLoose[MuonMask]")
        .Define("MMuon_tunepRelPt", "Muon_tunepRelPt[MuonMask]")
        .Define("MMuon_nMuon", "Muon_nMuon[MuonMask]")

# columns are as above after the MMuon_ prefix

muon_names = [
    "charge",
    "cleanmask",
    "dxy",
    "dxyErr",
    "dxybs",
    "dz",
    "dzErr",
    "etaMinusGen",
    "highPtId",
    "highPurity",
    "inTimeMuon",
    "ip3d",
    "isGlobal",
    "isPFcand",
    "isStandalone",
    "isTracker",
    "jetPtRelv2",
    "jetRelIso",
    "looseId",
    "mediumId",
    "mediumPromptId",
    "miniIsoId",
    "miniPFRelIso_all",
    "miniPFRelIso_chg",
    "multiIsoId",
    "mvaId",
    "mvaLowPt",
    "mvaLowPtId",
    "mvaTTH",
    "nStations",
    "nTrackerLayers",
    "pfIsoId",
    "pfRelIso03_all",
    "pfRelIso03_chg",
    "pfRelIso04_all",
    "filteredphi",
    "phiMinusGen",
    "ptRatio",
    "ptErr",
    "puppiIsoId",
    "segmentComp",
    "sip3d",
    "softId",
    "softMva",
    "softMvaId",
    "tightCharge",
    "tightId",
    "tkIsoId",
    "tkRelIso",
    "triggerIdLoose",
    "nMuon",
]

# NOTE Charge is not included in the reco columns here, but afterwards
reco_columns = [f"MMuon_{name}" for name in muon_names]