import os
import ROOT

from columns import jet_cond, reco_columns

module_path = os.path.join(os.path.dirname(__file__), "jets.h")

ROOT.gInterpreter.ProcessLine(f'#include "{module_path}"')


def extractFakeJetFeatures(df):
    """for going from GenJet to reco jet

    Args:
        df (rdataframe): original rdataframe (should be cleaned? to be decided)

    Returns:
        rdataframe: rdataframe with new features
    """
    extracted = (
        df.Define("JetMask", "Jet_genJetIdx < 0 || Jet_genJetIdx > nGenJet")
        .Define("MJet_area", "Jet_area[JetMask]")
        .Define("MJet_bRegCorr", "Jet_bRegCorr[JetMask]")
        .Define("MJet_bRegRes", "Jet_bRegRes[JetMask]")
        .Define("MJet_btagCSVV2", "Jet_btagCSVV2[JetMask]")
        .Define("MJet_btagDeepB", "Jet_btagDeepB[JetMask]")
        .Define("MJet_btagDeepCvB", "Jet_btagDeepCvB[JetMask]")  # was DeepC
        .Define("MJet_btagDeepCvL", "Jet_btagDeepCvL[JetMask]")
        .Define("MJet_btagDeepFlavB", "Jet_btagDeepFlavB[JetMask]")
        .Define("MJet_btagDeepFlavCvB", "Jet_btagDeepFlavCvB[JetMask]")  # was FlavC
        .Define("MJet_btagDeepFlavCvL", "Jet_btagDeepFlavCvL[JetMask]")
        .Define("MJet_btagDeepFlavQG", "Jet_btagDeepFlavQG[JetMask]")
        .Define("MJet_cRegCorr", "Jet_cRegCorr[JetMask]")
        .Define("MJet_cRegRes", "Jet_cRegRes[JetMask]")
        .Define("MJet_chEmEF", "Jet_chEmEF[JetMask]")
        .Define("MJet_chFPV0EF", "Jet_chFPV0EF[JetMask]")
        .Define("MJet_chHEF", "Jet_chHEF[JetMask]")
        .Define("MJet_cleanmask", "Jet_cleanmask[JetMask]")
        .Define("MJet_eta", "Jet_eta[JetMask]")
        .Define("MJet_hadronFlavour", "Jet_hadronFlavour[JetMask]")
        .Define("MJet_hfadjacentEtaStripsSize", "Jet_hfadjacentEtaStripsSize[JetMask]")
        .Define("MJet_hfcentralEtaStripSize", "Jet_hfcentralEtaStripSize[JetMask]")
        .Define("MJet_hfsigmaEtaEta", "Jet_hfsigmaEtaEta[JetMask]")
        .Define("MJet_hfsigmaPhiPhi", "Jet_hfsigmaPhiPhi[JetMask]")
        .Define("MJet_jetId", "Jet_jetId[JetMask]")
        .Define("MJet_mass", "Jet_mass[JetMask]")
        .Define("MJet_muEF", "Jet_muEF[JetMask]")
        .Define("MJet_muonSubtrFactor", "Jet_muonSubtrFactor[JetMask]")
        .Define("MJet_nConstituents", "Jet_nConstituents[JetMask]")
        .Define("MJet_nElectrons", "Jet_nElectrons[JetMask]")
        .Define("MJet_nMuons", "Jet_nMuons[JetMask]")
        .Define("MJet_neEmEF", "Jet_neEmEF[JetMask]")
        .Define("MJet_neHEF", "Jet_neHEF[JetMask]")
        .Define("MJet_partonFlavour", "Jet_partonFlavour[JetMask]")
        .Define("MJet_phi", "Jet_phi[JetMask]")
        .Define("MJet_pt", "Jet_pt[JetMask]")
        .Define("MJet_puId", "Jet_puId[JetMask]")
        .Define("MJet_puIdDisc", "Jet_puIdDisc[JetMask]")
        .Define("MJet_qgl", "Jet_qgl[JetMask]")
        .Define("MJet_rawFactor", "Jet_rawFactor[JetMask]")
    )

    return extracted

def extract_fakejets(inputname, outputname, dict):
    ROOT.EnableImplicitMT()

    print(f"Processing {inputname}...")

    d = ROOT.RDataFrame("Events", inputname)

    d = extractFakeJetFeatures(d)

    n_match, n_reco = dict["RECOJET_GENJET"]

    n_match += d.Histo1D("MJet_pt").GetEntries()
    n_reco += d.Histo1D("Jet_pt").GetEntries()

    dict["RECOJET_GENJET"] = (n_match, n_reco)

    cols = jet_cond + reco_columns

    d.Snapshot("MFakeJets", outputname, cols)

    print(f"{outputname} written")
