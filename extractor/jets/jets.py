import os
import ROOT

module_path = os.path.join(os.path.dirname(__file__), "jets.h")

ROOT.gInterpreter.ProcessLine(f'#include "{module_path}"')


def extractGenJetFeatures(df):
    """for going from GenJet to reco jet

    Args:
        df (rdataframe): original rdataframe (should be cleaned? to be decided)

    Returns:
        rdataframe: rdataframe with new features
    """
    extracted = (
        df
        .Define("JetMask", "Jet_genJetIdx >=0  && Jet_genJetIdx < nGenJet")
        .Define("MatchedGenJets", "Jet_genJetIdx[JetMask]")
        .Define(
            "MuonMaskJ",
            "(GenPart_pdgId == 13 | GenPart_pdgId == -13)&&((GenPart_statusFlags & 8192) > 0)",
        )
        .Define("MMuon_pt", "GenPart_pt[MuonMaskJ]")
        .Define("MMuon_eta", "GenPart_eta[MuonMaskJ]")
        .Define("MMuon_phi", "GenPart_phi[MuonMaskJ]")
        .Define("MGenJet_pt", "Take(GenJet_pt, MatchedGenJets)")
        .Define("MGenJet_eta", "Take(GenJet_eta, MatchedGenJets)")
        .Define("MGenJet_phi", "Take(GenJet_phi, MatchedGenJets)")
        .Define("MGenJet_mass", "Take(GenJet_mass, MatchedGenJets)")
        .Define("MGenJet_hadronFlavourUChar", "Take(GenJet_hadronFlavour, MatchedGenJets)")
        .Define(
            "MGenJet_hadronFlavour",
            "static_cast<ROOT::VecOps::RVec<int>>(MGenJet_hadronFlavourUChar)",
        )
        .Define("MGenJet_partonFlavour", "Take(GenJet_partonFlavour, MatchedGenJets)")
        .Define(
            "MGenJet_EncodedPartonFlavour_light",
            "gen_jet_flavour_encoder(MGenJet_partonFlavour, ROOT::VecOps::RVec<int>{1,2,3})",
        )
        .Define(
            "MGenJet_EncodedPartonFlavour_gluon",
            "gen_jet_flavour_encoder(MGenJet_partonFlavour, ROOT::VecOps::RVec<int>{21})",
        )
        .Define(
            "MGenJet_EncodedPartonFlavour_c",
            "gen_jet_flavour_encoder(MGenJet_partonFlavour, ROOT::VecOps::RVec<int>{4})",
        )
        .Define(
            "MGenJet_EncodedPartonFlavour_b",
            "gen_jet_flavour_encoder(MGenJet_partonFlavour, ROOT::VecOps::RVec<int>{5})",
        )
        .Define(
            "MGenJet_EncodedPartonFlavour_undefined",
            "gen_jet_flavour_encoder(MGenJet_partonFlavour, ROOT::VecOps::RVec<int>{0})",
        )
        .Define(
            "MGenJet_EncodedHadronFlavour_b",
            "gen_jet_flavour_encoder(MGenJet_hadronFlavour, ROOT::VecOps::RVec<int>{5})",
        )
        .Define(
            "MGenJet_EncodedHadronFlavour_c",
            "gen_jet_flavour_encoder(MGenJet_hadronFlavour, ROOT::VecOps::RVec<int>{4})",
        )
        .Define(
            "MGenJet_EncodedHadronFlavour_light",
            "gen_jet_flavour_encoder(MGenJet_hadronFlavour, ROOT::VecOps::RVec<int>{0})",
        )
        .Define(
            "MClosestMuon_dr",
            "closest_muon_dr(GenJet_eta, GenJet_phi,MMuon_eta, MMuon_phi)",
        )
        .Define(
            "MClosestMuon_deta",
            "closest_muon_deta(GenJet_eta, GenJet_phi,MMuon_eta, MMuon_phi)",
        )
        .Define(
            "MClosestMuon_dphi",
            "closest_muon_dphi(GenJet_eta, GenJet_phi,MMuon_eta, MMuon_phi)",
        )
        .Define(
            "MClosestMuon_pt",
            "closest_muon_pt(GenJet_eta, GenJet_phi,MMuon_eta, MMuon_phi, MMuon_pt)",
        )
        .Define(
            "MSecondClosestMuon_dr",
            "second_muon_dr(GenJet_eta, GenJet_phi,MMuon_eta, MMuon_phi)",
        )
        .Define(
            "MSecondClosestMuon_deta",
            "second_muon_deta(GenJet_eta, GenJet_phi,MMuon_eta, MMuon_phi)",
        )
        .Define(
            "MSecondClosestMuon_dphi",
            "second_muon_dphi(GenJet_eta, GenJet_phi,MMuon_eta, MMuon_phi)",
        )
        .Define(
            "MSecondClosestMuon_pt",
            "second_muon_pt(GenJet_eta, GenJet_phi,MMuon_eta, MMuon_phi, MMuon_pt)",
        )
        .Define("MJet_area", "Jet_area[JetMask]")
        .Define("MJet_bRegCorr", "Jet_bRegCorr[JetMask]")
        .Define("MJet_bRegRes", "Jet_bRegRes[JetMask]")
        .Define("MJet_btagCSVV2", "Jet_btagCSVV2[JetMask]")
        .Define("MJet_btagDeepB", "Jet_btagDeepB[JetMask]")
        .Define("MJet_btagDeepCvB", "Jet_btagDeepCvB[JetMask]") # was DeepC
        .Define("MJet_btagDeepCvL", "Jet_btagDeepCvL[JetMask]")
        .Define("MJet_btagDeepFlavB", "Jet_btagDeepFlavB[JetMask]")
        .Define("MJet_btagDeepFlavCvB", "Jet_btagDeepFlavCvB[JetMask]") # was FlavC
        .Define("MJet_btagDeepFlavCvL", "Jet_btagDeepFlavCvL[JetMask]")
        .Define("MJet_btagDeepFlavQG", "Jet_btagDeepFlavQG[JetMask]")
        .Define("MJet_cRegCorr", "Jet_cRegCorr[JetMask]")
        .Define("MJet_cRegRes", "Jet_cRegRes[JetMask]")
        .Define("MJet_chEmEF", "Jet_chEmEF[JetMask]")
        .Define("MJet_chFPV0EF", "Jet_chFPV0EF[JetMask]")		
        .Define("MJet_chHEF", "Jet_chHEF[JetMask]")			
        .Define("MJet_cleanmask", "Jet_cleanmask[JetMask]")			
        .Define("MJet_etaMinusGen", "Jet_eta[JetMask]-MGenJet_eta")			
        .Define("MJet_hadronFlavour", "Jet_hadronFlavour[JetMask]")
        .Define("MJet_hfadjacentEtaStripsSize", "Jet_hfadjacentEtaStripsSize[JetMask]")
        .Define("MJet_hfcentralEtaStripSize", "Jet_hfcentralEtaStripSize[JetMask]")
        .Define("MJet_hfsigmaEtaEta", "Jet_hfsigmaEtaEta[JetMask]")
        .Define("MJet_hfsigmaPhiPhi", "Jet_hfsigmaPhiPhi[JetMask]")
        .Define("MJet_jetId", "Jet_jetId[JetMask]")
        .Define("MJet_massRatio", "Jet_mass[JetMask]/MGenJet_mass")
        .Define("MJet_muEF", "Jet_muEF[JetMask]")
        .Define("MJet_muonSubtrFactor", "Jet_muonSubtrFactor[JetMask]")
        .Define("MJet_nConstituents", "Jet_nConstituents[JetMask]")
        .Define("MJet_nElectrons", "Jet_nElectrons[JetMask]")
        .Define("MJet_nMuons", "Jet_nMuons[JetMask]")
        .Define("MJet_neEmEF", "Jet_neEmEF[JetMask]")
        .Define("MJet_neHEF", "Jet_neHEF[JetMask]")
        .Define("MJet_partonFlavour", "Jet_partonFlavour[JetMask]")
        .Define("MJet_phifiltered", "Jet_phi[JetMask]")
        .Define("MJet_phiMinusGen", "DeltaPhi(MJet_phifiltered, MGenJet_phi)")
        .Define("MJet_ptRatio", "Jet_pt[JetMask]/MGenJet_pt")
        .Define("MJet_puId", "Jet_puId[JetMask]")
        .Define("MJet_puIdDisc", "Jet_puIdDisc[JetMask]")
        .Define("MJet_qgl", "Jet_qgl[JetMask]")
        .Define("MJet_rawFactor", "Jet_rawFactor[JetMask]")
    )

    return extracted