import os
import json

from muons import extract_muons

root = "/gpfs/ddn/srm/cms/store/mc/RunIISummer20UL18NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/"

ttbar_training_files = [
    "120000/0520A050-AF68-EF43-AA5B-5AA77C74ED73.root",
    # "120000/0E9EA19A-AE0E-3149-88C3-D733240FF5AB.root",
    # "120000/143F7726-375A-3D48-9D53-D6B071CED8F6.root",
    # "120000/15FC5EA3-70AA-B640-8748-BD5E1BB84CAC.root",
    # "120000/1CD61F25-9DE8-D741-9200-CCBBA61E5A0A.root",
    # "120000/1D885366-E280-1243-AE4F-532D326C2386.root",
    # "120000/23AD2392-C48B-D643-9E16-C93730AA4A02.root",
    # "120000/245961C8-DE06-8F4F-9E92-ED6F30A097C4.root",
    # "120000/262EAEE2-14CC-2A44-8F4B-B1A339882B25.root",
]


file_paths = [os.path.join(root, f) for f in ttbar_training_files]

if not os.path.exists(os.path.join(os.path.dirname(__file__), "dataset")):
    os.mkdir(os.path.join(os.path.dirname(__file__), "dataset"))

extracted = [os.path.join("dataset", f"MMuons_{i}.root") for i in range(len(file_paths))]

d = {
    "RECOMUON_GENMUON": (0, 0),
}

if __name__ == "__main__":
    for file_in, file_out in zip(file_paths, extracted):
        extract_muons(file_in, file_out, d)

    with open(os.path.join(os.path.dirname(__file__), "match_dict.json"), "w") as f:
        json.dump(d, f)
