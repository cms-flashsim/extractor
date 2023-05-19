import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

from preprocessing import make_dataset
from prep_actions import target_dictionary_muons as target_dictionary
from columns import muon_cond, reco_columns


datasets = os.listdir(os.path.join(os.path.dirname(__file__), "dataset"))

scale_file = os.path.join(os.path.dirname(__file__), "scale_factors_muons.json")

inputtrees = [
    f"{os.path.join(os.path.dirname(__file__), 'dataset', f)}:MMuons" for f in datasets
]

scale_file = os.path.join(os.path.dirname(__file__), "scale_factors_muons.json")

make_dataset(
    inputtrees, "MMuons", target_dictionary, scale_file, muon_cond, reco_columns
)
