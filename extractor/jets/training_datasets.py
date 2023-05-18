import os
from extractor.preprocessing.preprocessing import make_dataset
from prep_actions import target_dictionary_jets as target_dictionary
from columns import jet_cond, reco_columns


datasets = os.listdir(os.path.join(os.path.dirname(__file__), "dataset"))

scale_file = os.path.join(os.path.dirname(__file__), "scale_factors_jets.json")

inputtrees = [
    f"{os.path.join(os.path.dirname(__file__), 'dataset', f)}:MJets" for f in datasets
]

make_dataset(inputtrees, "MJets", target_dictionary, scale_file, jet_cond, reco_columns)
