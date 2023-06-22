import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

from preprocessing import make_dataset
from prep_actions import target_dictionary_jets as target_dictionary
from columns import jet_cond, reco_columns

if __name__ == "__main__":

    print("Starting to make dataset for fake jets")
    datasets = os.listdir(os.path.join(os.path.dirname(__file__), "dataset"))

    scale_file = os.path.join(os.path.dirname(__file__), "scale_factors_fakejets.json")
    range_file = os.path.join(os.path.dirname(__file__), "ranges_fakejets.json")

    inputtrees = [
        f"{os.path.join(os.path.dirname(__file__), 'dataset', f)}:MFakeJets" for f in datasets
    ]
    print("Read input trees")
    make_dataset(
        inputtrees,
        "MFakeJets",
        target_dictionary,
        scale_file,
        range_file,
        jet_cond,
        reco_columns,
    )
