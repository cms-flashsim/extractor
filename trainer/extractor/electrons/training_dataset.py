import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

from preprocessing import make_dataset

from columns import gen_ele, gen_pho, gen_jet, reco_columns

from prep_actions_ele import target_dictionary as target_dictionary_ele
from prep_actions_pho import target_dictionary as target_dictionary_pho
from prep_actions_jet import target_dictionary as target_dictionary_jet

dataset_path = os.path.join(os.path.dirname(__file__), "dataset")

nfiles = 9

# GenElectron training dataset

filenames = [f"MElectrons_{i}_ele.root:MElectrons" for i in range(nfiles)]

filepaths = [os.path.join(dataset_path, f) for f in filenames]

scale_file = os.path.join(os.path.dirname(__file__), "scale_factors_ele_ele.json")
range_file = os.path.join(os.path.dirname(__file__), "ranges_ele_ele.json")

make_dataset(
    filepaths,
    "MElectrons_ele",
    target_dictionary_ele,
    scale_file,
    range_file,
    gen_ele,
    reco_columns,
)


# GenPhoton training dataset

filenames = [f"MElectrons_{i}_pho.root:MElectrons" for i in range(nfiles)]

filepaths = [os.path.join(dataset_path, f) for f in filenames]

scale_file = os.path.join(os.path.dirname(__file__), "scale_factors_ele_pho.json")
range_file = os.path.join(os.path.dirname(__file__), "ranges_ele_pho.json")

make_dataset(
    filepaths,
    "MElectrons_pho",
    target_dictionary_ele,
    scale_file,
    range_file,
    gen_pho,
    reco_columns,
)


# GenJet training dataset

filenames = [f"MElectrons_{i}_jet.root:MElectrons" for i in range(nfiles)]

filepaths = [os.path.join(dataset_path, f) for f in filenames]

scale_file = os.path.join(os.path.dirname(__file__), "scale_factors_ele_jet.json")
range_file = os.path.join(os.path.dirname(__file__), "ranges_ele_jet.json")

make_dataset(
    filepaths,
    "MElectrons_jet",
    target_dictionary_ele,
    scale_file,
    range_file,
    gen_jet,
    reco_columns,
)
