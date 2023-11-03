import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

from preprocessing import preprocessing
from columns import gen, reco

import pandas as pd
import numpy as np

if __name__ == "__main__":
    
    df = pd.read_pickle("~/Downloads/preprocessed_concat.pkl")

    # Define the exact columns to keep in the specified order
    columns_to_keep_exact = [
        "MgenjetAK8_pt",
        "MgenjetAK8_phi",
        "MgenjetAK8_eta",
        "MgenjetAK8_hadronFlavour",
        "MgenjetAK8_partonFlavour",
        "MgenjetAK8_mass",
        "MgenjetAK8_ncFlavour",
        "MgenjetAK8_nbFlavour",
        "has_H_within_0_8",
        "is_signal",
        "Mpt_ratio",
        "Meta_sub",
        "Mphi_sub",
        "Mfatjet_msoftdrop",
        "Mfatjet_particleNetMD_XbbvsQCD",
    ]

    # Select only the specified columns from the dataframe
    df_selected = df[columns_to_keep_exact]

    signal = df_selected[df_selected['is_signal'] == 1].values
    background = df_selected[df_selected['is_signal'] == 0].values

    # oversample the signal
    signal = np.repeat(signal, 15, axis=0)

    # concatenate the signal and background
    data = np.concatenate([signal, background])

    # shuffle the data
    np.random.shuffle(data)

    # get back to pandas dataframe
    df = pd.DataFrame(data, columns=columns_to_keep_exact)
    # print stats
    print(df.head())
    print(df.describe())
    # save the dataframe
    df.to_pickle("preprocessed_concat_oversampled.pkl")



