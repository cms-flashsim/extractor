import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

from columns import gen, reco


if __name__ == "__main__":

    print("Starting to make dataset for fake jets")
    df = pd.read_pickle("/home/fvaselli/Downloads/preprocessed_concat(1).pkl")

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

    # print info
    print("df_selected shape: ", df_selected.shape)
    print(df_selected.head())

    # oversample the signal
    sig_values = df_selected[df_selected["is_signal"] == 1].values
    sig_values = sig_values.repeat(15, axis=0)

    # concat
    df_selected = pd.concat([df_selected, pd.DataFrame(sig_values, columns=df_selected.columns)])
    # shuffle
    df_selected = df_selected.sample(frac=1).reset_index(drop=True)

    # print info
    print("df_selected shape: ", df_selected.shape)
    print(df_selected.head())

    # save
    # df_selected.to_pickle("fatjet_oversampled.pkl")
