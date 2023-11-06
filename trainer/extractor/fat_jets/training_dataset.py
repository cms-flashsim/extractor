import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

from columns import gen, reco


if __name__ == "__main__":

    print("Starting to make dataset for fake jets")
    df = pd.read_pickle("/home/fvaselli/Downloads/preprocessed_QCD_noPU_VBF_SM.pkl")

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

    # plot histogram of Mfatjet_particleNetMD_XbbvsQCD
    plt.hist(df_selected["Mfatjet_particleNetMD_XbbvsQCD"], bins=100)
    plt.show()
    
    # oversample the signal
    sig_values = df_selected[df_selected["is_signal"] == 1].values
    sig_values = sig_values.repeat(20, axis=0)

    # concat
    df_selected = pd.concat([df_selected, pd.DataFrame(sig_values, columns=df_selected.columns)])
    # shuffle df
    df_selected = df_selected.sample(frac=1).reset_index(drop=True)
    # print is_signal values
    # print(df_selected["is_signal"].values[45:70])

    # print info
    print("df_selected shape: ", df_selected.shape)
    print(df_selected.head())

    # plot histogram of Mfatjet_particleNetMD_XbbvsQCD
    plt.hist(df_selected["Mfatjet_particleNetMD_XbbvsQCD"], bins=100)
    plt.show()
    # save
    df_selected.to_pickle("fatjet_oversampled.pkl")
