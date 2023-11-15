import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

from columns import gen, reco

def postprocess_disc(disc):
    range_disc = 21.543403195332097
    min = -0.7688173116541402
    disc = np.where(disc<min, -0.1, (np.tanh(disc*range_disc)+1)/2)
    return disc

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
    #plt.hist(df_selected["Mfatjet_particleNetMD_XbbvsQCD"], bins=100)
    #plt.show()
    
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
    #plt.hist(df_selected["Mfatjet_particleNetMD_XbbvsQCD"], bins=100)
    #plt.show()

    # print min
    print("min: ", df_selected["Mfatjet_particleNetMD_XbbvsQCD"].min())

    # create mask to add only on points with Mfatjet_particleNetMD_XbbvsQCD > -0.1
    mask = df_selected["Mfatjet_particleNetMD_XbbvsQCD"] > -0.1
    # print number of points <=-0.1
    print("number of points <=-0.1: ", len(df_selected["Mfatjet_particleNetMD_XbbvsQCD"][~mask]))

    # on Mfatjet_particleNetMD_XbbvsQCD if x>0.9 put 0.9+tanh(x-0.9)
    # bin = 0.5
    # df_selected["Mfatjet_particleNetMD_XbbvsQCD"] = df_selected["Mfatjet_particleNetMD_XbbvsQCD"].apply(
    #     lambda x: bin + np.arctanh(x) - np.arctanh(bin) if x > bin else x
    # )
    #  # if x<bin put 0.9-tanh(x)- np.arctanh(bin)
    # df_selected["Mfatjet_particleNetMD_XbbvsQCD"] = df_selected["Mfatjet_particleNetMD_XbbvsQCD"].apply(
    #     lambda x: bin + np.arctanh(2*x - 1) + np.arctanh(2*bin-1) if x < bin else x)

    # apply np.arctanh(2*x - 1) on Mfatjet_particleNetMD_XbbvsQCD only in mask
    df_selected["Mfatjet_particleNetMD_XbbvsQCD"][mask] = df_selected["Mfatjet_particleNetMD_XbbvsQCD"][mask].apply(
        lambda x: np.arctanh(2*x - 1))
    # normalize points in mask dividing by max-min
    range = (df_selected["Mfatjet_particleNetMD_XbbvsQCD"].max() - df_selected["Mfatjet_particleNetMD_XbbvsQCD"].min())
    print("range: ", range)
    df_selected["Mfatjet_particleNetMD_XbbvsQCD"][mask] = df_selected["Mfatjet_particleNetMD_XbbvsQCD"][mask]/range
    print('min: ', df_selected["Mfatjet_particleNetMD_XbbvsQCD"].min())
    df_selected["Mfatjet_particleNetMD_XbbvsQCD"][~mask] = -1
    # plot hist between 0.8 and 3
    # data = df_selected["Mfatjet_particleNetMD_XbbvsQCD"].values
    # data = data[(data > 0.4) & (data < 3)]
    # plt.hist(data, bins=100)
    # plt.show()

    # df_selected["Mfatjet_particleNetMD_XbbvsQCD"] = postprocess_disc(df_selected["Mfatjet_particleNetMD_XbbvsQCD"])

    # plot all
    plt.hist(df_selected["Mfatjet_particleNetMD_XbbvsQCD"], bins=100)
    #plt.yscale("log")
    plt.show()
    # save
    
    df_selected.to_pickle("fatjet_oversampled.pkl")
