import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

from columns import gen, reco


def postprocess_disc(disc):
    range_disc = 20.56
    min = -0.8281470664258219
    disc = np.where(disc < min, -0.1, (np.tanh(disc * range_disc) + 1) / 2)
    return disc

 # Function to oversample signal
def oversample_signal(data):
    signal = data[data['is_signal'] == 1]
    oversampled_signal = signal.sample(len(data) - len(signal), replace=True)
    return pd.concat([data, oversampled_signal]).sample(frac=1).reset_index(drop=True)

def reshape_dataset(df, reshape_disc=True):
    if reshape_disc:
        print("Dataset shape: ", df.shape)
        print(df.head())

        print("Minimum value before reshaping: ", df["Mfatjet_particleNetMD_XbbvsQCD"].min())

        # Create a mask for values > -0.1
        mask = df["Mfatjet_particleNetMD_XbbvsQCD"] > -0.1
        print("Number of points <= -0.1: ", len(df["Mfatjet_particleNetMD_XbbvsQCD"][~mask]))

        # Apply transformation on selected points
        df.loc[mask, "Mfatjet_particleNetMD_XbbvsQCD"] = df.loc[mask, "Mfatjet_particleNetMD_XbbvsQCD"].apply(lambda x: np.arctanh(2 * x - 1))

        # Calculate the range for normalization
        value_range = 20.56
        # df.loc[mask, "Mfatjet_particleNetMD_XbbvsQCD"].max() - df.loc[mask, "Mfatjet_particleNetMD_XbbvsQCD"].min()
        print("Range after transformation: ", value_range)

        # Normalize the values within the mask
        df.loc[mask, "Mfatjet_particleNetMD_XbbvsQCD"] /= value_range
        min_value = df.loc[mask, "Mfatjet_particleNetMD_XbbvsQCD"].min()
        print("Minimum value after reshaping: ", min_value)

        # Set values outside the mask to -1
        df.loc[~mask, "Mfatjet_particleNetMD_XbbvsQCD"] = -1

        # Plot histogram
        plt.hist(df["Mfatjet_particleNetMD_XbbvsQCD"], bins=100)
        plt.show()
    else:
        value_range = 1
        min_value = 0

    return df, value_range, min_value

RESHAPE = True

if __name__ == "__main__":
    print("Starting to make dataset for fake jets")
    df = pd.read_pickle(
        "/home/fvaselli/Documents/trainer/trainer/extractor/fat_jets/preprocessed_all_QCD_PU200_VBF_SM_and_BSM.pkl"
    )
    print(df.columns)

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
        "Mhas_H_within_0_8",
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

    # Split dataset into training and validation sets
    train_df, val_df = train_test_split(
        df_selected, test_size=0.2, random_state=42, stratify=df_selected["is_signal"]
    )

    # print info
    print("train_df shape: ", train_df.shape)
    print("val_df shape: ", val_df.shape)

    # Oversample signal separately in training and validation sets
    train_df_oversampled = oversample_signal(train_df)
    val_df_oversampled = oversample_signal(val_df)

    # print info
    print("train_df_oversampled shape: ", train_df_oversampled.shape)
    print("val_df_oversampled shape: ", val_df_oversampled.shape)

    # Reshape the discriminator
    train_df_reshaped, range_disc, min_value = reshape_dataset(train_df_oversampled, reshape_disc=RESHAPE)
    val_df_reshaped, _, _ = reshape_dataset(val_df_oversampled, reshape_disc=RESHAPE)

    # apply postprocessing to discriminator and plot dists
    if RESHAPE:
        disc_train = postprocess_disc(train_df_reshaped["Mfatjet_particleNetMD_XbbvsQCD"])
        disc_val = postprocess_disc(val_df_reshaped["Mfatjet_particleNetMD_XbbvsQCD"])
    else:
        disc_train = train_df_reshaped["Mfatjet_particleNetMD_XbbvsQCD"]
        disc_val = val_df_reshaped["Mfatjet_particleNetMD_XbbvsQCD"]

    # Plot discriminator distributions
    plt.hist(disc_train, bins=100, label="train")
    plt.hist(disc_val, bins=100, label="val")
    plt.legend()
    plt.show()

    # Save the datasets
    if not RESHAPE:
        train_df_reshaped.to_pickle("fatjet_train.pkl")
        val_df_reshaped.to_pickle("fatjet_val.pkl")
    else:
        train_df_reshaped.to_pickle("fatjet_train_reshaped.pkl")
        val_df_reshaped.to_pickle("fatjet_val_reshaped.pkl")

    # df_selected.to_pickle("fatjet_oversampled.pkl")
