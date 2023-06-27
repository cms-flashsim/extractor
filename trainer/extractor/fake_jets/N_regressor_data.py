import pandas as pd
import numpy as np
import uproot
import awkward as ak
import h5py
import sys

STOP = None


def get_df(file_num):
    
    tree = uproot.open(f"/home/users/fvaselli/trainer/trainer/extractor/fake_jets/dataset/MFakeJets_{file_num}.root:MFakeJets", num_workers=20)
    # define pandas df for fast manipulation
    dfgl = tree.arrays(
        [
            "Pileup_nTrueInt",
        ],
        library="pd",
        entry_stop=STOP,
    ).astype("float32")
    print(dfgl)

    # define pandas df for fast manipulation
    dfft = tree.arrays(
        ["MJet_pt"],
        library="pd",
        entry_stop=STOP,
    ).astype("float32")

    num_fakes = dfft.reset_index(level=1).index.value_counts(sort=False).reindex(np.arange(len(dfgl)), fill_value=0).values
    # limit num fakes to 10
    num_fakes[num_fakes>10] = 10


    df = pd.concat([dfgl, pd.DataFrame(num_fakes, columns=['num_fakes'])], axis=1)
    df["num_fakes"] = df["num_fakes"].apply(
        lambda x: x + 0.5 * np.random.uniform(-1, 1)
    )
    print(df)
    return df

if __name__ == "__main__":

    df = get_df(0)
    df1 = get_df(1)

    df = pd.concat([df, df1], axis=0)
    df = df.reset_index(drop=True)
    print(df)

    save_file = h5py.File(f"../../training/datasets/N_regressor_data.hdf5", "w")

    dset = save_file.create_dataset("data", data=df.values, dtype="f4")

    save_file.close()