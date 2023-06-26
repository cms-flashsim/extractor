import os
import sys
import json

import torch
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import corner

from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "postprocessing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "extractor"))

from postprocessing import postprocessing
from post_actions import target_dictionary_jets
from corner_plots import make_corner
from val_funcs import tagROC, profile_hist

from fake_jets.columns import jet_cond as jet_cond_M
from fake_jets.columns import jet_names
import mplhep as hep
import matplotlib as mpl


def validate(
    test_loader,
    model,
    epoch,
    writer,
    save_dir,
    args,
    device,
):
    if writer is not None:
        save_dir = os.path.join(save_dir, f"./figures/validation@epoch-{epoch}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None

    times = []
    model.eval()
    # Generate samples
    with torch.no_grad():
        gen = []
        reco = []
        samples = []

        for bid, data in enumerate(test_loader):
            x, y = data[0], data[1]
            inputs_y = y.cuda(device)
            start = time.time()
            x_sampled = model.sample(
                num_samples=1, context=inputs_y.view(-1, args.y_dim)
            )
            t = time.time() - start
            print(f"Objects per second: {len(x_sampled) / t} [Hz]")
            times.append(t)

            x_sampled = x_sampled.cpu().detach().numpy()
            inputs_y = inputs_y.cpu().detach().numpy()
            x = x.cpu().detach().numpy()
            x_sampled = x_sampled.reshape(-1, args.x_dim)
            gen.append(inputs_y)
            reco.append(x)
            samples.append(x_sampled)

    print(f"Average objs/sec: {len(x_sampled)/np.mean(np.array(times))}")

    # Fix cols names to remove M at beginning
    reco_columns = ["Jet_" + x for x in jet_names]
    jet_cond = [x[1:] for x in jet_cond_M]
    # Making DataFrames

    gen = np.array(gen).reshape((-1, args.y_dim))
    reco = np.array(reco).reshape((-1, args.x_dim))
    samples = np.array(samples).reshape((-1, args.x_dim))

    fullarray = np.concatenate((gen, reco, samples), axis=1)
    full_sim_cols = ["FullSJet_" + x for x in jet_names]
    full_df = pd.DataFrame(
        data=fullarray, columns=jet_cond + full_sim_cols + reco_columns
    )
    full_df.to_pickle(os.path.join(save_dir, "./full_fakejet_df.pkl"))

    # optionally you can read full_df from pickle and skip the previous steps
    # full_df = pd.read_pickle("./full_jet_df.pkl") # please provide column names (full_sim_cols = ["FullSJet_" + x for x in jet_names])
    gen = pd.DataFrame(data=full_df[jet_cond].values, columns=jet_cond)
    reco = pd.DataFrame(data=full_df[full_sim_cols].values, columns=reco_columns)
    samples = pd.DataFrame(data=full_df[reco_columns].values, columns=reco_columns)

    # Postprocessing
    # NOTE maybe add saturation here as done in nbd??
    reco = postprocessing(
        reco,
        gen,
        target_dictionary_jets,
        "scale_factors_fakejets.json",
        saturate_ranges_path="ranges_fakejets.json",
    )

    samples = postprocessing(
        samples,
        gen,
        target_dictionary_jets,
        "scale_factors_fakejets.json",
        saturate_ranges_path="ranges_fakejets.json",
    )

    # New DataFrame containing FullSim-range saturated samples

    saturated_samples = pd.DataFrame()

    # 1D FlashSim/FullSim comparison
    mpl.rcParams.update(mpl.rcParamsDefault)
    for column in reco_columns:
        ws = wasserstein_distance(reco[column], samples[column])

        fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)
        fig.suptitle(f"{column} comparison")

        # RECO histogram
        _, rangeR, _ = axs[0].hist(
            reco[column], histtype="step", lw=1, bins=100, label="FullSim"
        )

        # Saturation based on FullSim range
        saturated_samples[column] = np.where(
            samples[column] < np.min(rangeR), np.min(rangeR), samples[column]
        )
        saturated_samples[column] = np.where(
            saturated_samples[column] > np.max(rangeR),
            np.max(rangeR),
            saturated_samples[column],
        )

        # Samples histogram
        axs[0].hist(
            saturated_samples[column],
            histtype="step",
            lw=1,
            range=[np.min(rangeR), np.max(rangeR)],
            bins=100,
            label=f"FlashSim, ws={round(ws, 4)}",
        )

        axs[0].legend(frameon=False, loc="upper right")

        # Log-scale comparison

        axs[1].set_yscale("log")
        axs[1].hist(reco[column], histtype="step", lw=1, bins=100)
        axs[1].hist(
            saturated_samples[column],
            histtype="step",
            lw=1,
            range=[np.min(rangeR), np.max(rangeR)],
            bins=100,
        )
        plt.savefig(os.path.join(save_dir, f"{column}.png"))
        plt.savefig(os.path.join(save_dir, f"{column}.pdf"))
        writer.add_figure(f"1D_Distributions/{column}", fig, global_step=epoch)
        writer.add_scalar(f"ws/{column}_wasserstein_distance", ws, global_step=epoch)
        plt.close()


    mpl.rcParams.update(mpl.rcParamsDefault)
    # Return to physical kinematic variables

    # Zoom-in for high ws distributions

    incriminated = [
        ["Jet_mass", [0, 100]],
    ]
    for elm in incriminated:
        column = elm[0]
        rangeR = elm[1]
        inf = rangeR[0]
        sup = rangeR[1]

        full = reco[column].values
        full = np.where(full > sup, sup, full)
        full = np.where(full < inf, inf, full)

        flash = samples[column].values
        flash = np.where(flash > sup, sup, flash)
        flash = np.where(flash < inf, inf, flash)

        ws = wasserstein_distance(full, flash)

        fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)
        fig.suptitle(f"{column} comparison")

        axs[0].hist(
            full, histtype="step", lw=1, bins=100, range=rangeR, label="FullSim"
        )
        axs[0].hist(
            flash,
            histtype="step",
            lw=1,
            range=rangeR,
            bins=100,
            label=f"FlashSim, ws={round(ws, 4)}",
        )

        axs[0].legend(frameon=False, loc="upper right")

        axs[1].set_yscale("log")
        axs[1].hist(full, histtype="step", range=rangeR, lw=1, bins=100)
        axs[1].hist(
            flash,
            histtype="step",
            lw=1,
            range=rangeR,
            bins=100,
        )
        plt.savefig(f"{save_dir}/{column}_incriminated.png", format="png")
        plt.savefig(f"{save_dir}/{column}_incriminated.pdf")
        writer.add_figure(f"Zoom_in_1D_Distributions/{column}", fig, global_step=epoch)
        plt.close()

    # Corner plots:

    # taggers
    # first, add the pt column from gen to reco and samples
    reco["Jet_pt"] = gen["Jet_pt"]
    samples["Jet_pt"] = gen["Jet_pt"]

    labels = [
        "pt",
        "btagCSVV2",
        "btagDeepB",
        "btagDeepCvB",
        "btagDeepCvL",
        "btagDeepFlavB",
        "btagDeepFlavCvB",
        "btagDeepFlavCvL",
        "btagDeepFlavQG",
    ]
    labels = [f"Jet_{label}" for label in labels]

    ranges = [
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
    ]

    fig = make_corner(reco, saturated_samples, labels, "Taggers", ranges=ranges)
    plt.savefig(f"{save_dir}/Taggers_corner.png", format="png")
    plt.savefig(f"{save_dir}/Taggers_corner.pdf")
    writer.add_figure("Corner_plots/Taggers", fig, global_step=epoch)

    # # ROC
    # hep.style.use("CMS")

    # # NOTE: to get back to the orginal mpl style, use: mpl.rcParams.update(mpl.rcParamsDefault)
    # taggers = [
    #     "btagDeepFlavB",
    #     "btagCSVV2",
    #     "btagDeepB",
    # ]
    # taggers = [f"Jet_{tagger}" for tagger in taggers]

    # b_mask = gen["GenJet_EncodedPartonFlavour_b"].values
    # uds_mask = gen["GenJet_EncodedPartonFlavour_light"].values

    # for tagger in taggers:
    #     fpr, tpr, roc_auc, bs, nbs = tagROC(reco, b_mask, uds_mask, tagger)
    #     fpr2, tpr2, roc_auc2, bs2, nbs2 = tagROC(samples, b_mask, uds_mask, tagger)
    #     fig = plt.figure()
    #     lw = 2

    #     plt.plot(
    #         tpr,
    #         fpr,
    #         lw=lw,
    #         label="ROC curve (area = %0.2f) FullSim" % roc_auc,
    #         ls="--",
    #     )
    #     plt.plot(
    #         tpr2,
    #         fpr2,
    #         lw=lw,
    #         label=f"ROC curve (area = %0.2f) FlashSim" % roc_auc2,
    #     )

    #     plt.xlim([0.0, 1.0])
    #     plt.yscale("log")
    #     plt.ylim([0.0005, 1.05])

    #     plt.xlabel(f"Efficiency for {tagger} (TP)", fontsize=35)
    #     plt.ylabel("Mistagging prob (FP)", fontsize=35)
    #     hep.cms.text("Simulation Preliminary")
    #     plt.legend(fontsize=16, frameon=False, loc="best")

    #     plt.savefig(f"{save_dir}/{tagger}_roc.png", format="png")
    #     plt.savefig(f"{save_dir}/{tagger}_roc.pdf")
    #     writer.add_figure(f"rocs/{tagger}", fig, global_step=epoch)
    #     plt.close()
