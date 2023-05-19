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
from post_actions import target_dictionary
from utils.corner_plots import make_corner
from trainer.utils.val_funcs import tagROC, profile_hist

from extractor.jets.columns import jet_cond, jet_names
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
            z, y = data[0], data[1]
            inputs_y = y.cuda(device)
            start = time.time()
            z_sampled = model.sample(
                num_samples=1, context=inputs_y.view(-1, args.y_dim)
            )
            t = time.time() - start
            print(f"Objects per second: {len(z_sampled) / t} [Hz]")
            times.append(t)

            z_sampled = z_sampled.cpu().detach().numpy()
            inputs_y = inputs_y.cpu().detach().numpy()
            z = z.cpu().detach().numpy()
            z_sampled = z_sampled.reshape(-1, args.zdim)
            gen.append(inputs_y)
            reco.append(z)
            samples.append(z_sampled)

    print(f"Average objs/sec: {np.mean(np.array(times))}")

    # Fix cols names to remove M at beginning
    reco_columns = ["Jet_" + x for x in jet_names]
    # Making DataFrames

    gen = np.array(gen).reshape((-1, args.y_dim))
    reco = np.array(reco).reshape((-1, args.zdim))
    samples = np.array(samples).reshape((-1, args.zdim))

    fullarray = np.concatenate((gen, reco, samples), axis=1)
    full_sim_cols = ["FullSJet_" + x for x in jet_names]
    full_df = pd.DataFrame(
        data=fullarray, columns=jet_cond + full_sim_cols + reco_columns
    )
    full_df.to_pickle(os.path.join(save_dir, "./full_jet_df.pkl"))

    # optionally you can read full_df from pickle and skip the previous steps
    # full_df = pd.read_pickle("./full_jet_df.pkl") # please provide column names
    gen = pd.DataFrame(data=full_df[jet_cond].values, columns=jet_cond)
    reco = pd.DataFrame(data=full_df[full_sim_cols].values, columns=reco_columns)
    samples = pd.DataFrame(data=full_df[reco_columns].values, columns=reco_columns)

    # Postprocessing
    # NOTE maybe add saturation here as done in nbd??
    reco = postprocessing(
        reco,
        target_dictionary,
        "scale_factors_jets.json",
        saturate_ranges_path="ranges_jets.json",
    )

    samples = postprocessing(
        samples,
        target_dictionary,
        "scale_factors_jets.json",
        saturate_ranges_path="ranges_jets.json",
    )

    # New DataFrame containing FullSim-range saturated samples

    saturated_samples = pd.DataFrame()

    # 1D FlashSim/FullSim comparison

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

    # Profile histograms
    columns = [
        ["Jet_pt", "GenJet_pt", [10, 1000]],
    ]

    n_bins = 30
    hep.style.use("CMS")

    # NOTE: to get back to the orginal mpl style, use: mpl.rcParams.update(mpl.rcParamsDefault)

    for elm in columns:
        x_slice_mean, x_slice_rms, xbinwn, xe = profile_hist(
            n_bins, reco[elm[0]], gen[elm[1]]
        )
        x_slice_mean_sat, x_slice_rms_sat, xbinwn_sat, xe_sat = profile_hist(
            n_bins, saturated_samples[elm[0]], gen[elm[1]]
        )
        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 6), tight_layout=True)

        hep.cms.text('Simulation Preliminary')

        ax1.errorbar(xe[:-1]+ xbinwn/2, x_slice_mean, x_slice_rms, marker='o', fmt='_',  label='FullSim')
        ax1.errorbar(xe_sat[:-1]+ xbinwn_sat/2, x_slice_mean_sat, x_slice_rms_sat, marker='o', fmt='_', label="FlashSim", color='tab:orange')
        ax1.set_xscale('log')
        ax1.set_ylabel(r"p$_T$Ratio", fontsize=18)
        ax1.set_xlabel(r"GenJet p$_T$ [GeV]", fontsize=18)
        ax1.set_xlim(elm[2])
        ax1.legend(fontsize=16, frameon=False)
        plt.savefig(os.path.join(save_dir, f"profhist_{elm[0]}.png"))
        plt.savefig(os.path.join(save_dir, f"profhist_{elm[0]}.pdf"))
        writer.add_figure(f"prof_hists/profhist_{elm[0]}", fig, global_step=epoch)

        fig, (ax2) = plt.subplots(1, 1, figsize=(10, 6), tight_layout=True)
        hep.cms.text('Simulation Preliminary')
        ax2.errorbar(xe[:-1]+ xbinwn/2, x_slice_rms, label='FullSim', ls='--')
        # ax2.errorbar(xef[:-1]+ xbinwf/2, x_slice_rmsf, color='tab:green', label='FastSim')
        ax2.errorbar(xe_sat[:-1]+ xbinwn_sat/2, x_slice_rms_sat, color='tab:orange', label='FlashSim')

        ax2.set_xscale('log')
        ax2.set_xlim([10,1000])
        ax2.set_xlabel(r"GenJet p$_T$ [GeV]", fontsize=18)
        ax2.set_ylabel("RMS", fontsize=18)
        ax2.legend(fontsize=16, frameon=False)
        plt.savefig(os.path.join(save_dir, f"profhistrms_{elm[0]}.png"))
        plt.savefig(os.path.join(save_dir, f"profhistrms_{elm[0]}.pdf"))
        writer.add_figure(f"prof_hists/profhistrms_{elm[0]}", fig, global_step=epoch)



    mpl.rcParams.update(mpl.rcParamsDefault)
    # Return to physical kinematic variables

    for df in [reco, samples, saturated_samples]:
        df["Jet_pt"] = df["Jet_ptRatio"] * gen["GenJet_pt"]
        df["Jet_eta"] = df["Jet_etaMinusGen"] + gen["GenJet_eta"]
        df["Jet_phi"] = df["Jet_phiMinusGen"] + gen["GenJet_phi"]
        df["Jet_mass"] = df["Jet_massRatio"] * gen["GenJet_mass"]

    # Zoom-in for high ws distributions

    incriminated = [
        ["Jet_pt", [0, 100]],
        ["Jet_eta", [-3, 3]],
        ["Jet_phi", [-3.14, 3.14]],
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

    labels = [
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

    # ROC
    hep.style.use("CMS")

    # NOTE: to get back to the orginal mpl style, use: mpl.rcParams.update(mpl.rcParamsDefault)
    taggers = [
        "btagDeepFlavB",
        "btagCSVV2",
        "btagDeepB",
    ]

    b_content = gen["GenJet_partonFlavour"].values

    for tagger in taggers:
        fpr, tpr, roc_auc, bs, nbs = tagROC(reco, b_content, tagger)
        fpr2, tpr2, roc_auc2, bs2, nbs2 = tagROC(samples, b_content, tagger)
        plt.figure()
        lw = 2

        plt.plot(
            tpr,
            fpr,
            lw=lw,
            label="ROC curve (area = %0.2f) FullSim" % croc_auc,
            ls="--",
        )
        plt.plot(
            tpr2,
            fpr2,
            lw=lw,
            label=f"ROC curve (area = %0.2f) FlashSim" % roc_auc,
        )

        plt.xlim([0.0, 1.0])
        plt.yscale("log")
        plt.ylim([0.0005, 1.05])

        plt.xlabel(f"Efficiency for {tagger} (TP)", fontsize=35)
        plt.ylabel("Mistagging prob (FP)", fontsize=35)
        hep.cms.text("Simulation Preliminary")
        plt.legend(fontsize=16, frameon=False, loc="best")

        plt.savefig(f"{save_dir}/{tagger}_roc.png", format="png")
        plt.savefig(f"{save_dir}/{tagger}_roc.pdf")
        writer.add_figure(f"rocs/{tagger}", fig, global_step=epoch)
        plt.close()
