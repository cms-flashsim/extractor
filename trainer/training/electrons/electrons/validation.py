import os
import sys
import json

import torch
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import corner
import mplhep

from scipy.stats import wasserstein_distance

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "postprocessing")
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "utils"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "extractor")
)

from postprocessing import postprocessing
from post_actions_ele import target_dictionary
from corner_plots import make_corner

from nan_resampling import nan_resampling

from electrons.columns import gen_ele as gen_eleM
from electrons.columns import reco_columns as reco_columnsM


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
    # Making DataFrames

    reco_columns = [col.replace("M", "", 1) for col in reco_columnsM]
    gen_ele = [
        col.replace("M", "", 1) if col.startswith("M") else col for col in gen_eleM
    ]

    gen = np.array(gen).reshape((-1, args.y_dim))
    reco = np.array(reco).reshape((-1, args.x_dim))
    samples = np.array(samples).reshape((-1, args.x_dim))

    if np.isnan(samples).any():
        print("RESAMPLING")

    samples = nan_resampling(samples, gen, model, device)

    fullarray = np.concatenate((gen, reco, samples), axis=1)
    full_sim_cols = ["Full_" + x for x in reco_columns]
    full_df = pd.DataFrame(
        data=fullarray, columns=gen_ele + full_sim_cols + reco_columns
    )
    full_df.to_pickle(os.path.join(save_dir, "./full_ele_ele_df.pkl"))

    gen = pd.DataFrame(data=full_df[gen_ele].values, columns=gen_ele)
    reco = pd.DataFrame(data=full_df[full_sim_cols].values, columns=reco_columns)
    samples = pd.DataFrame(data=full_df[reco_columns].values, columns=reco_columns)

    # Postprocessing

    reco = postprocessing(
        reco,
        gen,
        target_dictionary,
        "scale_factors_ele_ele.json",
        saturate_ranges_path="ranges_ele_ele.json",
    )

    samples = postprocessing(
        samples,
        gen,
        target_dictionary,
        "scale_factors_ele_ele.json",
        saturate_ranges_path="ranges_ele_ele.json",
    )

    # New DataFrame containing FullSim-range saturated samples

    saturated_samples = pd.DataFrame()

    range_dict = {}

    for column in reco_columns:
        fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)
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
        range_dict[column] = [float(np.min(rangeR)), float(np.max(rangeR))]
        plt.close()

    for df in [reco, samples, saturated_samples]:
        df["Electron_pt"] = df["Electron_ptRatio"] * gen["GenElectron_pt"]
        df["Electron_eta"] = df["Electron_etaMinusGen"] + gen["GenElectron_eta"]
        df["Electron_phi"] = df["Electron_phiMinusGen"] + gen["GenElectron_phi"]

    # 1D FlashSim/FullSim comparison

    cols = [
        "Electron_pt",
        "Electron_eta",
        "Electron_phi",
        "Electron_sip3d",
        "Electron_charge",
    ]
    xaxis = [r"$p_T$ [GeV]", r"$\eta$", r"$\phi$", "Significance IP", "Charge"]
    logscale = ["Electron_sip3d"]

    for column in cols:
        mplhep.style.use("CMS")
        fig, axs = plt.subplots(
            2, 1, figsize=(12, 12), tight_layout=False, height_ratios=[3, 1]
        )
        mplhep.cms.text("Simulation Preliminary")

        # RECO histogram
        ns0, rangeR, _ = axs[0].hist(
            reco[column], histtype="step", lw=2, ls="--", bins=100, label="FullSim"
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
        ws = wasserstein_distance(reco[column], saturated_samples[column])

        range_dict[column] = [float(np.min(rangeR)), float(np.max(rangeR))]

        # Samples histogram
        ns1, bins1, _ = axs[0].hist(
            saturated_samples[column],
            histtype="step",
            lw=2,
            range=[np.min(rangeR), np.max(rangeR)],
            bins=100,
            label=f"FlashSim, ws={round(ws, 4)}",
        )
        axs[0].set_xscale("log" if column in logscale else "linear")
        axs[0].legend(frameon=False, loc="upper right")

        # FlashSim/FullSim ratio
        axs[1].plot(bins1[:-1], ns1 / ns0, marker="o", ls="none", color="black")
        axs[1].set_ylim(-2, 2)
        axs[1].set_xlim(np.min(rangeR), np.max(rangeR))
        axs[1].axhline(1, ls="--", color="gray")

        axs[0].set_ylabel("Entries")
        axs[1].set_ylabel("FlashSim/FullSim")
        axs[1].set_xlabel(xaxis[cols.index(column)])

        plt.savefig(os.path.join(save_dir, f"{column}.png"))
        plt.savefig(os.path.join(save_dir, f"{column}.pdf"))
        plt.close()

    # Conditioning

    targets = ["Electron_ip3d", "Electron_pfRelIso03_all"]
    xaxis = ["Impact parameter [cm]", r"Relative PF isolation $\Delta R < 0.3$"]

    ranges = [[0, 0.1], [0, 10], [0, 0.5]]

    conds = [f"GenElectron_statusFlag{i}" for i in (0, 2, 7)]
    conds.append("ClosestJet_EncodedPartonFlavour_b")

    names = [
        "isPrompt",
        "isTauDecayProduct",
        "isHardProcess",
        "ClosestJet partonFlavour is b",
    ]

    colors = ["tab:red", "tab:green", "tab:blue", "tab:orange"]

    for target, rangeR in zip(targets, ranges):
        mplhep.style.use("CMS")
        fig, axs = plt.subplots(1, 1, figsize=(12, 12), tight_layout=False)
        mplhep.cms.text("Private Work")

        axs.set_xlabel(xaxis[targets.index(target)])

        axs.set_yscale("log")

        inf = rangeR[0]
        sup = rangeR[1]

        labels = []

        for cond, color, name in zip(conds, colors, names):
            mask = gen[cond].values.astype(bool)
            full = reco[target].values
            full = full[mask]
            full = full[~np.isnan(full)]
            full = np.where(full > sup, sup, full)
            full = np.where(full < inf, inf, full)

            flash = samples[target].values
            flash = flash[mask]
            flash = flash[~np.isnan(flash)]
            flash = np.where(flash > sup, sup, flash)
            flash = np.where(flash < inf, inf, flash)

            axs.hist(
                full,
                bins=50,
                range=rangeR,
                histtype="step",
                ls="--",
                lw=2,
                color=color,
                density=False,
            )
            axs.hist(
                flash,
                bins=50,
                range=rangeR,
                histtype="step",
                lw=2,
                color=color,
                density=False,
            )

            labels.append(Patch(edgecolor=color, fill=False, lw=2, label=f"{name}"))

            del full, flash

        mask = (
            gen["ClosestJet_EncodedPartonFlavour_gluon"].values
            + gen["ClosestJet_EncodedPartonFlavour_light"].values
        ).astype(bool)
        full = reco[target].values
        full = full[mask]
        full = full[~np.isnan(full)]
        full = np.where(full > sup, sup, full)
        full = np.where(full < inf, inf, full)

        flash = samples[target].values
        flash = flash[mask]
        flash = flash[~np.isnan(flash)]
        flash = np.where(flash > sup, sup, flash)
        flash = np.where(flash < inf, inf, flash)

        axs.hist(
            full,
            bins=50,
            range=rangeR,
            histtype="step",
            ls="--",
            lw=2,
            color="tab:purple",
            density=False,
        )
        axs.hist(
            flash,
            bins=50,
            range=rangeR,
            histtype="step",
            lw=2,
            color="tab:purple",
            density=False,
        )

        labels.append(
            Patch(
                edgecolor="tab:purple",
                fill=False,
                lw=2,
                label="ClosestJet partonFlavour is udsg",
            )
        )

        del full, flash

        labels.append(
            Patch(edgecolor="black", fill=False, lw=2, ls="--", label="FullSim")
        )
        labels.append(Patch(edgecolor="black", fill=False, lw=2, label="FlashSim"))

        axs.legend(handles=labels, frameon=False, loc="upper center")
        plt.savefig(
            f"{save_dir}/{target}_conditioning_normalized_cms.png", format="png"
        )
        plt.savefig(
            f"{save_dir}/{target}_conditioning_normalized_cms.pdf", format="pdf"
        )
        # writer.add_figure(
        #     f"Conditioning/{target}_conditioning_normalized.png", fig, global_step=epoch
        # )
        plt.close()

    labels = [
        "Electron_pt",
        "Electron_deltaEtaSC",
        "Electron_hoe",
        "Electron_sieie",
        "Electron_r9",
        "Electron_eInvMinusPInv",
    ]

    names = [
        r"$p_T$ [GeV]",
        r"$\Detlta\eta_{SC}$",
        r"$H/E$",
        r"$\sigma_{i\eta i\eta}$",
        r"$R_9$",
        r"$|1/E_{SC} - 1/p_{trk}|$ [GeV$^{-1}$]",
    ]

    ranges = [
        (0, 200),
        (-0.1, 0.1),
        (0, 0.16),
        (0, 0.05),
        (0.3, 1.2),
        (-0.1, 0.1),
    ]
    blue_line = mlines.Line2D([], [], color="tab:blue", ls="--", label="FullSim")
    red_line = mlines.Line2D([], [], color="tab:orange", label="FlashSim")

    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure()

    fig = corner.corner(
        reco[labels],
        range=ranges,
        labels=names,
        color="tab:blue",
        levels=[0.5, 0.9, 0.99],
        hist_bin_factor=3,
        scale_hist=True,
        plot_datapoints=False,
        hist_kwargs={"ls": "--"},
        contour_kwargs={"linestyles": "--"},
        label_kwargs={"fontsize": 16},
    )
    corner.corner(
        samples[labels],
        range=ranges,
        labels=names,
        fig=fig,
        color="tab:orange",
        levels=[0.5, 0.9, 0.99],
        hist_bin_factor=3,
        scale_hist=True,
        plot_datapoints=False,
        label_kwargs={"fontsize": 16},
    )
    plt.legend(
        fontsize=24,
        frameon=False,
        handles=[blue_line, red_line],
        bbox_to_anchor=(0.0, 1.0, 1.0, 4.0),
        loc="upper right",
    )

    plt.suptitle(
        r"$\bf{CMS}$ $\it{Private \; Work}$",
        fontsize=16,
        x=0.29,
        y=1.0,
        horizontalalignment="right",
        fontname="sans-serif",
    )
    plt.savefig(f"{save_dir}/Supercluster_corner_cms.png", format="png")
    plt.savefig(f"{save_dir}/Supercluster_corner_cms.pdf", format="pdf")
