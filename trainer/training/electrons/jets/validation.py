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

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "postprocessing")
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "utils"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "extractor")
)

from postprocessing import postprocessing
from post_actions_jet import target_dictionary
from corner_plots import make_corner

from nan_resampling import nan_resampling

from electrons.columns import gen_jet as gen_jetM
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
    print(reco_columns)
    gen_jet = [
        col.replace("M", "", 1) if col.startswith("M") else col for col in gen_jetM
    ]

    gen = np.array(gen).reshape((-1, args.y_dim))
    reco = np.array(reco).reshape((-1, args.x_dim))
    samples = np.array(samples).reshape((-1, args.x_dim))
    print(gen.shape, samples.shape)

    if np.isnan(samples).any():
        print("RESAMPLING")

    samples = nan_resampling(samples, gen, model, device)
    
    fullarray = np.concatenate((gen, reco, samples), axis=1)
    full_sim_cols = ["Full_" + x for x in reco_columns]
    full_df = pd.DataFrame(
        data=fullarray, columns=gen_jet + full_sim_cols + reco_columns
    )
    full_df.to_pickle(os.path.join(save_dir, "./full_ele_jet_df.pkl"))

    gen = pd.DataFrame(data=full_df[gen_jet].values, columns=gen_jet)
    reco = pd.DataFrame(data=full_df[full_sim_cols].values, columns=reco_columns)
    print(reco.columns)
    samples = pd.DataFrame(data=full_df[reco_columns].values, columns=reco_columns)

    # Postprocessing

    reco = postprocessing(
        reco,
        gen,
        target_dictionary,
        "scale_factors_ele_jet.json",
        saturate_ranges_path="ranges_ele_jet.json",
    )

    samples = postprocessing(
        samples,
        gen,
        target_dictionary,
        "scale_factors_ele_jet.json",
        saturate_ranges_path="ranges_ele_jet.json",
    )

    # New DataFrame containing FullSim-range saturated samples

    saturated_samples = pd.DataFrame()

    range_dict = {}

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

        range_dict[column] = [float(np.min(rangeR)), float(np.max(rangeR))]

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
        writer.add_figure(f"1D_Distributions/{column}", fig, global_step=epoch)
        writer.add_scalar(f"ws/{column}_wasserstein_distance", ws, global_step=epoch)
        plt.close()

    # with open(os.path.join(os.path.dirname(__file__), "range_dict.json"), "w") as f:
    #     json.dump(range_dict, f)

    # Return to physical kinematic variables

    for df in [reco, samples, saturated_samples]:
        df["Electron_pt"] = df["Electron_ptRatio"] * gen["GenJet_pt"]
        df["Electron_eta"] = df["Electron_etaMinusGen"] + gen["GenJet_eta"]
        df["Electron_phi"] = df["Electron_phiMinusGen"] + gen["GenJet_phi"]

    # Zoom-in for high ws distributions

    incriminated = [
        ["Electron_dr03HcalDepth1TowerSumEt", [0, 10]],
        ["Electron_dr03TkSumPt", [0, 10]],
        ["Electron_dr03TkSumPtHEEP", [0, 10]],
        ["Electron_dr03EcalRecHitSumEt", [0, 10]],
        ["Electron_dxyErr", [0, 0.1]],
        ["Electron_dzErr", [0, 0.2]],
        ["Electron_energyErr", [0, 5]],
        ["Electron_hoe", [0, 0.4]],
        ["Electron_ip3d", [0, 0.1]],
        ["Electron_jetPtRelv2", [0, 10]],
        ["Electron_jetRelIso", [0, 2]],
        ["Electron_miniPFRelIso_all", [0, 1]],
        ["Electron_miniPFRelIso_chg", [0, 1]],
        ["Electron_pfRelIso03_all", [0, 0.5]],
        ["Electron_pfRelIso03_chg", [0, 0.5]],
        ["Electron_sieie", [0.005, 0.02]],
        ["Electron_sip3d", [0, 10]],
        ["Electron_pt", [0, 100]],
        ["Electron_eta", [-3, 3]],
        ["Electron_phi", [-3.14, 3.14]],
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
        # plt.savefig(f"{save_dir}/{column}_incriminated.png", format="png")
        writer.add_figure(f"Zoom_in_1D_Distributions/{column}", fig, global_step=epoch)
        plt.close()

    # Return to physical kinematic variables

    physical = ["Electron_pt", "Electron_eta", "Electron_phi"]

    for column in physical:
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
        writer.add_figure(f"1D_Distributions/{column}", fig, global_step=epoch)
        writer.add_scalar(f"ws/{column}_wasserstein_distance", ws, global_step=epoch)
        plt.close()

    # Conditioning

    targets = ["Electron_ip3d", "Electron_sip3d", "Electron_pfRelIso03_all"]

    ranges = [[0, 0.1], [0, 10], [0, 5]]

    conds = ["GenJet_EncodedPartonFlavour_b", "GenJet_EncodedPartonFlavour_c"]

    names = conds

    colors = ["tab:red", "tab:green"]

    for target, rangeR in zip(targets, ranges):
        fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

        axs[0].set_xlabel(f"{target}")
        axs[1].set_xlabel(f"{target}")

        axs[1].set_yscale("log")

        inf = rangeR[0]
        sup = rangeR[1]

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

            axs[0].hist(
                full, bins=50, range=rangeR, histtype="step", ls="--", color=color
            )
            axs[0].hist(
                flash,
                bins=50,
                range=rangeR,
                histtype="step",
                label=f"{name}",
                color=color,
            )

            axs[1].hist(
                full, bins=50, range=rangeR, histtype="step", ls="--", color=color
            )
            axs[1].hist(
                flash,
                bins=50,
                range=rangeR,
                histtype="step",
                label=f"{name}",
                color=color,
            )

            del full, flash

        mask = (
            gen["GenJet_EncodedPartonFlavour_gluon"].values
            + gen["GenJet_EncodedPartonFlavour_light"].values
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

        axs[0].hist(
            full, bins=50, range=rangeR, histtype="step", ls="--", color="tab:purple"
        )
        axs[0].hist(
            flash,
            bins=50,
            range=rangeR,
            histtype="step",
            label="GenJet partonFlavour is udsg",
            color="tab:purple",
        )

        axs[1].hist(
            full, bins=50, range=rangeR, histtype="step", ls="--", color="tab:purple"
        )
        axs[1].hist(
            flash,
            bins=50,
            range=rangeR,
            histtype="step",
            label="GenJet partonFlavour is udsg",
            color="tab:purple",
        )
        del full, flash

        axs[0].legend(frameon=False, loc="upper right")
        # plt.savefig(f"{save_dir}/{target}_conditioning.png", format="png")
        writer.add_figure(
            f"Conditioning/{target}_conditioning.png", fig, global_step=epoch
        )
        plt.close()

    # Normalized version

    for target, rangeR in zip(targets, ranges):
        fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

        axs[0].set_xlabel(f"{target}")
        axs[1].set_xlabel(f"{target}")

        axs[1].set_yscale("log")

        inf = rangeR[0]
        sup = rangeR[1]

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

            axs[0].hist(
                full,
                bins=50,
                range=rangeR,
                histtype="step",
                ls="--",
                color=color,
                density=True,
            )
            axs[0].hist(
                flash,
                bins=50,
                range=rangeR,
                histtype="step",
                label=f"{name}",
                color=color,
                density=True,
            )

            axs[1].hist(
                full,
                bins=50,
                range=rangeR,
                histtype="step",
                ls="--",
                color=color,
                density=True,
            )
            axs[1].hist(
                flash,
                bins=50,
                range=rangeR,
                histtype="step",
                label=f"{name}",
                color=color,
                density=True,
            )

            del full, flash

        mask = (
            gen["GenJet_EncodedPartonFlavour_gluon"].values
            + gen["GenJet_EncodedPartonFlavour_light"].values
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

        axs[0].hist(
            full,
            bins=50,
            range=rangeR,
            histtype="step",
            ls="--",
            color="tab:purple",
            density=True,
        )
        axs[0].hist(
            flash,
            bins=50,
            range=rangeR,
            histtype="step",
            label="GenJet partonFlavour is udsg",
            color="tab:purple",
            density=True,
        )

        axs[1].hist(
            full,
            bins=50,
            range=rangeR,
            histtype="step",
            ls="--",
            color="tab:purple",
            density=True,
        )
        axs[1].hist(
            flash,
            bins=50,
            range=rangeR,
            histtype="step",
            label="GenJet partonFlavour is udsg",
            color="tab:purple",
            density=True,
        )
        del full, flash

        axs[0].legend(frameon=False, loc="upper right")
        # plt.savefig(f"{save_dir}/{target}_conditioning_normalized.png", format="png")
        writer.add_figure(
            f"Conditioning/{target}_conditioning_normalized.png", fig, global_step=epoch
        )
        plt.close()

    # Corner plots:

    # Isolation

    labels = [
        "Electron_pt",
        "Electron_eta",
        "Electron_jetRelIso",
        "Electron_miniPFRelIso_all",
        "Electron_miniPFRelIso_chg",
        # "Electron_mvaFall17V1Iso",
        # "Electron_mvaFall17V1noIso",
        "Electron_mvaFall17V2Iso",
        "Electron_mvaFall17V2noIso",
        "Electron_pfRelIso03_all",
        "Electron_pfRelIso03_chg",
    ]

    ranges = [
        (0, 200),
        (-2, 2),
        (0, 0.5),
        (0, 0.5),
        (0, 0.5),
        # (-1, 1),
        # (-1, 1),
        (-1, 1),
        (-1, 1),
        (0, 0.5),
        (0, 0.5),
    ]

    fig = make_corner(reco, saturated_samples, labels, "Isolation", ranges=ranges)
    writer.add_figure("Corner_plots/Isolation", fig, global_step=epoch)

    # Impact parameter (range)

    labels = [
        "Electron_pt",
        "Electron_eta",
        "Electron_ip3d",
        "Electron_sip3d",
        "Electron_dxy",
        "Electron_dxyErr",
        "Electron_dz",
        "Electron_dzErr",
    ]

    ranges = [
        (0, 200),
        (-2, 2),
        (0, 0.2),
        (0, 5),
        (-0.2, 0.2),
        (0, 0.05),
        (-0.2, 0.2),
        (0, 0.05),
    ]

    fig = make_corner(
        reco, saturated_samples, labels, "Impact parameter", ranges=ranges
    )
    writer.add_figure("Corner_plots/Impact parameter", fig, global_step=epoch)

    # Impact parameter comparison

    reco["Electron_sqrt_xy_z"] = np.sqrt(
        (reco["Electron_dxy"].values) ** 2 + (reco["Electron_dz"].values) ** 2
    )
    saturated_samples["Electron_sqrt_xy_z"] = np.sqrt(
        (saturated_samples["Electron_dxy"].values) ** 2
        + (saturated_samples["Electron_dz"].values) ** 2
    )

    labels = ["Electron_sqrt_xy_z", "Electron_ip3d"]

    ranges = [(0, 0.2), (0, 0.2)]

    fig = make_corner(
        reco,
        saturated_samples,
        labels,
        r"Impact parameter vs $\sqrt{dxy^2 + dz^2}$",
        ranges=ranges,
    )
    writer.add_figure(
        r"Corner_plots/Impact parameter vs \sqrt(dxy^2 + dz^2)", fig, global_step=epoch
    )

    # Kinematics

    labels = ["Electron_pt", "Electron_eta", "Electron_phi"]

    fig = make_corner(reco, saturated_samples, labels, "Kinematics")
    writer.add_figure("Corner_plots/Kinematics", fig, global_step=epoch)

    # Supercluster

    labels = [
        "Electron_pt",
        "Electron_eta",
        "Electron_sieie",
        "Electron_r9",
        # "Electron_mvaFall17V1Iso",
        # "Electron_mvaFall17V1noIso",
        "Electron_mvaFall17V2Iso",
        "Electron_mvaFall17V2noIso",
    ]

    ranges = [
        (0, 200),
        (-2, 2),
        (0, 0.09),
        (0, 1.5),
        # (-1, 1),
        # (-1, 1),
        (-1, 1),
        (-1, 1),
    ]

    fig = make_corner(reco, saturated_samples, labels, "Supercluster", ranges=ranges)
    writer.add_figure("Corner_plots/Supercluster", fig, global_step=epoch)
