# trains the regressor model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "utils"))
from dataset import ObjectDataset
from N_regressor_model import N_regressor

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = N_regressor(input_size=1, hidden_size=64, output_size=1)
    model = model.to(device)

    lr = 1e-2
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    filepath = '/home/users/fvaselli/trainer/trainer/extractor/fake_jets/dataset/N_regressor_data.hdf5'
    tr_dataset = ObjectDataset(
        [filepath],
        x_dim=1,
        y_dim=1,
        start=0,
        limit=1500000,
    )
    te_dataset = ObjectDataset(
        [filepath],
        x_dim=1,
        y_dim=1,
        start=1500000,
        limit=500000,
    )


    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset,
        batch_size=10000,
        num_workers=20,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset,
        batch_size=10000,  # manually set batch size to avoid diff shapes
        shuffle=False,
        num_workers=20,
        pin_memory=True,
        drop_last=True,
    )

    # main training loop
    start_time = time.time()
    train_history = []
    test_history = []
    for epoch in range(1000):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = F.mse_loss(output, target)
            train_history.append(loss.item())
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

        test_loss = 0
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.mse_loss(output, target, reduction="sum").item()

        test_loss /= len(test_loader.dataset)
        test_history.append(test_loss)
        print(
            "\nTest set: Average loss: {:.4f}\n".format(
                test_loss,
            )
        )
        # if epoch % 10 == 0:
        #     torch.save(model.state_dict(), f"./models/N_regressor_{epoch}.pt")
        #     print("Saved model")

    print("Total time: {}".format(time.time() - start_time))
    # plot loss history
    plt.plot(train_history, label="train")
    plt.plot(test_history, label="test")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_history.png")
    plt.close()
