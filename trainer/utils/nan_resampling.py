import numpy as np
import torch


def nan_resampling(sample, gen, model, device):
    sample = torch.tensor(sample)
    gen = torch.tensor(gen)
    nan_mask = torch.isnan(sample).any(axis=1)
    if nan_mask.any():
        nan_idx = torch.argwhere(nan_mask)
        # Generate new samples
        model.eval()
        model.to("cpu")
        while True:
            with torch.no_grad():
                sample[nan_idx] = model.sample(num_samples=1, context=gen[nan_mask])
                if not torch.isnan(sample[nan_idx]).any():
                    break
    model.to(device)
    sample = sample.numpy()
    return sample
