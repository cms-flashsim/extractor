import numpy as np
import torch


def nan_resampling(sample, gen, model, device):
    sample = torch.tensor(sample).to(device)
    gen = torch.tensor(gen).to(device)
    nan_mask = torch.isnan(sample).any(axis=1)
    if nan_mask.any():
        nan_idx = torch.argwhere(nan_mask)
        # Generate new samples
        model.eval()
        while True:
            with torch.no_grad():
                sample[nan_idx] = model.sample(num_samples=1, context=gen[nan_mask])
                if not torch.isnan(sample[nan_idx]).any():
                    break
    sample = sample.detach().cpu().numpy()
    gen = gen.detach().cpu().numpy()
    return sample
