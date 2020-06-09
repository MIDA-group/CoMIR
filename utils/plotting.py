import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.image import *

def plot_losses(data, steps, steps_per_epoch, awidth=20, **kwargs):
    """Plots the losses from a dictionary containing the keys ["train", "test"]
    """
    def moving_average(x, steps=20):
        cs = np.cumsum(x)
        return (cs[steps:] - cs[:-steps]) / steps

    print(
        f'Train: {data["train"][-1]:.3f} '
        f'(min: {np.min(data["train"]):.3f}, max: {np.max(data["train"]):.3f})'
    )
    plt.figure(figsize=(15, 4), **kwargs)
    plt.plot(data["train"], label="Training loss", alpha=0.3)
    if len(data["train"]) > awidth:  
        plt.gca().set_prop_cycle(None)
        t = np.arange(awidth//2, len(data["train"])-awidth//2)
        plt.plot(t, moving_average(data["train"], awidth), label=f"Training loss (Avg/{awidth})")
    t = np.arange(0, steps+1, steps_per_epoch)[:len(data["test"])]
    plt.plot(t, data["test"], "o-", label="Testing loss")
    plt.title("Loss function over time")
    plt.grid(True)
    plt.xlim(0, steps)
    plt.xticks(np.arange(0, steps+1, steps_per_epoch))
    plt.xlabel("Iteration")
    plt.ylim(0, data["train"][0])
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

def plot_similarity_matrix(similarities, title="Similarity matrix"):
    """Plots the similarity matrix and adds annotations for better visual understanding.
    
    Note:
        * The diagonal is excluded from the calculation of the min/max for plotting.
        * We assume two modalities (M=2)
    """
    sim_nodiag = similarities[~torch.eye(similarities.shape[0], dtype=torch.bool)]
    vmin, vmax = sim_nodiag.min(), sim_nodiag.max()

    plt.title(title)
    plt.imshow(similarities.cpu().detach(), vmin=vmin, vmax=vmax, cmap="bwr")

    N = similarities.shape[0] // 2
    plt.xticks([0, N, 2*N-1], [0, N, 2*N-1])
    plt.yticks([0, N, 2*N-1], [0, N, 2*N-1])
    ax = plt.gca()
    ax.annotate("Latent A", xy=(0, 0), xytext=(N/2, 2*N+1), xycoords="data", textcoords="data",
                horizontalalignment="center", verticalalignment="center")
    ax.annotate("Latent B", xy=(0, 0), xytext=(N+N/2, 2*N+1), xycoords="data", textcoords="data",
                horizontalalignment="center", verticalalignment="center")
    ax.annotate("Latent A", xy=(0, 0), xytext=(-2, N/2), xycoords="data", textcoords="data",
                horizontalalignment="center", verticalalignment="center", rotation=90)
    ax.annotate("Latent B", xy=(0, 0), xytext=(-2, N+N/2), xycoords="data", textcoords="data",
                horizontalalignment="center", verticalalignment="center", rotation=90)

def plot_input_latent(x1, x2, L1, L2, figsize=(10, 10), **kwargs):
    """Plots both input images and latent spaces in a 2x2 figure."""
    fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True, **kwargs)
    ax[0, 0].set_title("Original inputs")

    # 1st column
    ax[0, 0].set_ylabel("Modality A")
    ax[0, 0].imshow(tensor2np(x1))
    ax[1, 0].set_ylabel("Modality B")
    ax[1, 0].imshow(tensor2np(x2))

    # 2nd column
    ax[0, 1].set_title("Latent space")
    minint, maxint = min(L1.min(), L2.min()), max(L1.max(), L2.max())
    L1n, L2n = normalize_common(L1, L2)
    ax[0, 1].imshow(L1n)
    ax[1, 1].imshow(L2n)

    plt.tight_layout()