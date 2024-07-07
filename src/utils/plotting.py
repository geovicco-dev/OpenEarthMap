import random
import numpy as np
import matplotlib.pyplot as plt

# Plot sample from the training dataset
def plot_train_batch(dm, n_samples=4, randomised=True):
    dm.setup('fit')
    # Get the train dataloader
    dataloader = dm.train_dataloader()

    if randomised:
        # Randomly select a batch of data
        x, y = random.choice(list(dataloader))
    else:
        # Select from first batch of data
        x, y = next(iter(dataloader))

    # Plot the results
    fig, axs = plt.subplots(n_samples, 2, figsize=(10, n_samples*5))
    for i in range(n_samples):
        # Plot the image
        image = x[i].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        # Get Vmin and Vmax as 2nd and 98th percentile
        vmin = np.percentile(image, 2)
        vmax = np.percentile(image, 98)
        axs[i, 0].imshow(image, vmin=vmin, vmax=vmax)
        axs[i, 0].axis('off')
        if i == 0:
            axs[i, 0].set_title('Image')
        
        # Plot the ground truth mask
        ground_truth_mask = y[i].cpu().numpy().squeeze()  # (1, H, W) -> (H, W)
        axs[i, 1].imshow(ground_truth_mask, cmap='binary_r')
        axs[i, 1].axis('off')
        if i == 0:
            axs[i, 1].set_title('Ground Truth Mask')

    plt.tight_layout()
    plt.show()