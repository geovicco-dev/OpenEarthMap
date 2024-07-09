import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap

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
    
def plot_test_batch(pipeline, savefig_path, randomised=True): 

    # Define a discrete colormap with nine distinct colors
    cmap = ListedColormap([
        '#e6194b', '#3cb44b', '#ffe119', '#0082c8',
        '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#d2f53c'
    ])

    # Assuming the test dataloader and model setup as in the previous context
    pipeline.dm.setup('test')

    # Get the test dataloader
    test_dataloader = pipeline.dm.test_dataloader()
    test_dataloader_list = list(test_dataloader)

    if randomised:
        # Randomly select a batch of data
        x, y = random.choice(test_dataloader_list)
    else:
        # Select from first batch of data
        x, y = next(iter(test_dataloader))

    # Put the model in evaluation mode
    pipeline.model.eval()

    # Disable gradients for this step
    with torch.no_grad():
        # Pass the data through the model
        y_hat = pipeline.model(x)  # Ensure this is on the correct device if necessary

    # Get the predicted mask by taking the argmax across the class dimension
    predicted_mask = torch.argmax(y_hat, dim=1).cpu().numpy()

    # Plot the results
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))  # 3 rows: image, ground truth, predicted mask
    for i in range(4):
        # Plot the image
        image = np.transpose(x[i][:3, :, :], (1, 2, 0))
        axs[0, i].imshow(image)
        axs[0, i].axis('off')
        if i == 0:
            axs[0, i].set_title('Image')
        
        # Plot the ground truth mask
        ground_truth_mask = y[i][0].cpu().numpy()
        axs[1, i].imshow(ground_truth_mask, cmap=cmap, vmin=0, vmax=8)
        axs[1, i].axis('off')
        if i == 0:
            axs[1, i].set_title('Ground Truth Mask')

        # Plot the predicted mask
        axs[2, i].imshow(predicted_mask[i], cmap=cmap, vmin=0, vmax=8)
        axs[2, i].axis('off')
        if i == 0:
            axs[2, i].set_title('Predicted Mask')

    plt.tight_layout()
    plt.savefig(savefig_path, bbox_inches='tight', pad_inches=0.1, dpi = 300)
    plt.show()