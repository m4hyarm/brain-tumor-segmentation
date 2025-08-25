import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

# ---------------- Reproducibility ----------------
def set_seed(seed: int = 42):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------- Metrics ----------------
class DiceScore(nn.Module):
    """Dice similarity coefficient."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        return (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)


class JaccardScore(nn.Module):
    """Jaccard index (IoU)."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        return (intersection + self.smooth) / (y_pred.sum() + y_true.sum() - intersection + self.smooth)


# ---------------- Training Setup ----------------
def get_training_setup(net, lr=1e-4, weight_decay=1e-3):
    """Return loss, optimizer, scheduler."""
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    return loss, optimizer, scheduler


# ---------------- Plotting ----------------
def plot_loss(train_loss, val_loss, num_epochs):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_loss, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.show()


def prediction_plots(model, dataloader, device, num_samples=5):
    """
    Plot sample predictions vs ground truth.
    """
    model.eval()
    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = torch.round(outputs)

    fig, axes = plt.subplots(num_samples, 3, figsize=(5, 8))
    for i in range(num_samples):
        # Input image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * 0.5 + 0.5  # unnormalize
        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")

        # Ground truth
        axes[i, 1].imshow(masks[i].cpu().squeeze(), cmap="gray")
        axes[i, 1].axis("off")

        # Prediction
        axes[i, 2].imshow(preds[i].cpu().squeeze(), cmap="gray")
        axes[i, 2].axis("off")

        if i == 0:
            axes[i, 0].set_title("Input")
            axes[i, 1].set_title("GT Mask")
            axes[i, 2].set_title("Pred Mask")

    plt.tight_layout(pad=0.5)
    plt.show()
