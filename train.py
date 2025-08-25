import time
import torch
from tqdm import tqdm
from utils import DiceScore, JaccardScore
import numpy as np


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=40, device=None, save_path="segmentaation_model"):
    """Train the model and return history."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dice, jaccard = DiceScore(), JaccardScore()
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": [], "train_jacc": [], "val_jacc": []}

    start = time.time()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss, train_dice, train_jacc, train_counter = 0, 0, 0, 0

        with tqdm(train_loader, desc="Train") as t:
            for imgs, masks in t:
                imgs, masks = imgs.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_dice += dice(torch.round(outputs), masks).item()
                train_jacc += jaccard(torch.round(outputs), masks).item()
                train_counter += imgs.size(0)
                t.set_postfix(loss=train_loss/train_counter, lr=optimizer.param_groups[0]["lr"])

        # Validation
        model.eval()
        val_loss, val_dice, val_jacc, val_counter = 0, 0, 0, 0
        with torch.no_grad():
            with tqdm(val_loader, desc="  Val") as t:
                for imgs, masks in t:
                    imgs, masks = imgs.to(device), masks.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)

                    val_loss += loss.item()
                    val_dice += dice(torch.round(outputs), masks).item()
                    val_jacc += jaccard(torch.round(outputs), masks).item()
                    val_counter += imgs.size(0)
                    t.set_postfix(loss=val_loss/val_counter)

        # Averages
        n_train, n_val = len(train_loader), len(val_loader)
        history["train_loss"].append(train_loss / n_train)
        history["val_loss"].append(val_loss / n_val)
        history["train_dice"].append(train_dice / n_train)
        history["val_dice"].append(val_dice / n_val)
        history["train_jacc"].append(train_jacc / n_train)
        history["val_jacc"].append(val_jacc / n_val)

        scheduler.step(history["train_loss"][-1])

        print(f"\nTrain Loss: {history['train_loss'][-1]:.4f}, Val Loss: {history['val_loss'][-1]:.4f}, "
              f"Val Dice: {history['val_dice'][-1]:.4f}, Val Jaccard: {history['val_jacc'][-1]:.4f}")

    print(f"\n[INFO] Training completed in {time.time()-start:.2f}s")

    # ---------- Save Model ----------
    torch.save(model.state_dict(), f"{save_path}.pth")
    np.save(f"{save_path}_history.npy", history)
    print(f"\n[INFO] Training complete. Model saved to {save_path}.pth")
    print(f"[INFO] Total training time: {time.time() - start:.2f}s")

    return model, history


def evaluate(model, test_loader, criterion, device=None):
    """Evaluate model on test set."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dice, jaccard = DiceScore(), JaccardScore()
    test_loss, test_dice, test_jacc = 0, 0, 0

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)

            test_loss += loss.item()
            test_dice += dice(torch.round(outputs), masks).item()
            test_jacc += jaccard(torch.round(outputs), masks).item()

    n = len(test_loader)
    print(f"\n[TEST] Loss: {test_loss/n:.4f}, Dice: {test_dice/n:.4f}, Jaccard: {test_jacc/n:.4f}")
    return {"loss": test_loss/n, "dice": test_dice/n, "jaccard": test_jacc/n}
