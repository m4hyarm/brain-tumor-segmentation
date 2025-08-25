import argparse
import torch
from data import split_dataset, BrainSegmentationDataset, make_dataloaders
from model import ResidualUNet
from utils import set_seed, get_training_setup, plot_loss, prediction_plots
from train import fit, evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="Brain MRI Segmentation with Residual UNet")
    parser.add_argument("--data_dir", type=str, default="./lgg-mri-segmentation/kaggle_3m", 
                        help="Path to LGG dataset")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default='./segmentation_model')
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & Dataloaders
    train_ids, val_ids, test_ids = split_dataset(args.data_dir)
    trainset = BrainSegmentationDataset(args.data_dir, train_ids, augment=True, image_size=args.img_size)
    valset = BrainSegmentationDataset(args.data_dir, val_ids, image_size=args.img_size)
    testset = BrainSegmentationDataset(args.data_dir, test_ids, image_size=args.img_size)
    train_loader, val_loader, test_loader = make_dataloaders(trainset, valset, testset, batch_size=args.batch_size)

    # Model
    net = ResidualUNet(in_channels=3, out_channels=1, init_features=32).to(device)

    # Loss, Optimizer, Scheduler
    criterion, optimizer, scheduler = get_training_setup(net, lr=args.lr, weight_decay=args.weight_decay)

    # Training
    net, history = fit(net, train_loader, val_loader, criterion, optimizer, scheduler, 
                       num_epochs=args.epochs, device=device, save_path=args.save_path)

    # Evaluation
    evaluate(net, test_loader, criterion, device=device)

    # Plots
    plot_loss(history["train_loss"], history["val_loss"], args.epochs)
    prediction_plots(net, test_loader, device, num_samples=5)


if __name__ == "__main__":
    main()
