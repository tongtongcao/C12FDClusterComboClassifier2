import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import time
import argparse
import os

from trainer import *
from plotter import Plotter


def parse_args():
    """
    Parse command-line arguments for training and inference.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="MLP Training and Inference")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "auto"], default="auto",
                        help="Choose device: cpu, gpu, or auto (default: auto)")
    parser.add_argument("inputs", type=str, nargs="*", default=["avgWiresSlopesLabel.csv"],
                        help="One or more input CSV files")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for DataLoader")
    parser.add_argument("--outdir", type=str, default="outputs/local",
                        help="Directory to save models and plots")
    parser.add_argument("--end_name", type=str, default="",
                        help="Optional suffix to append to output files (default: none)")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate for the MLP model (default: 0.2)")
    parser.add_argument("--no_train", action="store_true",
                        help="Skip training and only run inference using a saved model")
    parser.add_argument("--enable_progress_bar", action="store_true",
                        help="Enable progress bar during training (default: disabled)")
    return parser.parse_args()


class ClusterDataset(Dataset):
    """
    PyTorch Dataset for cluster data.

    Each row of the input CSV is expected to have 12 features:
        - First 6 columns: avgWire values (normalized by 112)
        - Last 6 columns: slope values

    The last column is assumed to be the label.
    """
    def __init__(self, csv_files):
        """
        Parameters
        ----------
        csv_files : str or list of str
            Path(s) to CSV file(s) containing cluster data.
        """
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        dfs = [pd.read_csv(f, header=None) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)

        # Split features
        X_avg = df.iloc[:, :6].values.astype(np.float32) / 112.0
        X_slope = df.iloc[:, 6:12].values.astype(np.float32)
        self.X = np.concatenate([X_avg, X_slope], axis=1)
        self.y = df.iloc[:, -1].values.astype(np.float32)

    def __len__(self):
        """
        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Retrieve a sample.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        tuple
            Tuple (features, label) as torch tensors.
        """
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def main():
    """
    Main script for loading data, training MLP, and running inference.
    """
    args = parse_args()

    outDir = args.outdir
    maxEpochs = args.max_epochs
    batchSize = args.batch_size
    end_name = args.end_name
    doTraining = not args.no_train

    os.makedirs(outDir, exist_ok=True)

    print('\n\nLoading data...')
    startT_data = time.time()

    # Load dataset
    dataset = ClusterDataset(args.inputs)
    y = torch.tensor(dataset.y)

    num_pos = (y == 1).sum().item()
    num_neg = (y == 0).sum().item()
    pos_weight = num_neg / num_pos

    print(f"num_pos={num_pos}, num_neg={num_neg}, pos_weight={pos_weight:.3f}")

    # Split train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    print('\n\nTrain size:', train_size)
    print('Test size:', val_size)

    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batchSize, shuffle=False)

    X_sample, y_sample = next(iter(train_loader))
    print('X_sample:', X_sample.shape)
    print('y_sample:', y_sample.shape)

    endT_data = time.time()
    print(f'Loading data took {endT_data - startT_data:.2f}s \n\n')

    # Define plotter
    plotter = Plotter(print_dir=outDir, end_name=end_name)

    # Initialize model
    model = ClusterComboMLP(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        pos_weight=pos_weight,
        dropout=args.dropout
    )

    loss_tracker = LossTracker()

    if doTraining:
        # Device selection
        if args.device == "cpu":
            accelerator, devices = "cpu", 1
        elif args.device == "gpu":
            if torch.cuda.is_available():
                accelerator, devices = "gpu", 1
            else:
                print("GPU requested but not available. Falling back to CPU.")
                accelerator, devices = "cpu", 1
        elif args.device == "auto":
            if torch.cuda.is_available():
                accelerator, devices = "gpu", "auto"
            else:
                accelerator, devices = "cpu", 1
        else:
            raise ValueError(f"Unknown device option: {args.device}")

        print(f"Using accelerator={accelerator}, devices={devices}")

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy="auto",
            max_epochs=maxEpochs,
            enable_progress_bar=args.enable_progress_bar,
            log_every_n_steps=1000,
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            logger=False,
            callbacks=[loss_tracker]
        )

        print('\n\nTraining...')
        startT_train = time.time()
        trainer.fit(model, train_loader, val_loader)
        endT_train = time.time()

        print(f'Training took {(endT_train - startT_train) / 60:.2f} minutes \n\n')

        plotter.plotTrainLoss(loss_tracker)

        # Save model
        model.to("cpu")
        torchscript_model = torch.jit.script(model)
        torchscript_model.save(f"{outDir}/mlp_{end_name}.pt")

    # Load model for inference
    if doTraining:
        model = torch.jit.load(f"{outDir}/mlp_{end_name}.pt")
    else:
        model = torch.jit.load(f"nets/mlp_default.pt")
    model.eval()

    # Run inference
    startT_test = time.time()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            y_pred = model(x)
            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())
    endT_test = time.time()
    print(f'Test with {val_size} samples took {endT_test - startT_test:.2f}s \n\n')

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    print(f"Predictions shape: {all_preds.shape}, Targets shape: {all_targets.shape}")

    # Plot results
    plotter.plotLabelProbs(all_preds, all_targets)
    plotter.plotTPR_TNR_vs_Threshold(all_preds, all_targets)
    plotter.plotPrecisionRecallF1_vs_Threshold(all_preds, all_targets)

    # Print sample predictions
    for i in range(min(10, all_preds.size(0))):
        print(f"Cluster {i}: Prob={all_preds[i]:.3f}, Target={all_targets[i].item()}")


if __name__ == "__main__":
    main()
