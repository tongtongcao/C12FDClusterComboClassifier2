import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

plt.rcParams.update({
    'font.size': 15,
    'legend.edgecolor': 'white',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.linewidth': 2,
    'lines.linewidth': 3
})


class Plotter:
    """
    Utility class for plotting training losses, prediction probabilities,
    and classification metrics (TPR/TNR, Precision/Recall/F1) versus thresholds.
    """
    def __init__(self, print_dir='', end_name=''):
        """
        Parameters
        ----------
        print_dir : str
            Directory to save plots.
        end_name : str
            Optional suffix for plot filenames.
        """
        self.print_dir = print_dir
        self.end_name = end_name

    def plotTrainLoss(self, tracker):
        """
        Plot training and validation loss over epochs.

        Parameters
        ----------
        tracker : LossTracker
            Callback object containing train_losses and val_losses.
        """
        train_losses = tracker.train_losses
        val_losses = tracker.val_losses

        plt.figure(figsize=(20, 20))
        plt.plot(train_losses, label='Train', color='royalblue')
        plt.plot(val_losses, label='Test', color='firebrick')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()

        os.makedirs(self.print_dir, exist_ok=True)
        outname = f"{self.print_dir}/loss_{self.end_name}.png"
        plt.savefig(outname)
        plt.close()

    def plotLabelProbs(self, preds, targets, bins=100):
        """
        Plot predicted probability distributions for positive and negative samples.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted probabilities of shape (N,) or (N, 1).
        targets : torch.Tensor
            Ground truth labels of shape (N,) or (N, 1), values 0 or 1.
        bins : int, optional
            Number of histogram bins (default: 100).
        """
        preds = preds.squeeze().cpu().numpy()
        targets = targets.squeeze().cpu().numpy()

        probs_1 = preds[targets == 1]
        probs_0 = preds[targets == 0]

        bin_edges = np.linspace(0.0, 1.0, bins + 1)

        plt.figure(figsize=(10, 6))
        if len(probs_1) > 0:
            plt.hist(probs_1, bins=bin_edges, alpha=0.6, color='green', label=f'Label=1 ({len(probs_1)})')
        if len(probs_0) > 0:
            plt.hist(probs_0, bins=bin_edges, alpha=0.6, color='red', label=f'Label=0 ({len(probs_0)})')

        plt.xlabel("Predicted probability")
        plt.ylabel("Count")
        plt.yscale('log')
        plt.title("Probability distribution")
        plt.legend()
        plt.tight_layout()

        os.makedirs(self.print_dir, exist_ok=True)
        outname = f"{self.print_dir}/probs_{self.end_name}.png"
        plt.savefig(outname, dpi=300)
        plt.close()

    def plotTPR_TNR_vs_Threshold(self, preds, targets, num_points=100):
        """
        Plot True Positive Rate (TPR) and True Negative Rate (TNR) versus thresholds.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted probabilities of shape (N,) or (N, 1).
        targets : torch.Tensor
            Ground truth labels of shape (N,) or (N, 1), values 0 or 1.
        num_points : int, optional
            Number of threshold points (default: 100).
        """
        preds = preds.squeeze().cpu().numpy()
        targets = targets.squeeze().cpu().numpy()

        thresholds = np.linspace(0, 1, num_points)
        tpr_list, tnr_list = [], []

        for th in thresholds:
            pred_labels = (preds >= th).astype(int)
            TP = ((pred_labels == 1) & (targets == 1)).sum()
            TN = ((pred_labels == 0) & (targets == 0)).sum()
            FP = ((pred_labels == 1) & (targets == 0)).sum()
            FN = ((pred_labels == 0) & (targets == 1)).sum()

            tpr_list.append(TP / (TP + FN + 1e-8))
            tnr_list.append(TN / (TN + FP + 1e-8))

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, tpr_list, label='TPR', color='green', linewidth=2)
        plt.plot(thresholds, tnr_list, label='TNR', color='red', linewidth=2)
        plt.xlabel("Threshold")
        plt.ylabel("Rate")
        plt.yscale('log')
        plt.ylim(0.8, 1.0)
        plt.title("TPR/TNR vs Threshold")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        outname = f"{self.print_dir}/tpr_tnr_{self.end_name}.png"
        plt.savefig(outname, dpi=300)
        plt.close()

        data = np.vstack([thresholds, tpr_list, tnr_list]).T
        header = "threshold  tpr  tnr"
        outname_txt = f"{self.print_dir}/tpr_tnr_{self.end_name}.txt"
        np.savetxt(outname_txt, data, fmt="%.6f", header=header)

    def plotPrecisionRecallF1_vs_Threshold(self, preds, targets, num_points=100):
        """
        Plot Precision, Recall, and F1-score versus thresholds and return the best threshold.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted probabilities of shape (N,) or (N, 1).
        targets : torch.Tensor
            Ground truth labels of shape (N,) or (N, 1), values 0 or 1.
        num_points : int, optional
            Number of threshold points (default: 100).

        Returns
        -------
        float
            Threshold corresponding to the maximum F1-score.
        """
        preds = preds.squeeze().cpu().numpy()
        targets = targets.squeeze().cpu().numpy()

        thresholds = np.linspace(0, 1, num_points)
        precision_list, recall_list, f1_list = [], [], []

        for th in thresholds:
            pred_labels = (preds >= th).astype(int)
            precision_list.append(precision_score(targets, pred_labels, zero_division=0))
            recall_list.append(recall_score(targets, pred_labels, zero_division=0))
            f1_list.append(f1_score(targets, pred_labels, zero_division=0))

        precision_list = np.array(precision_list)
        recall_list = np.array(recall_list)
        f1_list = np.array(f1_list)

        best_idx = f1_list.argmax()
        best_threshold = thresholds[best_idx]

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precision_list, label='Precision', color='blue', linewidth=2)
        plt.plot(thresholds, recall_list, label='Recall', color='orange', linewidth=2)
        plt.plot(thresholds, f1_list, label='F1-score', color='green', linewidth=2)
        plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold={best_threshold:.3f}')
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Precision/Recall/F1 vs Threshold")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        outname = f"{self.print_dir}/prf_{self.end_name}.png"
        plt.savefig(outname, dpi=300)
        plt.close()

        return best_threshold
