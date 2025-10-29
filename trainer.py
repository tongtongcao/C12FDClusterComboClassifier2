import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class ClusterComboMLP(pl.LightningModule):
    def __init__(
        self,
        input_dim=12,
        hidden_dim=16,
        num_layers=2,
        lr=1e-3,
        pos_weight=None,
        dynamic_epochs=5,
        ema_alpha=0.9,
        dropout=0.2,
    ):
        """
        Multi-layer Perceptron for predicting cluster combinations with optional dynamic
        positive class weighting.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dim : int
            Number of neurons in each hidden layer.
        num_layers : int
            Number of hidden layers.
        lr : float
            Learning rate for the optimizer.
        pos_weight : float or None
            Global positive sample weight for imbalanced datasets. If None, no weighting is applied.
        dynamic_epochs : int
            Number of initial epochs to use batch-wise dynamic positive weights.
        ema_alpha : float
            Exponential moving average coefficient for smoothing dynamic weights.
        dropout : float
            Dropout probability for hidden layers. If 0, no dropout is applied.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.pos_weight_global = pos_weight
        self.dynamic_epochs = dynamic_epochs
        self.ema_alpha = ema_alpha
        self.pos_weight_smooth = pos_weight  # Smoothed weight for current epoch

        layers = []
        in_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted probabilities of shape (batch_size,).
        """
        return self.model(x).squeeze(-1)

    def _compute_dynamic_pos_weight(self, y):
        """
        Compute dynamic positive class weight for the current batch using EMA smoothing.

        Parameters
        ----------
        y : torch.Tensor
            Ground truth labels of shape (batch_size,).

        Returns
        -------
        float
            Smoothed positive weight for the batch.
        """
        num_pos = (y == 1).sum().item()
        num_neg = (y == 0).sum().item()
        if num_pos == 0:
            return self.pos_weight_smooth
        batch_weight = num_neg / max(num_pos, 1)
        self.pos_weight_smooth = (
            self.ema_alpha * self.pos_weight_smooth + (1 - self.ema_alpha) * batch_weight
        )
        return self.pos_weight_smooth

    def _compute_loss(self, y_hat, y):
        """
        Compute the binary cross-entropy loss with optional dynamic weighting.

        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted probabilities of shape (batch_size,).
        y : torch.Tensor
            Ground truth labels of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        loss = self.loss_fn(y_hat, y)

        if self.pos_weight_global is not None:
            current_epoch = getattr(self.trainer, "current_epoch", 0)
            if current_epoch < self.dynamic_epochs:
                pos_weight = self._compute_dynamic_pos_weight(y)
            else:
                pos_weight = self.pos_weight_smooth
            weights = torch.ones_like(y)
            weights[y == 1] = pos_weight
            loss = loss * weights
        return loss.mean()

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.

        Parameters
        ----------
        batch : tuple
            Tuple of (inputs, targets).
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Training loss for the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self._compute_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("pos_weight", self.pos_weight_smooth, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.

        Parameters
        ----------
        batch : tuple
            Tuple of (inputs, targets).
        batch_idx : int
            Index of the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self._compute_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns
        -------
        dict
            Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LossTracker(Callback):
    """
    PyTorch Lightning callback to track training and validation losses.
    """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Record training loss at the end of each epoch.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer instance.
        pl_module : pl.LightningModule
            The model being trained.
        """
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Record validation loss at the end of each epoch.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer instance.
        pl_module : pl.LightningModule
            The model being validated.
        """
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())