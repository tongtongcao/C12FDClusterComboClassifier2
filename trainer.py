import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class ClusterComboMLP(pl.LightningModule):
    def __init__(self, input_dim=12, hidden_dim=16, num_layers=2, lr=1e-3, pos_weight=None,
                 dynamic_epochs=5, ema_alpha=0.9, dropout=0.2):
        """
        input_dim: 输入维度
        hidden_dim: 每层隐藏层神经元数量
        num_layers: 隐藏层数量
        lr: 学习率
        pos_weight: 全局正样本加权系数 (float)
        dynamic_epochs: 前 N epoch 使用 batch 内动态权重
        ema_alpha: 平滑过渡的指数移动平均系数
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.pos_weight_global = pos_weight
        self.dynamic_epochs = dynamic_epochs
        self.ema_alpha = ema_alpha
        self.pos_weight_smooth = pos_weight  # 当前平滑权重

        layers = []
        in_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))  # 添加 dropout
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def _compute_dynamic_pos_weight(self, y):
        """计算当前 batch 的 pos_weight"""
        num_pos = (y == 1).sum().item()
        num_neg = (y == 0).sum().item()
        if num_pos == 0:
            return self.pos_weight_smooth  # 避免除零
        batch_weight = num_neg / max(num_pos, 1)
        # 平滑更新：EMA 方式
        self.pos_weight_smooth = (
            self.ema_alpha * self.pos_weight_smooth + (1 - self.ema_alpha) * batch_weight
        )
        return self.pos_weight_smooth

    def _compute_loss(self, y_hat, y):
        loss = self.loss_fn(y_hat, y)

        # 动态 or 固定权重
        if self.pos_weight_global is not None:
            current_epoch = getattr(self.trainer, "current_epoch", 0)
            if current_epoch < self.dynamic_epochs:
                # 阶段 1：前 N epoch 使用 batch 内动态权重
                pos_weight = self._compute_dynamic_pos_weight(y)
            else:
                # 阶段 2：平滑过渡后固定
                pos_weight = self.pos_weight_smooth
            weights = torch.ones_like(y)
            weights[y == 1] = pos_weight
            loss = loss * weights
        return loss.mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._compute_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("pos_weight", self.pos_weight_smooth, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._compute_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # 建议加上简单的学习率衰减
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LossTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())
