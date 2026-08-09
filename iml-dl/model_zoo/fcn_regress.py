# FCN Code implemented from: https://github.com/Haozhoong/PUQ/blob/main/models/fcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, in_ch, out_ch,
                 hidden_ch, num_layers, bn=False, activation="None", dropout_rate=0.0, **kwargs):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential()
        self.fcn.add_module('fc0', nn.Linear(in_ch, hidden_ch))
        if bn:
            self.fcn.add_module('bn0', nn.BatchNorm1d(hidden_ch, affine=True))
        self.fcn.add_module('relu0', nn.ReLU())
        if dropout_rate > 0:
            self.fcn.add_module('dropout0', nn.Dropout(dropout_rate))
            
        for i in range(num_layers - 1):
            self.fcn.add_module('fc{}'.format(i+1),
                                nn.Linear(hidden_ch, hidden_ch))
            if bn:
                self.fcn.add_module('bn{}'.format(i+1),
                                    nn.BatchNorm1d(hidden_ch, affine=True))
            self.fcn.add_module('relu{}'.format(i+1), nn.ReLU())
            if dropout_rate > 0:
                self.fcn.add_module('dropout{}'.format(i+1), nn.Dropout(dropout_rate))
        
        self.t2_head = nn.Linear(hidden_ch, 1)
        if out_ch == 2:
            # T2 and S0
            self.s0_head = nn.Linear(hidden_ch, 1)

        if activation.lower() == "sigmoid":
            self.last_activation = nn.Sigmoid()
        elif activation.lower() == "relu":
            self.last_activation = nn.ReLU()
        
        
        
    def forward(self, x):
        x = self.fcn(x)
        t2 = self.t2_head(x)
        if hasattr(self, 'last_activation'):
            t2 = self.last_activation(t2)
        if hasattr(self, 's0_head'):
            s0 = self.s0_head(x)
            if hasattr(self, 'last_activation'):
                s0 = self.last_activation(s0)
            return t2, s0
        return t2



class HeteroscedasticMLP(nn.Module):
    def __init__(self, in_ch, out_ch,
                 hidden_ch, num_layers, bn=False, activation="None", dropout_rate=0.0, **kwargs):
        super(HeteroscedasticMLP, self).__init__()
        self.fcn = nn.Sequential()
        self.fcn.add_module('fc0', nn.Linear(in_ch, hidden_ch))
        if bn:
            self.fcn.add_module('bn0', nn.BatchNorm1d(hidden_ch, affine=True))
        self.fcn.add_module('relu0', nn.ReLU())
        if dropout_rate > 0:
            self.fcn.add_module('dropout0', nn.Dropout(dropout_rate))
            
        for i in range(num_layers - 1):
            self.fcn.add_module('fc{}'.format(i+1),
                                nn.Linear(hidden_ch, hidden_ch))
            if bn:
                self.fcn.add_module('bn{}'.format(i+1),
                                    nn.BatchNorm1d(hidden_ch, affine=True))
            self.fcn.add_module('relu{}'.format(i+1), nn.ReLU())
            if dropout_rate > 0:
                self.fcn.add_module('dropout{}'.format(i+1), nn.Dropout(dropout_rate))
        self.t2_mean = nn.Linear(hidden_ch, 1)
        self.t2_logvar = nn.Linear(hidden_ch, 1)
        if out_ch == 2:
            # T2 and S0
            self.s0_mean = nn.Linear(hidden_ch, 1)
            self.s0_logvar = nn.Linear(hidden_ch, 1)

        if activation.lower() == "sigmoid":
            self.last_activation = nn.Sigmoid()
        elif activation.lower() == "relu":
            self.last_activation = nn.ReLU()
        
        
        
    def forward(self, x):
        x = self.fcn(x)
        t2_mean, t2_logvar = self.t2_mean(x), self.t2_logvar(x)
        if hasattr(self, 'last_activation'):
            t2_mean = self.last_activation(t2_mean)
        if hasattr(self, 's0_mean'):
            s0_mean, s0_logvar = self.s0_mean(x), self.s0_logvar(x)
            if hasattr(self, 'last_activation'):
                s0_mean = self.last_activation(s0_mean)
            return (t2_mean, t2_logvar), (s0_mean, s0_logvar)
        return t2_mean, t2_logvar


class HeteroscedasticLoss(nn.Module):
    """
    β-NLL heteroscedastic loss with gradient stabilization.
    
    Based on:
        - Standard NLL: Kendall & Gal, "What Uncertainties Do We Need in 
          Bayesian Deep Learning for Computer Vision?" (NeurIPS 2017)
        - β-NLL fix: Seitzer et al., "On the Pitfalls of Heteroscedastic 
          Uncertainty Estimation with Probabilistic Neural Networks" (ICLR 2022)
    
    The β parameter mitigates gradient scaling issues that can cause the model
    to predict unrealistically low uncertainties.
    """
    def __init__(self, base_loss='l2', beta=0.5, min_log_var=-8, max_log_var=1):
        """
        Args:
            base_loss: 'l1' (Laplacian), 'l2' (Gaussian), or 'huber'
            beta: Variance term weight (0.5 recommended, 0.0 = standard NLL)
            min_log_var: Lower bound for log(σ²) (prevents numerical issues)
            max_log_var: Upper bound for log(σ²) (prevents infinite uncertainty)
        """
        super(HeteroscedasticLoss, self).__init__()
        self.base_loss = base_loss.lower()
        self.beta = beta
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        
        # For monitoring/debugging
        self.last_clipping_stats = {}

    def forward(self, mu, log_var, y_true):
        """
        Compute heteroscedastic negative log-likelihood loss.
        
        Args:
            mu: Predicted mean (batch, 1) or (batch,)
            log_var: Predicted log(σ²) (batch, 1) or (batch,)
            y_true: Ground truth (batch, 1) or (batch,)
        
        Returns:
            Scalar loss (mean over batch)
        """
        # Ensure consistent shapes
        if mu.dim() == 1 and y_true.dim() == 2:
            y_true = y_true.squeeze(-1)
        if mu.dim() == 2 and y_true.dim() == 1:
            mu = mu.squeeze(-1)
            log_var = log_var.squeeze(-1)
        
        # Shape validation
        assert mu.shape == y_true.shape, \
            f"Shape mismatch: mu {mu.shape} vs y_true {y_true.shape}"
        assert log_var.shape == y_true.shape, \
            f"Shape mismatch: log_var {log_var.shape} vs y_true {y_true.shape}"
        
        # Clamp log variance for numerical stability
        log_var_clamped = torch.clamp(log_var, 
                                       min=-9.0, 
                                       max=2.0)
        
        # Track clipping statistics (for logging/debugging)
        with torch.no_grad():
            self.last_clipping_stats = {
                'min_log_var': log_var.min().item(),
                'max_log_var': log_var.max().item(),
                'mean_log_var': log_var.mean().item(),
                'median_log_var': log_var.median().item(),
                'frac_clipped_low': (log_var <= self.min_log_var).float().mean().item(),
                'frac_clipped_high': (log_var >= self.max_log_var).float().mean().item()
            }
        
        # Compute loss based on assumed noise distribution
        if self.base_loss == 'l1':
            # ===== LAPLACIAN NLL =====
            # p(y|μ,σ) = 1/(2σ) exp(-|y-μ|/σ)
            # -log p = log(2σ) + |y-μ|/σ = log(2) + log(σ) + |y-μ|/σ
            log_sigma = 0.5 * log_var_clamped  # log(σ) = 0.5 * log(σ²)
            sigma = torch.exp(log_sigma)
            mae = torch.abs(y_true - mu)
            
            if self.beta > 0:
                # β-NLL: downweight variance term to prevent collapse
                loss = mae / (sigma + 1e-6) + self.beta * log_sigma
            else:
                # Standard NLL (can collapse to zero variance!)
                loss = mae / (sigma + 1e-6) + log_sigma
                
        elif self.base_loss == 'l2':
            # ===== GAUSSIAN NLL =====
            # p(y|μ,σ²) = 1/√(2πσ²) exp(-(y-μ)²/(2σ²))
            # -log p = 0.5*log(2π) + 0.5*log(σ²) + 0.5*(y-μ)²/σ²
            # (dropping constant log(2π) term)
            precision = torch.exp(-log_var_clamped)  # 1/σ²
            mse = (y_true - mu) ** 2
            
            if self.beta > 0:
                # β-NLL: mitigates gradient issues
                loss = 0.5 * precision * mse + self.beta * log_var_clamped
            else:
                # Standard NLL
                loss = 0.5 * precision * mse + 0.5 * log_var_clamped
        
        return torch.mean(loss)
    
    def get_clipping_stats(self):
        """
        Return statistics about variance clipping (for logging to wandb).
        
        Returns:
            dict: Keys are 'min_log_var', 'max_log_var', 'mean_log_var', 
                  'median_log_var', 'frac_clipped_low', 'frac_clipped_high'
        """
        return self.last_clipping_stats