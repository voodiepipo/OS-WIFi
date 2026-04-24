import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- GRN Layer --------
class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))
        
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x

# -------- CNN Block --------
class CNNBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        return x

# -------- Dual Stream Model --------
class DualStreamModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        # Channel stream (Amplitude)
        self.cnn_amp = CNNBlock(1)
        # Temporal stream (Phase)
        self.cnn_phase = CNNBlock(1)
        # GRN
        self.grn = GRN(64)
        # Classifier
        self.fc = nn.Linear(32 * 2, num_classes)
        
    def forward(self, amp, phase):
        # shape: (batch, time, subcarriers)
        amp = amp.mean(dim=2).unsqueeze(1) 
        phase = phase.mean(dim=2).unsqueeze(1)
        
        # (B, 1, T)
        amp_feat = self.cnn_amp(amp)
        phase_feat = self.cnn_phase(phase)
        
        # Global pooling
        amp_feat = amp_feat.mean(dim=2)
        phase_feat = phase_feat.mean(dim=2)
        
        # Combine
        x = torch.cat([amp_feat, phase_feat], dim=1)
        
        # GRN
        x = self.grn(x.unsqueeze(-1)).squeeze(-1)
        return self.fc(x)