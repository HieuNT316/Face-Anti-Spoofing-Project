import torch
import torch.nn as nn

class RPPG_RNN(nn.Module):
    def __init__(self, feature_dim=128, hidden_size=64, num_layers=1):
        super(RPPG_RNN, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(64, feature_dim, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # -> [B, feature_dim, 1, 1]
        )

        self.rnn = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):  # x: [B, T, 3, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)  # [B*T, 3, H, W]
        features = self.cnn(x).view(B, T, -1)  # [B, T, feature_dim]
        out, _ = self.rnn(features)  # [B, T, hidden]
        final_feat = out[:, -1, :]  # lấy hidden tại bước cuối
        logits = self.classifier(final_feat)
        return logits
