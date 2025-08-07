import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Thêm path để import được module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.datasets.rppg_dataset import RPPGDataset
from models.rppg_rnn import RPPG_RNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
num_epochs = 10
batch_size = 4
lr = 1e-4
save_dir = "checkpoints_rppg"
os.makedirs(save_dir, exist_ok=True)

# Dataset
dataset = RPPGDataset(root_dir=r"D:\project_new\data\frames\frames", seq_len=100)  # nhớ seq_len phải khớp với dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = RPPG_RNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train loop
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
    for video_seq, labels in loop:
        video_seq, labels = video_seq.to(device), labels.to(device)

        outputs = model(video_seq)                   # (B, 2)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=correct / total * 100)

    acc = correct / total * 100
    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch}/{num_epochs}] Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    # Save model theo từng epoch
    model_path = os.path.join(save_dir, f"rppg_epoch{epoch}.pth")
    torch.save(model.state_dict(), model_path)
