import os
import torch
import pickle

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

def load_data():
    history_path = sorted(Path('./data').glob('*.history'))[-1]

    with history_path.open(mode='rb') as f:
        return pickle.load(f)
    
class TicTacToeDataset(Dataset):
    def __init__(self):
        history = load_data()
        xs, y_policies, y_values = zip(*history)

        xs = torch.tensor(xs, dtype=torch.float32)
        self.xs = xs.view(len(xs), 2, 3, 3).permute(0, 2, 3, 1)
        self.y_policies = torch.tensor(y_policies, dtype=torch.float32)
        self.y_values = torch.tensor(y_values, dtype=torch.float32)

    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.y_policies[idx], self.y_values[idx]

if __name__ == '__main__':
    train_dataset = TicTacToeDataset()
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    print(f'Length of data : {len(train_dataset)}')
    print(f'Number of batches: {len(train_loader)}')

    for batch in train_loader:
        xs, y_policy, y_value = batch
        print(f'Batch Input Shape: {xs.shape}, Policy Shape: {y_policy.shape}, Value Shape: {y_value.shape}')
    #xs, y_policy, y_value = train_dataset[1]
    #print(f'Input Shape(Unbatched): {xs.shape}, Policy Shape: {y_policy.shape}, Value Shape: {y_value.shape}')
