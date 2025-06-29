
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class FilterDataset(Dataset):
    """Custom PyTorch Dataset for loading the filter data."""
    def __init__(self, csv_file, grid_size):
        """
        Args:
            csv_file (str): Path to the csv file with the data.
            grid_size (int): The size of the grid.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.grid_size = grid_size

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract layout and reshape
        layout_flat = self.data_frame.iloc[idx, 0:self.grid_size**2].values
        layout = layout_flat.astype('float').reshape(self.grid_size, self.grid_size)
        layout_tensor = torch.from_numpy(layout).unsqueeze(0) # Add channel dimension

        # Extract S-parameters
        s_params = self.data_frame.iloc[idx, self.grid_size**2:].values
        s_params = s_params.astype('float')
        s_params_tensor = torch.from_numpy(s_params)

        sample = {'layout': layout_tensor, 's_params': s_params_tensor}

        return sample
