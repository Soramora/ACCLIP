import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class KTHDataset(Dataset):
    '''
    PyTorch Dataset for KTH action sequences.
    Each item returns a list of blocks, where each block is a tensor
    of shape [seq_len, C, H, W].

    Args:
        csv_file (str): Path to CSV with two columns [video_path, video_length].
        root_dir (str): Root directory where frame subdirectories are located.
        seq_len (int): Number of frames per block (warm-up + predictions).
        transform (callable, optional): Transform to apply to each PIL image.
        target_size (tuple, optional): (H, W) to resize frames before transform.
        num_blocks_per_folder (int): Number of random blocks per sequence in 'train' mode.
        mode (str): 'train' or 'val'. In 'val' mode, returns one block per sequence.
    '''
    
    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        seq_len: int,
        transform=None,
        target_size=None,
        num_blocks_per_folder=1,
        mode='val'):
        
        self.dataframe = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.target_size = target_size
        self.num_blocks_per_folder = num_blocks_per_folder
        self.mode = mode.lower()
        assert self.mode in ('train', 'val'), "mode must be 'train' or 'val'"

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        video_rel_path, video_length = self.dataframe.iloc[idx]
        # Determine how many blocks to sample
        n_blocks = self.num_blocks_per_folder if self.mode == 'train' else 1
        blocks = []
        for _ in range(n_blocks):
            indices = self._generate_random_block(video_length)
            frames = []
            for i in indices:
                # Construct the frame file path
                img_path = os.path.join(
                    self.root_dir,
                    video_rel_path,
                    f"frame_{i:04d}.jpg"
                )
                img = Image.open(img_path).convert('RGB')
                # Resize if requested
                if self.target_size is not None:
                    img = img.resize(self.target_size, Image.BILINEAR)
                # Apply transform (e.g., ToTensor, normalization)
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
            # Stack into tensor [seq_len, C, H, W]
            block_tensor = torch.stack(frames, dim=0)
            blocks.append(block_tensor)
        return blocks

    def _generate_random_block(self, video_length: int) -> np.ndarray:
        max_start = video_length - self.seq_len
        if max_start < 0:
            raise ValueError(
                f"seq_len={self.seq_len} exceeds video_length={video_length}"
            )
        start = np.random.randint(0, max_start+1)
        return np.arange(start, start+self.seq_len)

