# test_caltech_dataset.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class TestCaltechDataset(Dataset):
    """
    Dataset for evaluating on the Caltech Pedestrian dataset.
    Returns a list of blocks, each block is a tensor of shape [seq_len, C, H, W].
    In 'train' mode, generates `num_blocks_per_folder` random blocks per video;
    in 'val' mode, only 1 block (you can adjust this if you need more).
    """

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        seq_len: int,
        transform=None,
        target_size: tuple[int, int] | None = None,
        num_blocks_per_folder: int = 1,
        mode: str = 'val'
    ):
        """
        Args:
            csv_file: Path to a CSV with two columns [video_path, video_length].
            root_dir: Root directory where Caltech Pedestrian subfolders live.
            seq_len: Number of frames per block (warm-up + prediction).
            transform: Transformations to apply to each PIL image.
            target_size: If not None, resize each image to (height, width).
            num_blocks_per_folder: Number of blocks to sample per video in 'train' mode.
            mode: 'train' or 'val'.
        """
        self.dataframe = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.target_size = target_size
        self.num_blocks_per_folder = num_blocks_per_folder
        self.mode = mode.lower()
        assert self.mode in ('train', 'val'), "mode must be 'train' or 'val'"

    def __len__(self):
        """Return the total number of videos in the split."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieve one video sample and return a list of blocks.
        Each block is a tensor of shape [seq_len, C, H, W].
        """
        video_rel_path, video_length = self.dataframe.iloc[idx]

        # Determine how many blocks to generate
        if self.mode == 'train':
            n_blocks = self.num_blocks_per_folder
        else:  # 'val' mode
            n_blocks = 1

        blocks = []
        for _ in range(n_blocks):
            frame_indices = self._generate_random_block(video_length)
            images = []
            for i in frame_indices:
                # Build absolute path to the optical flow image
                img_path = os.path.join(
                    self.root_dir,
                    video_rel_path,
                    f'I{(i+1):05d}.jpg'
                )
                img = Image.open(img_path).convert('RGB')

                # Optionally resize
                if self.target_size is not None:
                    img = img.resize(self.target_size, Image.BILINEAR)

                # Apply any additional transforms (ToTensor, normalization, etc.)
                if self.transform:
                    img = self.transform(img)

                images.append(img)

            # Stack list of images into a tensor [seq_len, C, H, W]
            block_tensor = torch.stack(images, dim=0)
            blocks.append(block_tensor)

        return blocks

    def _generate_random_block(self, video_length: int) -> np.ndarray:
        """
        Generate a contiguous block of indices of length `seq_len`,
        chosen uniformly at random within [0, video_length - seq_len].
        """
        max_start = video_length - self.seq_len
        if max_start < 0:
            raise ValueError(
                f"seq_len={self.seq_len} is larger than video_length={video_length}"
            )
        start = np.random.randint(0, max_start + 1)
        return np.arange(start, start + self.seq_len)
