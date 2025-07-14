
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class MiDataset(Dataset):
    def __init__(self, dataframe, seq_len, transform=None, target_size=(128, 128), num_blocks_per_folder=2, mode='train'):
        self.dataframe = dataframe
        self.transform = transform
        self.seq_len = seq_len
        self.target_size = target_size  #resize images
        self.num_blocks_per_folder = num_blocks_per_folder  #number of block per folder
        self.mode = mode  # Mode: 'train' o 'val'

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        vpath, vlen = self.dataframe.iloc[idx]
        npy_files = []

        # If in 'train' mode, generate 3 blocks; otherwise, generate 1 block for validation.
        all_blocks = []
        if self.mode == 'train':
            for _ in range(self.num_blocks_per_folder):
                block = self.generate_random_block(vlen)
                all_blocks.append(block)
        elif self.mode == 'val':
            for _ in range(self.num_blocks_per_folder):
                block = self.generate_random_block(vlen) 
                all_blocks.append(block)

        # Process each block separately.
        for block in all_blocks:
            images = []
            for i in block:
                # Path to load the optical flow image
                seq = os.path.join('/home/smora/', vpath, '%010d.png' % (i + 1)) 
                
                # open image
                image = Image.open(seq)
                
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            
            
            images_tensor = torch.stack(images, dim=0)

           
            npy_files.append(images_tensor)

        return npy_files

    def generate_random_block(self, vlen):
        """
        # Generate a random block, ensuring that the sequences do not overlap.
        """
        start_idx = np.random.choice(range(vlen - self.seq_len))
        block = np.arange(start_idx, start_idx + self.seq_len)
        return block



# dimension of image
height = 370
width = 1224
new_height = int(height/4)  

# Calculate the new width while maintaining the aspect ratio
aspect_ratio = width / height
new_width = int(new_height * aspect_ratio)

# Define the transformations if needed
transform = transforms.Compose([
    transforms.Resize((new_height,new_width),interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor()   ]) 
