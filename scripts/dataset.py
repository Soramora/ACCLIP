
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
        self.target_size = target_size  # Tamaño al que redimensionar las imágenes
        self.num_blocks_per_folder = num_blocks_per_folder  # Número de bloques por carpeta
        self.mode = mode  # Modo: 'train' o 'val'

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        vpath, vlen = self.dataframe.iloc[idx]
        npy_files = []

        # Si estamos en modo 'train', generamos 3 bloques, sino 1 bloque para validación
        all_blocks = []
        if self.mode == 'train':
            for _ in range(self.num_blocks_per_folder):
                block = self.generate_random_block(vlen)
                all_blocks.append(block)
        elif self.mode == 'val':
            for _ in range(self.num_blocks_per_folder):
                block = self.generate_random_block(vlen)  # Solo un bloque para validación
                all_blocks.append(block)

        # Procesar cada bloque por separado
        for block in all_blocks:
            images = []
            for i in block:
                # Ruta para cargar la imagen del flujo óptico
                seq = os.path.join('/home/smora/', vpath, '%010d.png' % (i + 1)) 
                
                # Cargar la imagen
                image = Image.open(seq)
                
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            
            # Apilar las imágenes del bloque en un tensor
            images_tensor = torch.stack(images, dim=0)

            # Devolver el tensor de imágenes (una secuencia de imágenes por bloque)
            npy_files.append(images_tensor)

        return npy_files

    def generate_random_block(self, vlen):
        """
        Generar un bloque aleatorio asegurando que no se solapen las secuencias.
        """
        start_idx = np.random.choice(range(vlen - self.seq_len))
        block = np.arange(start_idx, start_idx + self.seq_len)
        return block



# Dimensiones de ejemplo (370, 1224)
height = 370
width = 1224
new_height = int(height/4)  # Nueva altura (ejemplo)

# Calcular el nuevo ancho manteniendo la relación de aspecto
aspect_ratio = width / height
new_width = int(new_height * aspect_ratio)

# Definir las transformaciones si es necesario (por ejemplo, normalización)
transform = transforms.Compose([
    transforms.Resize((new_height,new_width),interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor()   ]) 