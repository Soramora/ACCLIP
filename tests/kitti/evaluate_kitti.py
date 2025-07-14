#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
#from skimage.metrics import structural_similarity as ssim_fn
#from skimage.metrics import peak_signal_noise_ratio as psnr_fn
import lpips

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


#from kitti.KITTI_Dataset import KITTI_Dataset
from modules.dataset import MiDataset
from modules.model_ACCLIP import ACCLIP
#from utils.metrics import calculate_ssim

#from utils.metrics import calculate_psnr
#from utils.metrics import calculate_sem


def normalize_image(image):
    """Normaliza la imagen al rango [0,1]."""
    image_min = image.min()
    image_max = image.max()
    return (image - image_min) / (image_max - image_min + 1e-8)  # Evitar divisiones por cero

def calculate_ssim(img1, img2, win_size=7):
    """
    Calcula el SSIM entre dos imágenes en formato Torch Tensor.
    """
    # 🔹 Solo normalizar si los datos no están en [0,1]
    if img1.max() > 1 or img2.max() > 1:
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())  # Normalización min-max
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())  # Normalización min-max

    # 🔹 Convertir a numpy y cambiar los ejes a (H, W, C)
    img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)  
    img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)  

    # 🔹 Ajustar `win_size` para que sea válido
    win_size = min(win_size, img1.shape[0], img1.shape[1])
    
    # 🔹 Usar `channel_axis=2` en lugar de `multichannel=True`
    ssim_value = ssim(img1, img2, win_size=win_size, data_range=1.0, channel_axis=2)
    
    return ssim_value


def calculate_lpips(pred, target, device):
    """Calcula el LPIPS entre la predicción y el target, asegurando que todo esté en 'device'."""
    pred_b = pred.unsqueeze(0).to(device)
    target_b = target.unsqueeze(0).to(device)
    lpips_value = lpips_loss(pred_b, target_b)
    return lpips_value.item()
    
def calculate_psnr(pred, target):
    """Calcula el Peak Signal-to-Noise Ratio (PSNR) entre la predicción y el target."""
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()

    # Normalizar imágenes antes del cálculo
    pred_np = normalize_image(pred_np)
    target_np = normalize_image(target_np)

    data_range = target_np.max() - target_np.min()
    return psnr(target_np, pred_np, data_range=data_range)

# Función para calcular el Error Estándar de la Media (SEM)
def calculate_sem(values):
    """Calcula el Error Estándar de la Media (SEM) para una lista de valores."""
    values = np.array(values)
    return np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0.0

# Inicializar el modelo LPIPS
lpips_loss = lpips.LPIPS(net='vgg')  

def test_model(model, testloader, device, num_predictions):
    model.to(device)
    lpips_loss.to(device)  # Mover LPIPS a GPU/CPU
    model.eval()

    # Variables para acumular métricas
    ssim_values, psnr_values, lpips_values, mse_values = [], [], [], []
    t_ssim_values = []

    num_frames = 0   # Contador de frames procesados
    num_sequences = 0  # Contador de secuencias evaluadas

    test_SSIM = [[] for _ in range(num_predictions)]
    
    with torch.no_grad():
        for data in testloader:
            for block in data:
                block = block.to(device)
                B, N, C, H, W = block.shape
                input_seq = block[:, :10, :, :, :]
                #print('block', block.shape)
                # 🔹 Generar predicciones
                predictions = model(block)
                
                predictions = predictions.view(num_predictions, C, H, W)

                # 🔹 Obtener la secuencia objetivo (target)
                seq_start = N - num_predictions
                target = block[:, seq_start:, :, :, :].view(num_predictions, C, H, W)
                
                # 🔹 1) Métricas frame a frame
                for i in range(num_predictions):
                    ssim_value = calculate_ssim(predictions[i], target[i])
                    psnr_value = calculate_psnr(predictions[i], target[i])
                    lpips_value = calculate_lpips(predictions[i], target[i], device)
                    
                    
                    ssim_values.append(ssim_value)
                    psnr_values.append(psnr_value)
                    lpips_values.append(lpips_value)
                    

                    num_frames += 1
                    
                    
               
                num_sequences += 1

    # 📌 Cálculo de promedios y SEM
    avg_ssim, sem_ssim = np.mean(ssim_values), calculate_sem(ssim_values)
    avg_psnr, sem_psnr = np.mean(psnr_values), calculate_sem(psnr_values)
    avg_lpips, sem_lpips = np.mean(lpips_values), calculate_sem(lpips_values)
    

    # 📌 Mostrar resultados finales con SEM
    print(f'Total frames evaluados: {num_frames}')
    print(f'SSIM promedio: {avg_ssim:.4f} ± {sem_ssim:.4f}')
    print(f'PSNR promedio: {avg_psnr:.4f} ± {sem_psnr:.4f}')
    print(f'LPIPS promedio: {avg_lpips:.4f} ± {sem_lpips:.4f}')


    return avg_ssim
def main():
    p = argparse.ArgumentParser(
        description="Evaluate SSIM/PSNR/LPIPS per-frame on KITTI predictions"
    )
    p.add_argument('-w','--weights',   required=True, help="Path to model .pth")
    p.add_argument('-c','--csv',       required=True, help="CSV of [folder,length]")
    p.add_argument('-r','--root-dir',  required=True, help="Root of KITTI frames")
    p.add_argument('-i','--initial-seq', type=int, default=10,
                   help="Warm-up frames before pred")
    p.add_argument('-n','--num-pred',   type=int, default=20,
                   help="Number of future frames to evaluate")
    p.add_argument('--batch-size',      type=int, default=1, help="Batch size")
    p.add_argument('--mode',            choices=['train','val'], default='val')
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare resizing (only once)
    height, width = 370, 1224
    new_height = height // 4
    new_width  = int(new_height * (width/height))
    transform = transforms.Compose([
        transforms.Resize((new_height, new_width),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])

    # Loop over each prediction horizon
    for num_predictions in range(2, args.num_pred + 1):
        print(f"\n=== Evaluating num_pred = {num_predictions} ===")

        # 1) Build model for this horizon
        model_n = ACCLIP(num_predictions=num_predictions).to(device)
        ckpt    = torch.load(args.weights, map_location=device)
        model_n.load_state_dict(ckpt)
        model_n.eval()

        # 2) Build dataset+loader with matching seq_len
        seq_len = args.initial_seq + num_predictions
        test_split = '/home/smora/kitti_test.csv'
        test_video_info = pd.read_csv(test_split, header=None)
        #ds =MiDataset(
         #   csv_file    = args.csv,
          #  root_dir    = args.root_dir,
           # seq_len     = seq_len,
            ##transform   = transform,
         #   target_size = (new_height, new_width),
         #   mode        = args.mode
        #)
        #loader = DataLoader(ds, batch_size=1, shuffle=False)
        test_dataset = MiDataset(dataframe=test_video_info, seq_len=seq_len, transform=transform, target_size=(new_height,         new_width), mode='val')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # 3) Run evaluation
        avg_ssim=test_model(model_n, test_loader, device, num_predictions=num_predictions)





if __name__ == '__main__':
    main()

