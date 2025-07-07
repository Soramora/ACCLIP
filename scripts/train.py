
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)


import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.losses import loss_function, calculate_ssim

#def train(model, train_loader, optimizer, config, device):
def train(model, train_loader, optimizer, config, device, epoch, total_epochs):
    model.train()
    total_loss, total_ssim, total_mse, num_images = 0.0, 0.0, 0.0, 0
    n_pred = config['dataset']['num_predictions']
    alpha = config['training']['alpha']
    beta = config['training']['beta']

    weights = torch.linspace(1.0, 0.5, n_pred).to(device)

    #for data in tqdm(train_loader, desc="Training", disable=True):
    pbar = tqdm(train_loader,
                desc=f"Training Epoch {epoch}/{total_epochs}",
                leave=False)
    for data in pbar:
        for block in data:
            block = block.to(device)
                  
            # Obtener las dimensiones del bloque
            batch_size , time_steps, channels, height, width = block.shape
    
            input_seq = block.shape[1] - n_pred
            input_img, target_seq = block[:, :input_seq,:,:,:], block[:, input_seq:,:,:,:]

            target = target_seq.view(n_pred, channels, height, width)
            
            optimizer.zero_grad()
            
            pred_seq = model(block)
            pred = pred_seq.view(n_pred,channels, height, width)
            
            loss = loss_function(pred_seq, target_seq, alpha=alpha, beta=beta, weights=weights)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * block.size(0)
            for i in range(n_pred):
                total_ssim += calculate_ssim(pred_seq[:, i], target_seq[:, i])
                total_mse += F.mse_loss(pred_seq[:, i], target_seq[:, i]).item()
                num_images += 1

    return total_loss / num_images, total_ssim / num_images, total_mse / num_images
