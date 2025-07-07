
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)


import torch
import torch.nn.functional as F
from utils.losses import loss_function, calculate_ssim

def validate(model, val_loader, config, device):
    model.eval()
    total_loss, total_ssim, total_mse, num_images = 0.0, 0.0, 0.0, 0
    n_pred = config['dataset']['num_predictions']
    alpha = config['training']['alpha']
    beta = config['training']['beta']

    weights = torch.linspace(1.0, 0.5, n_pred).to(device)

    with torch.no_grad():
        for data in val_loader:
            for block in data:
                block = block.to(device)
                input_seq = block.shape[1] - n_pred
                input_img, target_seq = block[:, :input_seq], block[:, input_seq:]

                pred_seq = model(block)
                loss = loss_function(pred_seq, target_seq, alpha=alpha, beta=beta, weights=weights)

                total_loss += loss.item() * block.size(0)
                for i in range(n_pred):
                    total_ssim += calculate_ssim(pred_seq[:, i], target_seq[:, i])
                    total_mse += F.mse_loss(pred_seq[:, i], target_seq[:, i]).item()
                    num_images += 1

    return total_loss / num_images, total_ssim / num_images, total_mse / num_images
