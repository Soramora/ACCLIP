import os
import sys
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from modules.dataset import MiDataset
from modules.model_ACCLIP import ACCLIP
from scripts.train import train
from scripts.validate import validate
from utils.visualization import plot_metrics
from utils.perf_logger import PerformanceLogger

'''
# 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(ROOT_DIR)
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")



'''

# ============================
# Configuración de rutas
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(ROOT_DIR)


#  config/config.yaml 
CONFIG_PATH = os.environ.get(
    "CONFIG_PATH",
    os.path.join(ROOT_DIR, "config", "config.yaml")
)

# ============================
# configuration
# ============================
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

print(f"🔧 Usando fichero de config: {CONFIG_PATH}")
print("  alpha en config:", config['training']['alpha'])
print("  beta  en config:", config['training']['beta'])
# ============================
# output folders
# ============================
for rel_path in [
    config['training']['checkpoint_dir'],
    "models",
    "logs",
    "plots"
]:
    abs_path = os.path.join(ROOT_DIR, rel_path)
    os.makedirs(abs_path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 
transform = transforms.Compose([
    transforms.Resize((config['dataset']['resize_height'], config['dataset']['resize_width'])),
    transforms.ToTensor()
])

# load csv
train_csv = os.path.join(config['dataset']['data_dir'], config['dataset']['train_csv'])
val_csv = os.path.join(config['dataset']['data_dir'], config['dataset']['val_csv'])

train_df = pd.read_csv(train_csv, header=None)
val_df = pd.read_csv(val_csv, header=None)

# Dataset y Dataloader
seq_len = config['dataset']['seq_len']
train_dataset = MiDataset(train_df, seq_len, transform=transform)
val_dataset = MiDataset(val_df, seq_len, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

# Modelo
model = ACCLIP(
    num_predictions=config['dataset']['num_predictions'],
    scheduled_sampling_start=config['scheduled_sampling']['start'],
    scheduled_sampling_end=config['scheduled_sampling']['end'],
    decay_rate=config['scheduled_sampling']['decay_rate']
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay']
)

# logger de performance
perf_log_cfg = config.get("performance_logging", {})
perf_logger = PerformanceLogger(
    enabled=perf_log_cfg.get("enabled", False),
    output_file=perf_log_cfg.get("output_file", "logs/performance_log.csv")
)

# Métrics
train_losses, val_losses, train_ssims, val_ssims = [], [], [], []

# train
best_val_loss = float("inf")
epochs = config['training']['epochs']

for epoch in range(epochs):
    perf_logger.start_epoch()
    print(f"[{epoch+1}/{epochs}]")

    train_loss, train_ssim, train_mse = train(model, train_loader, optimizer, config, device, epoch+1, epoch)
    val_loss, val_ssim, val_mse = validate(model, val_loader, config, device)
    model.update_scheduled_sampling()

    perf_logger.end_epoch(epoch)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_ssims.append(train_ssim)
    val_ssims.append(val_ssim)

    print(f"[{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Train SSIM: {train_ssim:.4f} | Val SSIM: {val_ssim:.4f}")

    #  chekpoint
    if config['training']['save_checkpoints']:
        ckpt_path = os.path.join(config['training']['checkpoint_dir'], f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, ckpt_path)

    # 
    # 
    alpha_val = float(os.environ.get('ALPHA', config['training']['alpha']))
    beta_val  = float(os.environ.get('BETA',  config['training']['beta']))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch + 1
      #  alpha_val     = config['training']['alpha']
       # beta_val      = config['training']['beta']
        # 
        filename      = f"best_model_alpha_{alpha_val}_beta_{beta_val}_epoch_{best_epoch}.pth"
        #filename      = f"best_model_epoch_{best_epoch}.pth"
        torch.save(model.state_dict(), os.path.join("models", filename))
        
        print(f"🔖 Mejor métrica en época {best_epoch} (Val Loss: {best_val_loss:.4f})")

# save scv metrics
metrics_df = pd.DataFrame({
    'epoch': list(range(1, len(train_losses)+1)),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'train_ssim': train_ssims,
    'val_ssim': val_ssims
})
metrics_df.to_csv("logs/training_metrics.csv", index=False)

#  performance log
perf_logger.save()

alpha_val = os.environ.get('ALPHA', config['training']['alpha'])
beta_val  = os.environ.get('BETA',  config['training']['beta'])
epochs    = config['training']['epochs']
# name: alpha, beta 
final_filename = (
    f"final_model_alpha_{alpha_val}"
    f"_beta_{beta_val}"
    f"_epoch_{epochs}.pth"
)
final_model_path = os.path.join(ROOT_DIR, "models", final_filename)

torch.save(model.state_dict(), final_model_path)
print(f"🔖 Modelo final guardado en {final_model_path}")
'''
# ============================
# save model
# ============================
final_model_path = os.path.join(ROOT_DIR, "models", "final_model.pth")
torch.save(model.state_dict(), final_model_path)
#print(f"🔖 Modelo final guardado en {final_model_path}")
'''
print(final_model_path)
