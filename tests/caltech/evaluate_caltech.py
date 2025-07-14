#!/usr/bin/env python3
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

# Import pretrained ACCLIP
from modules.model_ACCLIP import ACCLIP
# Import the TestCaltechDataset 
from caltech.TestCaltechDataset import TestCaltechDataset
# Import metric functions
from utils.caltech_metrics import calculate_ssim, calculate_psnr
# Import plotting helper


def infer_sequence(model, loader, device, num_predictions):
    """
    Perform inference on the first batch of the loader and split
    into warm-up frames and predictions.

    Returns:
        target:    Tensor [num_predictions, C, H, W]
        preds:     Tensor [num_predictions, C, H, W]
        input_seq: Tensor [seq_len,       C, H, W]
    """
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        # If dataset returns a list of blocks, take the first block
        if isinstance(batch, list):
            frames = batch[0]
        # Otherwise assume it's a single tensor already
        else:
            frames = batch

        # Ensure batch dim: [1, T, C, H, W]
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)
        frames = frames.to(device)

        B, T_tot, C, H, W = frames.shape
        T_past = T_tot - num_predictions

       
        outputs = model(frames)
        preds = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

        # If model returns more than needed, keep only the last num_predictions
        if preds.shape[1] > num_predictions:
            preds = preds[:, -num_predictions:]

        # Slice ground-truth and warm-up
        target    = frames[:, T_past:]   # [B, num_pred, C, H, W]
        input_seq = frames[:, :T_past]   # [B, T_past, C, H, W]

        # Remove batch dim
        return target.squeeze(0), preds.squeeze(0), input_seq.squeeze(0)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AE on Caltech Pedestrian: compute SSIM/PSNR and plot differences"
    )
    parser.add_argument(
        '-w','--weights', required=True,
        help="Path to the pretrained model weights (.pth file)"
    )
    parser.add_argument(
        '-c','--csv', required=True,
        help="CSV file with test split (video_path, video_length)"
    )
    parser.add_argument(
        '-r','--root-dir', required=True,
        help="Root directory of the Caltech Pedestrian dataset"
    )
    parser.add_argument(
        '-n','--num-pred', type=int, default=8,
        help="Number of consecutive frames to predict"
    )
    parser.add_argument(
        '-i','--initial-seq', type=int, default=10,
        help="Number of warm-up frames before prediction"
    )
    parser.add_argument(
        '--height', type=int, default=128,
        help="Height to resize input images"
    )
    parser.add_argument(
        '--width', type=int, default=128,
        help="Width to resize input images"
    )
    parser.add_argument(
        '--mode', choices=['train','val'], default='val',
        help="Dataset mode: 'train' (multiple blocks) or 'val' (single block)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load model and weights
    model = ACCLIP(num_predictions=args.num_pred).to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt)

    # 2) Prepare dataset & dataloader
    transform = Compose([
        Resize((args.height, args.width)),
        ToTensor(),
    ])
    dataset = TestCaltechDataset(
        csv_file    = args.csv,
        root_dir    = args.root_dir,
        seq_len     = args.initial_seq + args.num_pred,
        transform   = transform,
        target_size = (args.height, args.width),
        mode        = args.mode
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 3) Run inference
    target, preds, input_seq = infer_sequence(
        model, loader, device, args.num_pred
    )

    # 4) Compute and print SSIM & PSNR per predicted frame
    print(f"\n=== Metrics for first {args.num_pred} predictions ===")
    for idx in range(args.num_pred):
        ssim_val = calculate_ssim(preds[idx], target[idx])
        psnr_val = calculate_psnr(preds[idx], target[idx])
        print(f"Frame {idx+1:2d}: SSIM = {ssim_val:.4f} | PSNR = {psnr_val:.2f} dB")

    # 5) Plot each prediction with its difference
    prev_frame = input_seq[args.initial_seq - 1]
    for idx in range(args.num_pred):
        plot_with_diff(
            frame_n          = prev_frame,
            frame_np1_real   = target[0],
            frame_np1_pred   = preds[0],
            cmap_diff        = 'viridis',
            clip_percentiles = (1, 99),
            lognorm_vmin     = 0.03,
            lognorm_vmax     = 0.1
        )

if __name__ == "__main__":
    main()
