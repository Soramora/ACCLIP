#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from KTH.KTHDataset import KTHDataset
from modules.model_ACCLIP import ACCLIP
from utils.caltech_metrics import calculate_ssim, calculate_psnr  # ensure this has *_np
from utils.KTH_plot import KTH_plot

def evaluate(model, loader, device, initial_seq, num_pred):
    """
    Run inference over the entire loader, collecting SSIM and PSNR
    for each of the num_pred frames, then return overall averages.
    """
    ssim_vals = []
    psnr_vals = []
    
    first = True
    mid_print_done = False
    model.eval()
    with torch.no_grad():
        for batch in loader:
            # handle Dataset returning list of blocks
            frames = batch[0] if isinstance(batch, list) else batch
            if frames.ndim == 4:  # [T, C, H, W] → [1, T, C, H, W]
                frames = frames.unsqueeze(0)
            frames = frames.to(device)

            # Forward in inference mode
            outputs = model(frames)
            preds   = outputs[0] if isinstance(outputs, (list,tuple)) else outputs
            preds   = preds[:, :num_pred]   # keep only first num_pred

            # ground truth slice
            target  = frames[:, initial_seq:initial_seq+num_pred]
           # save the very first comparison grid
            if first:
                # remove batch dim: [1, T, C, H, W] → [T, C, H, W]
                t0 = target.squeeze(0)
                p0 = preds .squeeze(0)
                KTH_plot(t0, p0, n=20, start_index  = initial_seq + 1, save_path="first20_gt_pred.png")
                first = False

            B = preds.shape[0]
            for b in range(B):
                for i in range(num_pred):
                    p = preds[b, i]
                    t = target[b, i]
                    ssim_vals.append(calculate_ssim(p, t))
                    psnr_vals.append(calculate_psnr(p, t))
                  # the moment we finish the 10th prediction, print intermediate averages
                    if not mid_print_done and (i+1) == 10:
                        avg10_ssim = float(np.mean(ssim_vals))
                        avg10_psnr = float(np.mean(psnr_vals))
                        print(f"\n=== Intermediate averages at 10 predictions ===")
                        print(f"SSIM: {avg10_ssim:.4f} | PSNR: {avg10_psnr:.2f} dB\n")
                        mid_print_done = True
            
    # compute overall averages
    return float(np.mean(ssim_vals)), float(np.mean(psnr_vals))

def main():
    parser = argparse.ArgumentParser(
        description="Compute average SSIM/PSNR over 40-frame predictions on KTH"
    )
    parser.add_argument('--weights',  '-w', required=True,
                        help="Path to pretrained model weights (.pth)")
    parser.add_argument('--csv',      '-c', required=True,
                        help="CSV listing test sequences (video_path, video_length)")
    parser.add_argument('--root-dir', '-r', required=True,
                        help="Root folder of KTH frames")
    parser.add_argument('--initial-seq','-i', type=int, default=10,
                        help="Number of warm-up frames before prediction")
    parser.add_argument('--mode',      choices=['train','val'], default='val',
                        help="Dataset mode: 'train' or 'val'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model with exactly 40 future frames
    num_pred = 40
    model = ACCLIP(num_predictions=num_pred).to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt)

    # no resizing: convert to tensor only
    transform = ToTensor()

    # dataset + loader
    seq_len = args.initial_seq + num_pred
    ds = KTHDataset(
        csv_file    = args.csv,
        root_dir    = args.root_dir,
        seq_len     = seq_len,
        transform   = transform,
        target_size = None,      # keep original resolution
        mode        = args.mode
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # evaluation
    avg_ssim, avg_psnr = evaluate(
        model, loader, device,
        args.initial_seq, num_pred
    )

    # print results
    print(f"\n=== Average over {num_pred} predictions ===")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"PSNR: {avg_psnr:.2f} dB")

if __name__ == '__main__':
    main()


