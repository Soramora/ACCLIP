# ACCLIP
ACCLIP code


# ConvGRU-Based Optical Flow Prediction

A PyTorch framework to predict future frames from past frames using a 3D-convolutional encoder and a ConvGRU decoder and a integration of the optical flow in the loss function. 
It supports configurable training via a YAML file, performance logging, and automatic plotting of training curves.

## Table of Contents

- [Features](#features)  
- [Directory Structure](#directory-structure)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [Logging & Visualization](#logging--visualization)  
- [Contributing](#contributing)  
- [License](#license)  

## Features

- **Custom ConvGRU architecture** (`AE3`) for video prediction  
- **Data loader** for `.npy` optical-flow files (`MiDataset`)  
- **Training & validation** scripts with SSIM/MSE losses  
- **YAML-based configuration** for hyperparameters and paths  
- **Performance logging** (CSV) and automatic plotting of loss/SSIM curves  

## Directory Structure



## Requirements

- **Python** >= 3.8  
- **Virtualenv** (recommended)
- torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
PyYAML>=6.0.0
tqdm>=4.65.0
Pillow>=9.5.0

## Running the Pre-trained Model

To run the pre-trained model on the different datasets, navigate to the `test` directory and execute the shell script. The model weights are stored in the `weight` directory.

```bash
cd test
./run.sh
opencv-python>=4.8.0
scikit-image>=0.21.0
matplotlib>=3.7.0
