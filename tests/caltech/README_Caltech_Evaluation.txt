README - Caltech Pedestrian Evaluation

This file contains instructions to run the evaluation of your model on the Caltech Pedestrian dataset.

Prerequisites:
- Python 3.x
- Dependencies installed (use requirements.txt if available):
  torch
  torchvision
  numpy
  pandas
  scikit-image
  matplotlib

(Optional) Virtual environment setup:
  python3 -m venv venv
  source venv/bin/activate   # Linux/Mac
  .\venv\Scripts\activate    # Windows PowerShell
  pip install -r requirements.txt

Permissions:
Make the run script executable:
  chmod +x run_caltech.sh

Execution:
Use the following command from the directory containing run_caltech.sh:
  ./run_caltech.sh \
    /path/to/model_weights/*.pth \
    /home/smora/caltech_pedestrian_test.csv \
    caltech_pedestrian/frames

Arguments:
1. weights       Path to the pretrained model weights (.pth file)
2. csv           Path to the test split CSV file (video_path, video_length)
3. root_dir      Path to the Caltech Pedestrian dataset root directory

Optional arguments (defaults in parentheses):
4. num_pred      Number of frames to predict (8)
5. initial_seq   Number of warm-up frames before prediction (10)
6. height        Image height for resizing (128)
7. width         Image width for resizing (128)
8. mode          Dataset mode: 'val' (evaluation) or 'train' (multiple blocks) (val)

Example with all parameters:
  ./run_caltech.sh \
    /path/to/model.pth \
    /path/to/caltech_test_split.csv \
    caltech_pedestrian/frames \
    8 10 128 128 val

Output:
- Prints SSIM and PSNR for the first predicted frame.
- Saves an image file named first_prediction_diff.png in the current directory,
  showing: input frame, ground truth, prediction, and pixel-wise difference.
