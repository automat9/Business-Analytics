# NOTICE: 100% All ChatGPT generated!

# Check git version to see if installed
!git --version

# Debugging
!pip install backports.functools_lru_cache
!pip install opencv-python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["COMET_MODE"] = "DISABLED"

# Import and verify PyTorch installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Verify YOLOv5 installation by loading the model
from yolov5 import utils
print("YOLOv5 installation successful!")


##### Test Run #####
# Navigate to the yolov5 directory
%cd yolov5
# Train using the following syntax and config
!python train.py --img 640 --batch 8 --epochs 50 --data "C:/Users/mp967/OneDrive - University of Exeter/Exeter University/Units/Topics in Business Analytics/dataset/data.yaml" --weights yolov5s.pt --device cpu --name hi_vis_detector_cpu
# Result: Interrupted kernel after 1st epoch, code is working but way too slow (reason: using cpu instead of gpu)

# Replace cpu with gpu: for CUDA 11.8, install PyTorch with CUDA support 
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
