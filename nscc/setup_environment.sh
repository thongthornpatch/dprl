#!/bin/bash
# Environment Setup Script for NSCC
# Run this once after transferring code to NSCC

set -e  # Exit on error

echo "============================================"
echo "Setting up Denial Prompting RL Environment"
echo "============================================"

# Load modules (adjust versions based on NSCC availability)
echo "Loading modules..."
module load python/3.10
module load cuda/11.8
module load cudnn/8.6

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "============================================"
echo "Verifying installation..."
echo "============================================"

python -c "
import torch
import transformers
from trl import GRPOTrainer
import RestrictedPython

print('✅ PyTorch version:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ CUDA version:', torch.version.cuda)
    print('✅ GPU:', torch.cuda.get_device_name(0))
print('✅ Transformers version:', transformers.__version__)
print('✅ TRL installed')
print('✅ RestrictedPython installed')
"

echo ""
echo "============================================"
echo "Environment setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Download NeoCoder dataset:"
echo "   python scripts/download_neocoder.py"
echo ""
echo "2. Run a test:"
echo "   python scripts/test_setup.py"
echo ""
echo "3. Submit training job:"
echo "   sbatch nscc/train_job.sh"
echo ""
