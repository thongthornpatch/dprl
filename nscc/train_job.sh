#!/bin/bash
#SBATCH --job-name=denial_prompting_rl
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# NSCC SLURM Job Script for Denial Prompting RL Training
# This script sets up the environment and runs training on NSCC A100 GPU

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "============================================"

# Load required modules (adjust based on NSCC environment)
module load python/3.10
module load cuda/11.8
module load cudnn/8.6

# Activate virtual environment
source venv/bin/activate

# Verify GPU is available
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Set environment variables
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

# Create output directory
OUTPUT_DIR="./outputs/run_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Run training
echo ""
echo "Starting training..."
echo "Output directory: $OUTPUT_DIR"
echo "============================================"

python scripts/train.py \
    --config configs/config_nscc.yaml \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

TRAIN_EXIT_CODE=$?

echo ""
echo "============================================"
echo "Training finished with exit code: $TRAIN_EXIT_CODE"
echo "End Time: $(date)"
echo "============================================"

# Save job info
cat > "$OUTPUT_DIR/job_info.txt" <<EOF
Job ID: $SLURM_JOB_ID
Job Name: $SLURM_JOB_NAME
Node: $SLURM_NODELIST
GPUs: $CUDA_VISIBLE_DEVICES
Start Time: $(date)
Exit Code: $TRAIN_EXIT_CODE
EOF

exit $TRAIN_EXIT_CODE
