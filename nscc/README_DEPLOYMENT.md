# NSCC Deployment Guide

Complete instructions for deploying and running Denial Prompting RL training on NSCC.

## Prerequisites

- NSCC account with GPU access (dgx partition)
- SSH access to NSCC login nodes
- Basic familiarity with SLURM job submission

## Step 1: Transfer Code to NSCC

From your local machine:

```bash
# Create a tarball of the project (excluding large files)
tar -czf denial_prompting_rl.tar.gz \
    --exclude='outputs/*' \
    --exclude='data/raw/*' \
    --exclude='venv/*' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    .

# Transfer to NSCC (replace <username> with your NSCC username)
scp denial_prompting_rl.tar.gz <username>@nscc.sg:/home/users/ntu/<username>/
```

## Step 2: Extract and Setup on NSCC

SSH into NSCC and run:

```bash
# Extract the project
cd /home/users/ntu/<username>/
tar -xzf denial_prompting_rl.tar.gz
cd denial_prompting_RL

# Make scripts executable
chmod +x nscc/setup_environment.sh
chmod +x nscc/train_job.sh

# Run environment setup
./nscc/setup_environment.sh
```

This will:
- Load required modules (Python, CUDA, cuDNN)
- Create a virtual environment
- Install all dependencies including PyTorch with CUDA support
- Verify GPU access

**Expected time:** 10-15 minutes

## Step 3: Download NeoCoder Dataset

```bash
# Activate virtual environment
source venv/bin/activate

# Download dataset
python scripts/download_neocoder.py
```

The dataset should be placed in: `data/raw/neocoder_repo/`

**Expected size:** ~500MB

## Step 4: Verify Setup

Run the setup test:

```bash
python scripts/test_setup.py
```

Expected output:
```
✅ All tests passed (6/6)
```

If tests fail, check:
- Python version (should be 3.10+)
- CUDA availability (`python -c "import torch; print(torch.cuda.is_available())"`)
- Module versions (`pip list | grep torch`)

## Step 5: Submit Training Job

```bash
# Create logs directory
mkdir -p logs

# Submit job
sbatch nscc/train_job.sh
```

You should see:
```
Submitted batch job 123456
```

## Step 6: Monitor Training

```bash
# Check job status
squeue -u <username>

# View live output
tail -f logs/train_<job_id>.out

# View errors
tail -f logs/train_<job_id>.err

# Check GPU usage
ssh <node_name>
nvidia-smi
```

### Training Progress

The training will:
1. Load CodeGen-1B model (~2.7B parameters)
2. Train for 5000 steps
3. Save checkpoints every 500 steps
4. Log metrics to `outputs/run_<job_id>/metrics.json`

**Expected duration:** 12-18 hours on A100 GPU

### Output Directory Structure

```
outputs/run_<job_id>/
├── checkpoints/
│   ├── step_500/
│   ├── step_1000/
│   └── final_model/
├── metrics.json
├── training.log
└── job_info.txt
```

## Step 7: Download Results

After training completes:

```bash
# On your local machine
scp -r <username>@nscc.sg:/home/users/ntu/<username>/denial_prompting_RL/outputs/run_<job_id> ./outputs/
```

## Troubleshooting

### Job Fails Immediately

Check error log:
```bash
cat logs/train_<job_id>.err
```

Common issues:
- **Out of memory**: Reduce batch size in `configs/config_nscc.yaml`
- **Module not found**: Re-run `nscc/setup_environment.sh`
- **GPU not available**: Check partition and GRES settings

### Job Gets Killed (OOM)

If training crashes due to memory:

1. Edit `configs/config_nscc.yaml`:
```yaml
training:
  batch_size: 4  # Reduce from 8
  group_size: 4  # Reduce from 8
```

2. Resubmit job

### Slow Training

Expected speeds:
- ~20-30 seconds per step with batch_size=8, group_size=8
- ~150 steps per hour
- ~33 hours for 5000 steps

If slower:
- Check GPU utilization: `nvidia-smi`
- Verify using A100 (not older V100)
- Check for I/O bottlenecks in logs

### Need to Resume Training

```bash
sbatch nscc/train_job.sh --resume_from outputs/run_<old_job_id>/checkpoints/step_1000
```

## Advanced Configuration

### Custom Training Parameters

Edit `configs/config_nscc.yaml`:

```yaml
training:
  num_steps: 10000        # Increase for longer training
  batch_size: 16          # Increase if GPU memory allows
  group_size: 16          # More solutions per problem
  learning_rate: 5e-6     # Adjust learning rate

curriculum:
  enabled: true
  warmup_steps: 2000      # Steps before increasing constraints
  max_constraints: 3      # Maximum denial constraints
```

### Multiple Training Runs

For hyperparameter sweeps:

```bash
# Run 1: Low learning rate
sbatch nscc/train_job.sh  # Uses default config

# Run 2: High learning rate
# (First edit config, then submit)
sbatch nscc/train_job.sh

# Run 3: Different model
# (Change model name in config)
sbatch nscc/train_job.sh
```

## Resource Usage

### Typical Resource Requirements

- **GPU Memory**: 30-40GB (A100 has 40GB)
- **RAM**: 32-48GB
- **Disk**: 20GB for model + checkpoints
- **Time**: 12-18 hours for 5000 steps

### Cost Estimation

NSCC charges by GPU-hour. Estimate:
- 5000 steps ≈ 15 hours
- A100 GPU rate ≈ $X/hour (check current rates)
- Total ≈ $Y per training run

## Getting Help

If you encounter issues:

1. Check SLURM output: `logs/train_<job_id>.{out,err}`
2. Check training log: `outputs/run_<job_id>/training.log`
3. Verify environment: `python scripts/test_setup.py`
4. Contact NSCC support: https://www.nscc.sg/support/

## Next Steps

After training completes, proceed to Phase 7 (Evaluation):

```bash
python scripts/evaluate.py --checkpoint outputs/run_<job_id>/checkpoints/final_model
```

This will compute:
- Pass@k metrics (k=1,5,10)
- NeoGauge creativity scores
- Denial compliance rates
