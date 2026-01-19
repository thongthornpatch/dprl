# Denial Prompting for RL Code Generation

Research project combining **Denial Prompting** (from NeoCoder) with **Reinforcement Learning** (using GRPO) to improve code generation creativity and Pass@k performance.

## 🎯 Project Goal

Train a code generation model using RL with a **simplified reward function**:
- ✅ Rewards correctness (passes test cases)
- ❌ Penalizes using denied techniques (denial prompting)

**Formula:** `Reward = Correctness - (num_violations × penalty_weight)`

**Expected Outcome:** Higher Pass@k and creativity (denial prompting forces novel solutions).

## 📁 Project Structure

```
denial_prompting_RL/
├── configs/                 # Configuration files
│   ├── config_laptop.yaml  # Laptop testing config (CPU, small model)
│   └── config_nscc.yaml    # NSCC training config (GPU, full scale)
├── src/
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model wrappers
│   ├── training/           # GRPO training loop
│   ├── rewards/            # Reward function implementation
│   ├── evaluation/         # Pass@k and NeoGauge metrics
│   └── utils/              # Utilities (config loader, logging)
├── data/
│   ├── raw/                # Raw NeoCoder dataset
│   └── processed/          # Preprocessed data
├── scripts/                # Executable scripts
├── notebooks/              # Jupyter notebooks for analysis
├── experiments/            # Saved experiment results
├── logs/                   # Training logs
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### Laptop Testing (Development)

**Note:** Full training requires GPU. Laptop is for code testing only.

```bash
# 1. Download NeoCoder dataset
python scripts/download_neocoder.py

# 2. Test Phase 1: Setup
python scripts/test_setup.py

# 3. Test Phase 2: Dataset
python scripts/test_data_pipeline.py

# 4. Test Phase 3: Reward Function
python scripts/test_reward_function.py
```

### Google Colab Pilot Test (Recommended Before NSCC)

**Test with real model + real data before expensive NSCC deployment.**

1. Open [Colab_Pilot_Test.ipynb](Colab_Pilot_Test.ipynb) in Google Colab
2. Enable T4 GPU (Runtime → Change runtime type)
3. Run all cells (~30-60 minutes)
4. Review training curves and statistics

**See [COLAB_GUIDE.md](COLAB_GUIDE.md) for detailed instructions.**

This validates your system end-to-end with:
- Real GPT-2 model (124M params)
- Real NeoCoder data (10 problems)
- Real GRPO training (50 steps)
- Free GPU access

### NSCC Training (Production)

```bash
# 1. Transfer code to NSCC
tar -czf denial_prompting_rl.tar.gz .
scp denial_prompting_rl.tar.gz <username>@nscc.sg:/home/users/ntu/<username>/

# 2. SSH into NSCC and setup
ssh <username>@nscc.sg
tar -xzf denial_prompting_rl.tar.gz
cd denial_prompting_RL
./nscc/setup_environment.sh

# 3. Download dataset
source venv/bin/activate
python scripts/download_neocoder.py

# 4. Submit training job
sbatch nscc/train_job.sh

# 5. Monitor progress
tail -f logs/train_*.out
```

**See [nscc/README_DEPLOYMENT.md](nscc/README_DEPLOYMENT.md) for detailed instructions.**

## 📊 Configuration

The project uses YAML configs for easy switching between laptop and NSCC:

- **config_laptop.yaml**: CPU, GPT-2 (124M), 10 problems, 200 steps
- **config_nscc.yaml**: GPU, CodeGen-1B, 199 problems, 5000 steps

Edit configs to change:
- Model size
- Number of denial constraints (curriculum)
- Reward function weights
- Training hyperparameters

## 🧪 Current Status

- ✅ **Phase 1**: Project structure and configuration
- ✅ **Phase 2**: NeoCoder dataset loading and curriculum
- ✅ **Phase 3**: Reward function with sandboxing (simplified to 2 components)
- ✅ **Phase 4**: GRPO training loop implementation
- ✅ **Phase 5**: NSCC deployment scripts and setup
- ⏳ **Phase 6**: Full training run on NSCC A100 (ready to execute)
- ⏳ **Phase 7**: Evaluation and analysis (pending)

**Ready for NSCC training!** All code complete, just need to run on GPU cluster.

## 📚 References

- [NeoCoder Paper (NAACL 2025)](https://aclanthology.org/2025.naacl-long.141/)
- [NeoCoder GitHub](https://github.com/JHU-CLSP/NeoCoder)
- [GRPO Explained](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/)

## 👥 Team

Research project for code generation creativity with RL training.
