# Google Colab Testing Guide

Run a real training test with actual models and NeoCoder data before deploying to NSCC.

## Why Use Colab?

- ✅ **Free GPU access** (T4 GPU, ~16GB memory)
- ✅ **Real model training** (not just mock simulation)
- ✅ **Real NeoCoder data** (subset of 10 problems)
- ✅ **Quick validation** (30-60 minutes for 50 training steps)
- ✅ **Zero local setup** (runs entirely in browser)

## Quick Start

### Option 1: Upload Notebook Directly

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Upload notebook`
3. Upload `Colab_Pilot_Test.ipynb` from this repo
4. Follow the notebook instructions

### Option 2: Open from GitHub

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Open notebook` → `GitHub` tab
3. Enter your repo URL: `https://github.com/YOUR_USERNAME/denial_prompting_RL`
4. Select `Colab_Pilot_Test.ipynb`
5. Follow the notebook instructions

## Important: Enable GPU

⚠️ **You must enable GPU in Colab:**

1. Click `Runtime` → `Change runtime type`
2. Under "Hardware accelerator", select **T4 GPU**
3. Click `Save`

Without GPU, training will be extremely slow (hours instead of minutes).

## What the Test Does

The Colab notebook runs a **complete 50-step training session**:

1. **Setup**: Installs dependencies, clones your repo
2. **Data**: Downloads NeoCoder dataset (or uses test data)
3. **Model**: Loads GPT-2 (124M parameters)
4. **Training**: Runs 50 GRPO steps with actual gradient updates
5. **Analysis**: Shows training curves, statistics, generated code
6. **Results**: Downloads metrics and checkpoints

## Expected Results

Based on the mock test validation, you should see:

### Good Training Signs ✅
- **Rewards increasing** over time (e.g., -0.02 → +0.31)
- **Violations stable or decreasing** (model learning constraints)
- **Success rate stable** (20-40% is normal for GPT-2)
- **Loss decreasing** (policy converging)
- **Generated code is syntactically valid**

### Example Numbers (from Mock Test)
```
First 25 steps:  Reward = -0.025
Last 25 steps:   Reward = +0.312
Improvement:     +0.338 (+1350%)
```

### Warning Signs ⚠️
- Rewards staying completely flat (no learning)
- Success rate dropping to near 0% (model collapsing)
- All violations = 0 (detector broken)
- Generated code is nonsense or empty

## Runtime Expectations

With T4 GPU:
- **Setup**: 2-3 minutes (installing packages)
- **Model loading**: 3-5 minutes (downloading GPT-2)
- **Training**: 20-40 minutes (50 steps × ~30 sec/step)
- **Total**: 30-60 minutes

Each step processes:
- 2 problems × 4 solutions = 8 code generations
- Executes code in sandbox
- Computes rewards and advantages
- Updates model weights

## Configuration

The Colab config (`configs/config_colab.yaml`) uses:

```yaml
Model: GPT-2 (124M params)
Steps: 50
Batch size: 2 problems
Group size: 4 solutions per problem
Problems: 10 (subset of NeoCoder)
GPU: CUDA (Colab T4)
```

This is **scaled down** from the NSCC config:
- NSCC: CodeGen-1B (2.7B params), 5000 steps, 199 problems, A100 GPU
- Colab: GPT-2 (124M params), 50 steps, 10 problems, T4 GPU

## Interpreting Results

### If Training Looks Good ✅

**What this means:**
- All components working correctly
- Model can learn from RL training
- Denial prompting integration works
- Ready for full NSCC deployment

**Next steps:**
1. Review the training curves
2. Check generated code quality
3. Proceed to NSCC with confidence
4. Expect even better results with:
   - Bigger model (CodeGen-1B)
   - More steps (5000 vs 50)
   - More data (199 vs 10 problems)

### If Training Looks Bad ⚠️

**Possible issues:**
1. **GPU not enabled** → Check runtime settings
2. **Dataset issues** → Check if NeoCoder loaded correctly
3. **Hyperparameters** → May need tuning (rare)
4. **Random seed** → Try re-running with different seed

**Debugging:**
- Check the training logs for errors
- Verify GPU is actually being used
- Look at generated code - is it syntactically valid?
- Compare metrics to mock test baseline

## Comparison: Mock vs Colab vs NSCC

| Aspect | Mock Test | Colab Test | NSCC Training |
|--------|-----------|------------|---------------|
| **Model** | Simulated | GPT-2 (124M) | CodeGen-1B (2.7B) |
| **Data** | Pre-defined | Real NeoCoder | Real NeoCoder |
| **Learning** | Simulated | Real gradients | Real gradients |
| **Steps** | 20 | 50 | 5000 |
| **Problems** | 5 | 10 | 199 |
| **Hardware** | CPU | T4 GPU | A100 GPU |
| **Duration** | 30 sec | 30-60 min | 12-18 hours |
| **Cost** | Free | Free | $$$ |
| **Purpose** | Logic validation | System validation | Production training |

## After Colab Test

Once you've validated the system in Colab:

1. **Download results** (the notebook has a cell for this)
2. **Review metrics and plots**
3. **If successful**, proceed to NSCC:
   - Follow `nscc/README_DEPLOYMENT.md`
   - Use the full config (`configs/config_nscc.yaml`)
   - Submit job: `sbatch nscc/train_job.sh`
4. **If issues**, debug in Colab (faster iteration than NSCC)

## Tips

### Maximizing Colab Free Tier

- Colab gives ~12 hours of continuous GPU time
- Your 50-step test uses ~1 hour
- You can run multiple tests if needed
- Save checkpoints in case session disconnects

### Saving Results

The notebook automatically:
- Saves metrics to JSON
- Generates training curve plots
- Saves model checkpoints
- Creates downloadable zip file

### Troubleshooting

**"Runtime disconnected"**
- Colab times out after inactivity
- Re-run cells from the checkpoint

**"Out of memory"**
- Reduce batch_size to 1
- Reduce group_size to 2
- This shouldn't happen with GPT-2 on T4

**"Model download failing"**
- Check internet connection
- Try again (transient network issues)
- HuggingFace might be temporarily down

## Questions?

If Colab test fails or results are unclear:
1. Check the notebook outputs carefully
2. Compare to mock test baseline (PILOT_TEST_RESULTS.md)
3. Review error messages
4. Try with different random seed

## Ready for NSCC?

After successful Colab validation:
- ✅ System proven to work with real models
- ✅ Training loop validated
- ✅ Hyperparameters tested
- ✅ Confident for expensive NSCC run

See `nscc/README_DEPLOYMENT.md` for NSCC deployment instructions.
