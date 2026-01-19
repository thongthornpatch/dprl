# Denial Prompting for RL training

## 📁 Project Structure
```
denial_prompting_RL/
├── configs/                 
│   ├── config_laptop.yaml  # Laptop testing config (CPU)
│   └── config_nscc.yaml    # NSCC training config (GPU)
├── src/
│   ├── data/               
│   ├── models/          
│   ├── training/           # GRPO training
│   ├── rewards/            # Reward function implementation (Reward = Correctness - (num_violations × penalty_weight))
│   ├── evaluation/         # Pass@k and NeoGauge metrics
│   └── utils/             
├── data/
│   ├── raw/                # Raw NeoCoder dataset
│   └── processed/          # Preprocessed data
├── scripts/                # Executable scripts
├── notebooks/              # Jupyter notebooks for analysis
├── experiments/            # Saved experiment results
├── logs/                   # Training logs
└── requirements.txt        
```
