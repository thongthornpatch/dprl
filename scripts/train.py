#!/usr/bin/env python3
"""
Main training script for Denial Prompting RL

Usage:
    python scripts/train.py --config configs/config_laptop.yaml
    python scripts/train.py --config configs/config_nscc.yaml --output_dir ./outputs/run1
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config_loader import load_config
from data.neocoder_loader import NeoCoderLoader, NeoCoderDataset
from models.model_wrapper import ModelWrapper
from rewards.reward_function import RewardFunction
from training.grpo_trainer import GRPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train denial prompting RL model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/config_laptop.yaml)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/default_run",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    print("="*80)
    print("DENIAL PROMPTING RL TRAINING")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # Load configuration
    print("\nLoading configuration...")
    config = load_config(args.config)
    print(f"Config loaded")
    print(f"Model: {config['model']['name']}")
    print(f"Device: {config['model']['device']}")
    print(f"Training steps: {config['training']['num_steps']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Group size: {config['training']['group_size']}")

    # Load dataset
    print("\nLoading NeoCoder dataset...")
    try:
        loader = NeoCoderLoader()
        problems = loader.load()
        dataset = NeoCoderDataset(problems, config)
        print(f"Loaded {len(problems)} problems")
        print(f"Total rounds: {sum(len(p.rounds) for p in problems)}")
        print(f"Curriculum enabled: {config['curriculum']['enabled']}")
    except FileNotFoundError as e:
        print(f"Error: NeoCoder dataset not found")
        print(f"   {e}")
        print("\nPlease download the dataset first:")
        print("   python scripts/download_neocoder.py")
        return 1

    # Initialize model
    print("\nInitializing model...")
    model = ModelWrapper(
        model_name=config['model']['name'],
        device=config['model']['device'],
        max_length=config['model']['max_length'],
        torch_dtype=config['model'].get('torch_dtype', 'float32'),
    )
    print(f"Model loaded: {config['model']['name']}")
    print(f"Device: {model.device}")
    print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        model.load(args.resume_from)
        print("Checkpoint loaded")

    # Initialize reward function
    print("\nInitializing reward function...")
    reward_fn = RewardFunction(
        correctness_weight=float(config['reward']['correctness_weight']),
        denial_penalty_weight=float(config['reward']['denial_penalty_weight']),
        timeout=int(config['reward']['timeout']),
    )
    print("Reward function ready")
    print(f"Correctness weight: {config['reward']['correctness_weight']}")
    print(f"Denial penalty weight: {config['reward']['denial_penalty_weight']}")

    # Initialize trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_fn=reward_fn,
        dataset=dataset,
        config=config,
        output_dir=args.output_dir,
    )
    print("Trainer initialized")

    # Start training
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    try:
        results = trainer.train()

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Total steps: {results['total_steps']}")
        print(f"Final mean reward: {results['final_mean_reward']:.3f}")
        print(f"Final success rate: {results['final_success_rate']:.2%}")
        print(f"Final violation rate: {results['final_violation_rate']:.2%}")
        print(f"\nOutputs saved to: {args.output_dir}")
        print("="*80)

        return 0

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print(f"Outputs saved to: {args.output_dir}")
        return 1

    except Exception as e:
        print(f"\n\nTraining failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
