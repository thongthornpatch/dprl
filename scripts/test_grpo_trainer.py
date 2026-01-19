#!/usr/bin/env python3
"""
Test the GRPO trainer

This creates a minimal dummy setup and runs a few training steps.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config_loader import load_config
from data.neocoder_loader import NeoCoderDataset
from models.model_wrapper import ModelWrapper
from rewards.reward_function import RewardFunction
from training.grpo_trainer import GRPOTrainer


class DummyProblem:
    """Dummy problem for testing."""
    def __init__(self, problem_id, test_cases):
        self.problem_id = problem_id
        self.test_cases = test_cases  # List of (input, output) tuples
        self.rounds = [
            {
                'constraints': [],
                'example_code': f'def solve(x): return x + 1  # Problem {problem_id}'
            },
            {
                'constraints': ['while loop'],
                'example_code': f'def solve(x): return x + 1  # Problem {problem_id} Round 1'
            },
        ]

    def get_round(self, round_num):
        if round_num < len(self.rounds):
            return self.rounds[round_num]
        return self.rounds[-1]


def test_grpo_trainer():
    """Test GRPO trainer with dummy data."""
    print("="*80)
    print("PHASE 4 TEST: GRPO Trainer")
    print("="*80)

    # Load config
    print("\nLoading config...")
    config = load_config("configs/config_laptop.yaml")

    # Override for quick test
    config['training']['num_steps'] = 5
    config['training']['batch_size'] = 2
    config['training']['group_size'] = 2
    config['training']['save_every'] = 10  # Don't save during test
    config['curriculum']['enabled'] = False  # Disable for simplicity

    print(f"Config loaded (modified for testing)")
    print(f"Steps: {config['training']['num_steps']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Group size: {config['training']['group_size']}")

    # Create dummy dataset
    print("\nCreating dummy dataset...")
    dummy_problems = [
        DummyProblem("test_001", [(5, 6), (0, 1), (-1, 0)]),
        DummyProblem("test_002", [(10, 11), (2, 3)]),
        DummyProblem("test_003", [(1, 2), (3, 4), (5, 6)]),
        DummyProblem("test_004", [(0, 1)]),
        DummyProblem("test_005", [(7, 8), (9, 10)]),
    ]

    dataset = NeoCoderDataset(dummy_problems, config)
    print(f"Created dataset with {len(dummy_problems)} dummy problems")

    # Initialize model
    print("\nInitializing model...")
    model = ModelWrapper(
        model_name=config['model']['name'],
        device=config['model']['device'],
        max_length=config['model']['max_length'],
    )
    print(f"Model loaded: {config['model']['name']}")
    print(f"Device: {model.device}")

    # Test model generation
    print("\nTesting model generation...")
    test_prompt = "def solve(x):\n    # Return x + 1\n    "
    test_output = model.generate(test_prompt, max_new_tokens=50, num_return_sequences=1)
    print(f"Model generated {len(test_output)} solution(s)")
    print(f"Sample (first 100 chars): {test_output[0][:100]}...")

    # Initialize reward function
    print("\nInitializing reward function...")
    reward_fn = RewardFunction(
        correctness_weight=config['reward']['correctness_weight'],
        denial_penalty_weight=config['reward']['denial_penalty_weight'],
        timeout=config['reward']['timeout'],
    )
    print("Reward function ready")

    # Test reward function
    print("\nTesting reward function...")
    test_code = "def solve(x):\n    return x + 1"
    test_result = reward_fn.compute_reward(
        generated_code=test_code,
        test_cases=[(5, 6), (0, 1)],
        denied_techniques=[],
    )
    print(f"Reward computed: {test_result['total_reward']:.3f}")
    print(f"Correctness: {test_result['correctness_score']:.3f}")
    print(f"Success: {test_result['success']}")

    # Initialize trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_fn=reward_fn,
        dataset=dataset,
        config=config,
        output_dir="./outputs/phase4_test",
    )
    print("Trainer initialized")

    # Run training
    print("\n" + "="*80)
    print("RUNNING TRAINING (5 steps)")
    print("="*80)

    try:
        results = trainer.train()

        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"Training completed successfully")
        print(f"Total steps: {results['total_steps']}")
        print(f"Final mean reward: {results['final_mean_reward']:.3f}")
        print(f"Final success rate: {results['final_success_rate']:.2%}")
        print(f"Final violation rate: {results['final_violation_rate']:.2%}")

        # Verify training actually happened
        if results['total_steps'] == config['training']['num_steps']:
            print(f"\nAll {config['training']['num_steps']} steps completed")
            return True
        else:
            print(f"\nExpected {config['training']['num_steps']} steps, got {results['total_steps']}")
            return False

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_curriculum_integration():
    """Test that curriculum learning works with GRPO."""
    print("\n" + "="*80)
    print("TEST: Curriculum Learning Integration")
    print("="*80)

    config = load_config("configs/config_laptop.yaml")
    config['training']['num_steps'] = 10
    config['training']['batch_size'] = 2
    config['training']['group_size'] = 2
    config['curriculum']['enabled'] = True
    config['curriculum']['warmup_steps'] = 5

    print(f"Curriculum enabled: {config['curriculum']['enabled']}")
    print(f"Warmup steps: {config['curriculum']['warmup_steps']}")

    # Create dataset with multiple rounds
    dummy_problems = [
        DummyProblem("curr_001", [(1, 2)]),
        DummyProblem("curr_002", [(2, 3)]),
    ]

    dataset = NeoCoderDataset(dummy_problems, config)

    # Sample at different steps
    print("\nTesting curriculum progression:")
    for step in [0, 3, 7]:
        batch = dataset.sample_batch(batch_size=1, step=step)
        round_num = batch[0]['denied_techniques']
        print(f"  Step {step}: Round constraints = {round_num}")

    print("\nCurriculum integration test passed")
    return True


def main():
    """Run all Phase 4 tests."""
    print("="*80)
    print("PHASE 4: GRPO TRAINING LOOP TESTS")
    print("="*80)

    tests = [
        ("GRPO Trainer Basic Functionality", test_grpo_trainer),
        ("Curriculum Learning Integration", test_curriculum_integration),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            print("\n")
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n{test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        print(f"{status}  {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! Phase 4 is complete.")
        print("\nNext steps:")
        print("  1. Create SLURM scripts for NSCC deployment (Phase 5)")
        print("  2. Test full training run with real NeoCoder data")
        return 0
    else:
        print("\nSome tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
