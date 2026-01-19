#!/usr/bin/env python3
"""
Test the complete data pipeline for Phase 2.

This script tests:
1. NeoCoder dataset loading
2. Train/val/test splitting
3. Curriculum learning
4. Denial prompting augmentation
5. Batch sampling

Usage:
    python scripts/test_data_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.neocoder_loader import NeoCoderLoader, NeoCoderDataset
from data.denial_prompting import CurriculumScheduler, DenialPromptAugmenter
from utils.config_loader import load_config


def test_basic_loading():
    """Test 1: Basic dataset loading."""
    print("\n" + "="*80)
    print("TEST 1: Basic Dataset Loading")
    print("="*80)

    try:
        loader = NeoCoderLoader()
        problems = loader.load()

        print(f"✅ Loaded {len(problems)} problems")

        # Check first problem
        problem = problems[0]
        print(f"✅ First problem ID: {problem.problem_id}")
        print(f"✅ Number of rounds: {len(problem.rounds)}")
        print(f"✅ Number of test cases: {len(problem.test_cases)}")

        # Verify structure
        assert len(problems) == 199, f"Expected 199 problems, got {len(problems)}"
        assert len(problem.rounds) >= 2, "Problem should have at least 2 rounds"
        assert len(problem.test_cases) > 0, "Problem should have test cases"

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_train_val_test_split():
    """Test 2: Train/val/test splitting."""
    print("\n" + "="*80)
    print("TEST 2: Train/Val/Test Split")
    print("="*80)

    try:
        loader = NeoCoderLoader()
        train, val, test = loader.load_split(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42
        )

        print(f"✅ Train set: {len(train)} problems")
        print(f"✅ Val set: {len(val)} problems")
        print(f"✅ Test set: {len(test)} problems")

        total = len(train) + len(val) + len(test)
        assert total == 199, f"Split doesn't preserve total: {total}"

        # Check no overlap
        train_ids = {p.problem_id for p in train}
        val_ids = {p.problem_id for p in val}
        test_ids = {p.problem_id for p in test}

        assert len(train_ids & val_ids) == 0, "Train and val overlap"
        assert len(train_ids & test_ids) == 0, "Train and test overlap"
        assert len(val_ids & test_ids) == 0, "Val and test overlap"

        print(f"✅ No overlap between splits")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_curriculum_dataset():
    """Test 3: Curriculum learning dataset."""
    print("\n" + "="*80)
    print("TEST 3: Curriculum Learning Dataset")
    print("="*80)

    try:
        loader = NeoCoderLoader()
        train, _, _ = loader.load_split()

        # Create curriculum schedule
        curriculum = [
            {'stage': 1, 'steps': [0, 100], 'num_constraints': 0},
            {'stage': 2, 'steps': [101, 200], 'num_constraints': 1},
            {'stage': 3, 'steps': [201, 300], 'num_constraints': 2},
        ]

        dataset = NeoCoderDataset(train[:20], curriculum_schedule=curriculum)

        # Test at step 50 (stage 1: 0 constraints)
        dataset.set_step(50)
        num_constraints = dataset.get_num_constraints()
        print(f"✅ Step 50: {num_constraints} constraints (expected 0)")
        assert num_constraints == 0, f"Expected 0 constraints at step 50"

        # Test at step 150 (stage 2: 1 constraint)
        dataset.set_step(150)
        num_constraints = dataset.get_num_constraints()
        print(f"✅ Step 150: {num_constraints} constraints (expected 1)")
        assert num_constraints == 1, f"Expected 1 constraint at step 150"

        # Test at step 250 (stage 3: 2 constraints)
        dataset.set_step(250)
        num_constraints = dataset.get_num_constraints()
        print(f"✅ Step 250: {num_constraints} constraints (expected 2)")
        assert num_constraints == 2, f"Expected 2 constraints at step 250"

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_batch_sampling():
    """Test 4: Batch sampling."""
    print("\n" + "="*80)
    print("TEST 4: Batch Sampling")
    print("="*80)

    try:
        loader = NeoCoderLoader()
        train, _, _ = loader.load_split()

        curriculum = [
            {'stage': 1, 'steps': [0, 100], 'num_constraints': 0},
            {'stage': 2, 'steps': [101, 200], 'num_constraints': 1},
        ]

        dataset = NeoCoderDataset(train[:20], curriculum_schedule=curriculum, seed=42)
        dataset.set_step(150)  # Stage 2: 1 constraint

        # Sample a batch
        batch = dataset.sample_batch(batch_size=4)

        print(f"✅ Sampled batch of {len(batch)} problems")

        # Check batch structure
        for i, item in enumerate(batch):
            assert 'problem_id' in item
            assert 'prompt' in item
            assert 'constraints' in item
            assert 'test_cases' in item
            assert 'round_idx' in item

            print(f"  Problem {i+1}: {item['problem_id']}")
            print(f"    Round: {item['round_idx']}")
            print(f"    Constraints: {item['constraints'][:2] if item['constraints'] else 'None'}...")
            print(f"    Test cases: {len(item['test_cases'])}")

        print(f"✅ All batch items have correct structure")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_denial_augmentation():
    """Test 5: Denial prompting augmentation."""
    print("\n" + "="*80)
    print("TEST 5: Denial Prompting Augmentation")
    print("="*80)

    try:
        augmenter = DenialPromptAugmenter()

        original_prompt = "Write a function to calculate the factorial of a number."
        constraints = ["for loop", "while loop", "math.factorial()"]

        augmented = augmenter.augment(original_prompt, constraints)

        print(f"✅ Original prompt:\n  {original_prompt}")
        print(f"\n✅ Augmented prompt:\n{augmented}")

        # Verify augmentation
        assert "DO NOT use" in augmented
        assert "for loop" in augmented
        assert original_prompt in augmented

        print(f"\n✅ Augmentation successful")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_config_integration():
    """Test 6: Integration with config files."""
    print("\n" + "="*80)
    print("TEST 6: Config Integration")
    print("="*80)

    try:
        # Load laptop config
        config = load_config('configs/config_laptop.yaml')

        print(f"✅ Loaded config: {config['experiment_name']}")

        # Create dataset using config
        loader = NeoCoderLoader()
        problems = loader.load()

        # Use config's num_problems limit
        num_problems = config['data']['num_problems']
        problems_subset = problems[:num_problems]

        print(f"✅ Using {len(problems_subset)} problems (config limit: {num_problems})")

        # Create curriculum from config
        curriculum = config['denial_prompting']['curriculum_schedule']
        print(f"✅ Curriculum has {len(curriculum)} stages")

        scheduler = CurriculumScheduler(curriculum)
        print(f"\nCurriculum summary:")
        for stage in curriculum:
            start, end = stage['steps']
            print(f"  Stage {stage['stage']}: Steps {start}-{end} → {stage['num_constraints']} constraints")

        # Create dataset
        dataset = NeoCoderDataset(
            problems_subset,
            curriculum_schedule=curriculum,
            seed=config['data']['seed']
        )

        print(f"\n✅ Dataset created with {len(dataset)} problems")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("PHASE 2: DATA PIPELINE TESTS")
    print("="*80)

    tests = [
        ("Basic Loading", test_basic_loading),
        ("Train/Val/Test Split", test_train_val_test_split),
        ("Curriculum Dataset", test_curriculum_dataset),
        ("Batch Sampling", test_batch_sampling),
        ("Denial Augmentation", test_denial_augmentation),
        ("Config Integration", test_config_integration),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results[test_name] = False

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}  {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Phase 2 is complete.")
        print("\nNext steps:")
        print("  1. Implement reward function (Phase 3)")
        print("  2. Implement GRPO training loop (Phase 4)")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
