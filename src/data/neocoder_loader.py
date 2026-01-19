"""
NeoCoder Dataset Loader

This module handles loading and parsing the NeoCoder dataset for RL training.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Problem:
    """Represents a single problem from NeoCoder."""
    problem_id: str
    rounds: List[Dict[str, Any]]  # Each round has: statement, constraints, code
    test_cases: List[Tuple[Any, Any]]  # (input, output) pairs

    def get_round(self, round_idx: int = 0) -> Dict[str, Any]:
        """
        Get a specific round of denial prompting.

        Args:
            round_idx: Round number (0 = original, 1+ = with constraints)

        Returns:
            Dictionary with 'statement', 'constraints', and 'code'
        """
        if round_idx >= len(self.rounds):
            # If requesting beyond available rounds, return last round
            round_idx = len(self.rounds) - 1

        return self.rounds[round_idx]

    def get_num_constraints(self, round_idx: int) -> int:
        """Get number of constraints for a given round."""
        if round_idx == 0:
            return 0  # Original problem has no constraints

        round_data = self.get_round(round_idx)
        constraints = round_data['constraints']

        # Filter out empty constraints
        return len([c for c in constraints if c.strip()])

    def get_prompt(self, round_idx: int = 0) -> str:
        """Get the problem statement for a specific round."""
        return self.get_round(round_idx)['statement']

    def get_constraints(self, round_idx: int = 0) -> List[str]:
        """Get the denied techniques for a specific round."""
        return self.get_round(round_idx)['constraints']


class NeoCoderLoader:
    """
    Loader for the NeoCoder dataset.

    Handles parsing the JSON files and creating Problem objects.
    """

    def __init__(self, data_dir: str = "data/raw/neocoder_repo/datasets/CodeForce/NeoCoder"):
        """
        Initialize the NeoCoder loader.

        Args:
            data_dir: Path to the NeoCoder dataset directory
        """
        self.data_dir = Path(data_dir)
        self.problems_file = self.data_dir / "NeoCoder.json"
        self.test_cases_file = self.data_dir / "test_cases_annotated.json"

        if not self.problems_file.exists():
            raise FileNotFoundError(
                f"NeoCoder dataset not found at {self.problems_file}. "
                "Run scripts/download_neocoder.py first."
            )

    def load(self) -> List[Problem]:
        """
        Load all problems from the NeoCoder dataset.

        Returns:
            List of Problem objects
        """
        # Load problems with denial prompting rounds
        with open(self.problems_file, 'r') as f:
            problems_data = json.load(f)

        # Load test cases
        with open(self.test_cases_file, 'r') as f:
            test_cases_data = json.load(f)

        # Create mapping from problem_id to test cases
        test_cases_map = {
            tc['problem_id']: tc
            for tc in test_cases_data
        }

        # Parse problems
        problems = []
        for prob_data in problems_data:
            problem_id = prob_data['problem_id']

            # Parse rounds (each round is a denial prompting iteration)
            rounds = []
            for i, (stmt, constraints, code) in enumerate(zip(
                prob_data['problem_statements'],
                prob_data['constraints_list'],
                prob_data['codes']
            )):
                rounds.append({
                    'round_idx': i,
                    'statement': stmt,
                    'constraints': constraints if isinstance(constraints, list) else [constraints],
                    'example_code': code,
                })

            # Parse test cases
            test_cases = []
            if problem_id in test_cases_map:
                tc = test_cases_map[problem_id]
                # NeoCoder format: inputs and outputs are lists
                for inp, out in zip(tc['input'], tc['output']):
                    test_cases.append((inp, out))

            problems.append(Problem(
                problem_id=problem_id,
                rounds=rounds,
                test_cases=test_cases,
            ))

        return problems

    def load_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[Problem], List[Problem], List[Problem]]:
        """
        Load and split the dataset into train/val/test.

        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_problems, val_problems, test_problems)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        # Load all problems
        problems = self.load()

        # Shuffle with fixed seed
        random.seed(seed)
        random.shuffle(problems)

        # Calculate split indices
        n = len(problems)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_problems = problems[:train_end]
        val_problems = problems[train_end:val_end]
        test_problems = problems[val_end:]

        return train_problems, val_problems, test_problems


class NeoCoderDataset:
    """
    PyTorch-style dataset for NeoCoder problems with curriculum learning.

    This dataset supports:
    - Curriculum learning (gradually increasing denial constraints)
    - Random sampling of problems
    - Flexible round selection
    """

    def __init__(
        self,
        problems: List[Problem],
        config: Optional[Dict] = None,
        current_step: int = 0,
        curriculum_schedule: Optional[List[Dict]] = None,
        seed: int = 42,
    ):
        """
        Initialize the dataset.

        Args:
            problems: List of Problem objects
            config: Optional config dict (if provided, extracts curriculum and seed)
            current_step: Current training step (for curriculum)
            curriculum_schedule: List of curriculum stages from config
            seed: Random seed
        """
        # If config is provided, extract values from it
        if config is not None and isinstance(config, dict):
            seed = config.get('data', {}).get('seed', seed)

            # Build curriculum schedule from config
            if config.get('curriculum', {}).get('enabled', False):
                warmup_steps = config['curriculum'].get('warmup_steps', 1000)
                max_constraints = config['curriculum'].get('max_constraints', 3)
                total_steps = config['training'].get('num_steps', 5000)

                # Build progressive schedule
                curriculum_schedule = []
                steps_per_stage = warmup_steps // max(1, max_constraints)

                for level in range(max_constraints + 1):
                    start_step = level * steps_per_stage
                    end_step = (level + 1) * steps_per_stage - 1 if level < max_constraints else total_steps

                    curriculum_schedule.append({
                        'stage': level + 1,
                        'steps': [start_step, end_step],
                        'num_constraints': level
                    })

        self.problems = problems
        self.current_step = current_step
        self.curriculum_schedule = curriculum_schedule or []
        self.rng = random.Random(seed)

    def set_step(self, step: int):
        """Update current training step (for curriculum learning)."""
        self.current_step = step

    def get_num_constraints(self) -> int:
        """
        Get number of constraints to use based on current step and curriculum.

        Returns:
            Number of constraints to deny
        """
        if not self.curriculum_schedule:
            return 0

        # Find which curriculum stage we're in
        for stage in self.curriculum_schedule:
            start_step, end_step = stage['steps']
            if start_step <= self.current_step <= end_step:
                return stage['num_constraints']

        # If beyond all stages, use last stage's constraints
        return self.curriculum_schedule[-1]['num_constraints']

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Sample a batch of problems.

        Args:
            batch_size: Number of problems to sample

        Returns:
            List of problem dictionaries, each containing:
            - problem_id: str
            - prompt: str (problem statement)
            - constraints: List[str] (denied techniques)
            - test_cases: List[Tuple] (input, output pairs)
            - round_idx: int
            - num_constraints: int
        """
        num_constraints = self.get_num_constraints()
        batch = []

        for _ in range(batch_size):
            # Sample a random problem
            problem = self.rng.choice(self.problems)

            # Determine which round to use based on num_constraints
            # Round 0 = no constraints
            # Round 1+ = with constraints
            if num_constraints == 0:
                round_idx = 0
            else:
                # Try to find a round with the target number of constraints
                # If not available, use the last round
                round_idx = min(num_constraints, len(problem.rounds) - 1)

            batch.append({
                'problem_id': problem.problem_id,
                'prompt': problem.get_prompt(round_idx),
                'constraints': problem.get_constraints(round_idx),
                'test_cases': problem.test_cases,
                'round_idx': round_idx,
                'num_constraints': num_constraints,
                'current_step': self.current_step,
            })

        return batch

    def __len__(self):
        """Return number of problems."""
        return len(self.problems)

    def __getitem__(self, idx):
        """Get a problem by index."""
        problem = self.problems[idx]
        num_constraints = self.get_num_constraints()
        round_idx = min(num_constraints, len(problem.rounds) - 1)

        return {
            'problem_id': problem.problem_id,
            'prompt': problem.get_prompt(round_idx),
            'constraints': problem.get_constraints(round_idx),
            'test_cases': problem.test_cases,
            'round_idx': round_idx,
            'num_constraints': num_constraints,
        }


if __name__ == "__main__":
    # Test the loader
    print("="*80)
    print("Testing NeoCoder Loader")
    print("="*80)

    try:
        loader = NeoCoderLoader()
        problems = loader.load()

        print(f"\n✅ Successfully loaded {len(problems)} problems")

        # Show first problem
        print("\n" + "="*80)
        print("First Problem Example")
        print("="*80)

        problem = problems[0]
        print(f"\nProblem ID: {problem.problem_id}")
        print(f"Number of rounds: {len(problem.rounds)}")
        print(f"Number of test cases: {len(problem.test_cases)}")

        # Show round 0 (original)
        print("\n--- Round 0 (Original Problem) ---")
        print(f"Prompt (first 200 chars): {problem.get_prompt(0)[:200]}...")
        print(f"Constraints: {problem.get_constraints(0)}")

        # Show round 1 (first denial)
        if len(problem.rounds) > 1:
            print("\n--- Round 1 (With Denial Constraints) ---")
            print(f"Prompt (first 200 chars): {problem.get_prompt(1)[:200]}...")
            print(f"Constraints: {problem.get_constraints(1)}")

        # Test train/val/test split
        print("\n" + "="*80)
        print("Testing Data Split")
        print("="*80)

        train, val, test = loader.load_split()
        print(f"\nTrain: {len(train)} problems")
        print(f"Val: {len(val)} problems")
        print(f"Test: {len(test)} problems")

        # Test curriculum dataset
        print("\n" + "="*80)
        print("Testing Curriculum Dataset")
        print("="*80)

        curriculum = [
            {'stage': 1, 'steps': [0, 100], 'num_constraints': 0},
            {'stage': 2, 'steps': [101, 200], 'num_constraints': 1},
        ]

        dataset = NeoCoderDataset(train[:10], curriculum_schedule=curriculum)

        # Step 50 (should use 0 constraints)
        dataset.set_step(50)
        print(f"\nStep 50: {dataset.get_num_constraints()} constraints")
        batch = dataset.sample_batch(2)
        print(f"Sampled {len(batch)} problems")
        print(f"First problem: {batch[0]['problem_id']}")

        # Step 150 (should use 1 constraint)
        dataset.set_step(150)
        print(f"\nStep 150: {dataset.get_num_constraints()} constraints")
        batch = dataset.sample_batch(2)
        print(f"First problem constraints: {batch[0]['constraints']}")

        print("\n" + "="*80)
        print("✅ All tests passed!")
        print("="*80)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure NeoCoder dataset is downloaded.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
