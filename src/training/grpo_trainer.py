"""
GRPO Trainer for Denial Prompting RL

Implements Group Relative Policy Optimization (GRPO) for code generation.

GRPO Algorithm:
1. For each problem, generate N solutions (group)
2. Compute reward for each solution
3. Compute group-relative advantage: A_i = R_i - mean(R_group)
4. Update policy using advantages (positive advantage → increase prob, negative → decrease)

This is simpler than PPO because we don't need a critic model!
"""

import os
import json
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import time

# Handle both package and standalone imports
try:
    from ..models.model_wrapper import ModelWrapper
    from ..rewards.reward_function import RewardFunction
    from ..data.neocoder_loader import NeoCoderDataset
    from ..utils.logging_utils import MetricsTracker
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.model_wrapper import ModelWrapper
    from rewards.reward_function import RewardFunction
    from data.neocoder_loader import NeoCoderDataset
    from utils.logging_utils import MetricsTracker


class GRPOTrainer:
    """
    GRPO Trainer for Denial Prompting RL.

    Args:
        model: ModelWrapper instance
        reward_fn: RewardFunction instance
        dataset: NeoCoderDataset instance
        config: Training configuration dictionary
        output_dir: Directory to save checkpoints and logs
    """

    def __init__(
        self,
        model: ModelWrapper,
        reward_fn: RewardFunction,
        dataset: NeoCoderDataset,
        config: Dict[str, Any],
        output_dir: str = "./outputs",
    ):
        """Initialize GRPO trainer."""
        self.model = model
        self.reward_fn = reward_fn
        self.dataset = dataset
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training hyperparameters (convert to proper types)
        self.num_steps = int(config['training']['num_steps'])
        self.group_size = int(config['training']['group_size'])  # N solutions per problem
        self.learning_rate = float(config['training']['learning_rate'])
        self.clip_range = float(config['training'].get('clip_range', 0.2))
        self.batch_size = int(config['training']['batch_size'])

        # Generation parameters (convert to proper types)
        self.max_new_tokens = int(config['generation']['max_new_tokens'])
        self.temperature = float(config['generation']['temperature'])
        self.top_p = float(config['generation']['top_p'])

        # Curriculum learning is handled by dataset via set_step()
        self.use_curriculum = config['curriculum']['enabled']

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=self.learning_rate,
        )

        # Metrics tracking
        self.metrics = MetricsTracker()

        # Training state
        self.global_step = 0

    def train(self) -> Dict[str, Any]:
        """
        Run GRPO training loop.

        Returns:
            Dictionary with training results and final metrics
        """
        print("="*80)
        print("STARTING GRPO TRAINING")
        print("="*80)
        print(f"Total steps: {self.num_steps}")
        print(f"Group size: {self.group_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Curriculum learning: {self.use_curriculum}")
        print("="*80)

        # Training loop
        pbar = tqdm(range(self.num_steps), desc="Training")
        for step in pbar:
            self.global_step = step

            # Update dataset's current step for curriculum learning
            self.dataset.set_step(step)

            # Sample batch of problems (dataset handles curriculum internally)
            batch = self.dataset.sample_batch(
                batch_size=self.batch_size,
            )

            # Training step
            step_metrics = self._training_step(batch)

            # Log metrics
            self.metrics.log(step, step_metrics)

            # Update progress bar
            pbar.set_postfix({
                'reward': f"{step_metrics['mean_reward']:.3f}",
                'loss': f"{step_metrics['loss']:.3f}",
                'violations': f"{step_metrics['mean_violations']:.2f}",
            })

            # Save checkpoint
            if (step + 1) % self.config['training'].get('save_every', 500) == 0:
                self._save_checkpoint(step)

        # Final checkpoint
        self._save_checkpoint(self.num_steps - 1, is_final=True)

        # Compute final statistics
        final_metrics = self._compute_final_metrics()

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Final mean reward: {final_metrics['final_mean_reward']:.3f}")
        print(f"Final success rate: {final_metrics['final_success_rate']:.2%}")
        print(f"Final violation rate: {final_metrics['final_violation_rate']:.2%}")
        print("="*80)

        return final_metrics

    def _training_step(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute one GRPO training step.

        Args:
            batch: List of problem dictionaries

        Returns:
            Dictionary with step metrics
        """
        # Step 1: Generate N solutions per problem (group)
        all_solutions = []
        all_prompts = []
        all_problems = []

        for problem_data in batch:
            prompt = problem_data['prompt']

            # Generate group_size solutions for this problem
            solutions = self.model.generate(
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                num_return_sequences=self.group_size,
            )

            all_solutions.extend(solutions)
            all_prompts.extend([prompt] * self.group_size)
            all_problems.extend([problem_data] * self.group_size)

        # Step 2: Compute rewards for all solutions
        rewards = []
        violations = []
        successes = []

        for solution, problem_data in zip(all_solutions, all_problems):
            reward_result = self.reward_fn.compute_reward(
                generated_code=solution,
                test_cases=problem_data['test_cases'],
                denied_techniques=problem_data['constraints'],  # Dataset returns 'constraints'
            )

            rewards.append(reward_result['total_reward'])
            violations.append(reward_result['num_violations'])
            successes.append(reward_result['success'])

        # Step 3: Compute group-relative advantages
        advantages = self._compute_group_advantages(rewards, self.group_size)

        # Step 4: Compute policy loss and update
        loss = self._compute_policy_loss(
            prompts=all_prompts,
            solutions=all_solutions,
            advantages=advantages,
        )

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.model.parameters(),
            max_norm=self.config['training'].get('max_grad_norm', 1.0)
        )

        self.optimizer.step()

        # Collect metrics
        metrics = {
            'loss': loss.item(),
            'mean_reward': sum(rewards) / len(rewards),
            'mean_violations': sum(violations) / len(violations),
            'success_rate': sum(successes) / len(successes),
            'max_reward': max(rewards),
            'min_reward': min(rewards),
        }

        return metrics

    def _compute_group_advantages(
        self,
        rewards: List[float],
        group_size: int,
    ) -> torch.Tensor:
        """
        Compute group-relative advantages.

        For each solution in a group, advantage = reward - mean(group_rewards)

        Args:
            rewards: List of rewards (length = batch_size * group_size)
            group_size: Number of solutions per problem

        Returns:
            Tensor of advantages
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        # Reshape to [batch_size, group_size]
        num_problems = len(rewards) // group_size
        rewards_grouped = rewards_tensor.view(num_problems, group_size)

        # Compute mean reward per group (baseline)
        group_means = rewards_grouped.mean(dim=1, keepdim=True)

        # Advantage = reward - baseline
        advantages_grouped = rewards_grouped - group_means

        # Flatten back to [batch_size * group_size]
        advantages = advantages_grouped.view(-1)

        # Normalize advantages (optional but helps stability)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def _compute_policy_loss(
        self,
        prompts: List[str],
        solutions: List[str],
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GRPO policy loss.

        Loss = -mean(log_prob(solution | prompt) * advantage)

        With PPO-style clipping for stability.

        Args:
            prompts: List of prompts
            solutions: List of generated solutions
            advantages: Tensor of advantages

        Returns:
            Scalar loss tensor
        """
        # Tokenize prompts and solutions
        full_texts = [p + s for p, s in zip(prompts, solutions)]

        # Get model log probabilities
        # This is a simplified version - in practice you'd want to:
        # 1. Store old log probs during generation
        # 2. Compute new log probs here
        # 3. Use ratio = exp(new_log_prob - old_log_prob)
        # 4. Apply PPO clipping

        # For now, we'll do a simplified version:
        log_probs = self._compute_log_probs(prompts, solutions)

        # Move advantages to same device
        advantages = advantages.to(self.model.device)

        # Policy gradient loss with clipping
        # Note: In full GRPO, you'd compute ratio and clip
        # Simplified version: just use log_prob * advantage
        policy_loss = -(log_probs * advantages).mean()

        return policy_loss

    def _compute_log_probs(
        self,
        prompts: List[str],
        solutions: List[str],
    ) -> torch.Tensor:
        """
        Compute log probabilities of solutions given prompts.

        This is a simplified implementation. In practice, you'd want to:
        1. Tokenize properly with attention masks
        2. Only compute log probs for solution tokens (not prompt tokens)
        3. Handle batching efficiently

        Args:
            prompts: List of prompts
            solutions: List of solutions

        Returns:
            Tensor of log probabilities
        """
        log_probs = []

        # Process each solution individually to get per-sample log probs
        for prompt, solution in zip(prompts, solutions):
            # Tokenize full sequence (prompt + solution)
            full_text = prompt + solution
            encodings = self.model.tokenizer(
                full_text,
                truncation=True,
                max_length=self.model.max_length,
                return_tensors="pt",
            ).to(self.model.device)

            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = self.model.model(**encodings, labels=encodings['input_ids'])

                # Get log probability (negative loss)
                # Note: This is simplified - ideally only compute for solution tokens
                log_prob = -outputs.loss
                log_probs.append(log_prob)

        return torch.stack(log_probs)

    def _save_checkpoint(self, step: int, is_final: bool = False):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if is_final:
            checkpoint_path = checkpoint_dir / "final_model"
        else:
            checkpoint_path = checkpoint_dir / f"step_{step}"

        # Save model
        self.model.save(checkpoint_path)

        # Save training state
        state = {
            'global_step': step,
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
        }

        with open(checkpoint_path / "training_state.json", 'w') as f:
            # Can't save optimizer state in JSON, so skip it
            json.dump({
                'global_step': state['global_step'],
                'config': state['config'],
            }, f, indent=2)

        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        self.metrics.save(str(metrics_path))

    def _compute_final_metrics(self) -> Dict[str, Any]:
        """Compute final training statistics."""
        # Get last 100 steps of metrics
        recent_steps = list(self.metrics.history.keys())[-100:]

        if not recent_steps:
            return {
                'final_mean_reward': 0.0,
                'final_success_rate': 0.0,
                'final_violation_rate': 0.0,
            }

        recent_rewards = [
            self.metrics.history[step]['mean_reward']
            for step in recent_steps
        ]
        recent_success = [
            self.metrics.history[step]['success_rate']
            for step in recent_steps
        ]
        recent_violations = [
            self.metrics.history[step]['mean_violations']
            for step in recent_steps
        ]

        return {
            'final_mean_reward': sum(recent_rewards) / len(recent_rewards),
            'final_success_rate': sum(recent_success) / len(recent_success),
            'final_violation_rate': sum(v > 0 for v in recent_violations) / len(recent_violations),
            'total_steps': self.global_step + 1,
        }


if __name__ == "__main__":
    # Test GRPO trainer with dummy setup
    print("="*80)
    print("Testing GRPO Trainer (Dummy Run)")
    print("="*80)

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from utils.config_loader import load_config
    from data.neocoder_loader import NeoCoderLoader

    # Load config
    config = load_config("configs/config_laptop.yaml")

    # Override for quick test
    config['training']['num_steps'] = 10
    config['training']['batch_size'] = 2
    config['training']['group_size'] = 2

    print("\n📋 Loading dataset...")
    try:
        loader = NeoCoderLoader()
        problems = loader.load()
        dataset = NeoCoderDataset(problems, config)
        print(f"✅ Loaded {len(problems)} problems")
    except FileNotFoundError:
        print("⚠️  NeoCoder dataset not found, creating dummy dataset")
        # Create dummy dataset for testing
        class DummyProblem:
            def __init__(self):
                self.problem_id = "test_001"
                self.test_cases = [(1, 2), (2, 3)]
                self.rounds = [
                    {'constraints': [], 'example_code': 'def solve(x): return x+1'}
                ]
            def get_round(self, r):
                return self.rounds[0]

        problems = [DummyProblem() for _ in range(10)]
        dataset = NeoCoderDataset(problems, config)

    print("\n🤖 Initializing model...")
    model = ModelWrapper(
        model_name=config['model']['name'],
        device=config['model']['device'],
        max_length=config['model']['max_length'],
    )
    print(f"✅ Loaded model: {config['model']['name']}")

    print("\n🎯 Initializing reward function...")
    reward_fn = RewardFunction(
        correctness_weight=config['reward']['correctness_weight'],
        denial_penalty_weight=config['reward']['denial_penalty_weight'],
        timeout=config['reward']['timeout'],
    )
    print("✅ Reward function ready")

    print("\n🏋️  Initializing trainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_fn=reward_fn,
        dataset=dataset,
        config=config,
        output_dir="./outputs/test_run",
    )
    print("✅ Trainer ready")

    print("\n🚀 Starting training (10 steps)...")
    results = trainer.train()

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    for key, value in results.items():
        print(f"{key}: {value}")

    print("\n✅ Test complete!")
